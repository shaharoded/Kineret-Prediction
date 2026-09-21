import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import math
from pathlib import Path
from tqdm.auto import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from kineret.intervene.dataset import EMRTokenizer
from kineret.intervene.utils import set_seed
from kineret.intervene.config.model_config import SEED
from kineret.intervene.config.model_config import *
from kineret.intervene.config.dataset_config import OUTCOMES, TERMINAL_OUTCOMES
from kineret.intervene.utils import compute_legality_masks_tf, get_temporal_multi_hot_targets, plot_losses, build_luts, logger
from kineret.intervene.schedulers import LambdaScheduleController

torch.serialization.add_safe_globals([
    EMRTokenizer,
    sklearn.preprocessing.StandardScaler,
    numpy._core.multiarray.scalar,
    numpy._core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
    numpy.dtypes.Int64DType,
    numpy.dtypes.Float64DType,
    numpy.float64,
    numpy.int64,
    numpy.int32,
    numpy.float32,
    numpy.bool_,
    numpy.ufunc,
])


class Time2Vec(nn.Module):
    """
    Time2Vec layer for encoding continuous time intervals (Δt) into fixed-size vectors.

    This layer captures both linear trends and periodic patterns in the time between events.

    Output dimensions:
        - 1 dimension for linear time progression (trend)
        - k-1 dimensions for periodic representation using learnable sine functions

    Args:
        out_dim (int): Output dimensionality of the time embedding (must be >= 2)

    Input:
        t (Tensor): FloatTensor of shape [B, T] or [B*T], representing time deltas in days (as float representing partial days too)

    Output:
        Tensor of shape [B, T, out_dim] or [B*T, out_dim], where out_dim = 1 + k
    """

    def __init__(self, out_dim):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >= 2")
        self.linear = nn.Linear(1, 1, bias=True)          # ω0 t + b0 -> Linear component
        self.freq   = nn.Linear(1, out_dim - 1, bias=True) # ω_i t + b_i -> Sinusiodal component

        # Log-spaced periodic-frequency init: input t is hours/336, so values
        # are O(1e-3)–O(1) across the dataset. With default Linear init the
        # frequencies cluster in [-1, 1] rad/unit, which puts sin(ω·t) into the
        # quasi-linear regime for every realistic Δt → no per-event resolution.
        # Anchor ωs on a log grid that resolves 5 min (ω≈25k) through 1 week
        # (ω≈12), so the basis spans the inter-event timescales actually present
        # in the data.  Biases stay at their default (small uniform) so each
        # channel still gets an independent random phase.
        with torch.no_grad():
            k = out_dim - 1
            freqs = torch.logspace(math.log10(12.0), math.log10(25000.0), k)
            sign  = torch.where(torch.arange(k) % 2 == 0, 1.0, -1.0)
            self.freq.weight.copy_((freqs * sign).unsqueeze(-1))

    def forward(self, t):
        """
        t: [B, T] or [B*T] float tensor of time deltas
        Returns: [B, T, out_dim] or [B*T, out_dim]
        """
        t = t.unsqueeze(-1)                     # [B, T, 1]
        linear_out = self.linear(t)             # [B, T, 1]
        periodic_out = torch.sin(self.freq(t))  # [B, T, k-1]
        return torch.cat([linear_out, periodic_out], dim=-1)


class EMREmbedding(nn.Module):
    """
    Embedding layer for electronic medical record (EMR) sequences.

    This module creates time-aware, context-enhanced event representations suitable
    for Transformer-based models. It replaces traditional token and positional embeddings
    by explicitly decomposing each event into structured components:
      - Parent Raw Concept IDs (e.g., "GLUCOSE") - All source raw concepts for this Concept
      - Concept ID (e.g., "GLUCOSE_STATE")
      - Concept + Value ID (e.g., "GLUCOSE_STATE_Low")
      - Concept + Value + Position ID (e.g., "GLUCOSE_STATE_Low_START")
      - Absolute time since admission (Δt abs)
      - Patient-level context vector (e.g., age, sex, No. prior admissions in 6 months...)

    These components are embedded, concatenated, and projected into a shared
    fixed-size embedding space. A [CTX] projection is added to each sequence to incorporate patient-level context.

    All embeddings are regularized with dropout and normalized with LayerNorm.

    In addeition, a tiny MLP head (self.time_head) is attached solely for Phase-1 pre-training.  
    It encourages Time2Vec to encode absolute-time information by regressing the normalised Δt  ∈ [0,1].  
    The head is ignored / discarded in Phase-2.

    Args:
        tokenizer (EMRTokenizer): The tokenizer object managing vocabularies and token metadata.
        ctx_dim (int): Dimensionality of the patient context vector.
        time2vec_dim (int): Output dimension of each Time2Vec component (must be ≥ 2).
        embed_dim (int): Final embedding size for each event.
        dropout (float): Dropout rate applied to the combined embeddings.

    Attributes:
        tokenizer (EMRTokenizer): Stores the vocab and special token mappings.
        decoder (nn.Linear): Tied to the position embedding for predicting next token.
        output_dim (int): Final embedding size (matches `embed_dim`).
        padding_idx (int): Token index reserved for padding ([PAD]).
    """

    def __init__(self, tokenizer, ctx_dim, time2vec_dim=8, embed_dim=128, dropout=0.1):
        super().__init__()
        set_seed(SEED)  # reproducible weight init (constructed in immutable api.py before train fns run)

        # Public attributes consumed by InterveneEncoder, training loops, inference and diagnose.
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.pad_token_id
        self.output_dim = embed_dim

        # --- Token-level embeddings ---
        self.raw_concept_embed = nn.Embedding(len(tokenizer.rawconcept2id), embed_dim) # Embed for "GLUCOSE_MEASURE"
        self.concept_embed = nn.Embedding(len(tokenizer.concept2id), embed_dim) # Embed for "GLUCOSE_MEASURE_STATE"
        self.value_embed = nn.Embedding(len(tokenizer.value2id), embed_dim) # Embed for "GLUCOSE_MEASURE_STATE_High"
        self.position_embed = nn.Embedding(len(tokenizer.token2id), embed_dim) # Embed for "GLUCOSE_MEASURE_STATE_High_Start" -> the full vocab size

        # --- Time embeddings ---
        self.time2vec_abs = Time2Vec(time2vec_dim)

        # --- Time projection + supervised head ---
        self.time_proj = nn.Linear(time2vec_dim, embed_dim, bias=False)
        
        hidden = max(4, time2vec_dim // 2) # Small regression head for Δt supervision (only used in phase‑1)
        self.time_head = nn.Sequential(
            nn.Linear(time2vec_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Skip path for direct time regression (init at 0 to preserve old behavior).
        self.time_skip = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.time_skip.weight)
        nn.init.zeros_(self.time_skip.bias)

        # Gate head: predicts P(Δt > 0) for Phase-1 temporal supervision (time-only path).
        self.dt_gate_head = nn.Linear(time2vec_dim, 1)

        # Sequence-aware Δt heads: use event embedding + time2vec for better gap prediction.
        # These take [embed_dim + time2vec_dim] as input and have more predictive power
        # than the time-only heads since event type is correlated with inter-event timing.
        _seq_dt_in = embed_dim + time2vec_dim
        _seq_dt_hidden = max(16, time2vec_dim)
        self.dt_gate_head_seq = nn.Sequential(
            nn.Linear(_seq_dt_in, _seq_dt_hidden),
            nn.ReLU(),
            nn.Linear(_seq_dt_hidden, 1),
        )
        self.dt_mag_head_seq = nn.Sequential(
            nn.Linear(_seq_dt_in, _seq_dt_hidden),
            nn.ReLU(),
            nn.Linear(_seq_dt_hidden, 1),
        )

        # --- patient‑context slot ----------------------------------------
        # NOTE: In phase-1, this projection's output is added directly to the event
        # embeddings as an additive bias in forward_with_decoder, so it trains
        # alongside the decoder via those gradients.
        # In phase-2, forward() returns it separately as `cond` — the transformer's
        # AdaLN blocks use it for shift/scale/gate conditioning (no direct addition).
        self.context_proj = nn.Linear(ctx_dim, embed_dim, bias=False)

        # --- Final projection ---
        concat_dim = 5 * embed_dim  # raw + concept + value + pos + time
        self.final_proj = nn.Linear(concat_dim, embed_dim)

        # --- regularisation ----------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)

        # --- Decoder tied to token embeddings ----------------------------
        self.decoder = nn.Linear(embed_dim, len(tokenizer.token2id), bias=False)
        self.decoder.weight = self.position_embed.weight  # weight tying


    def predict_time(self, abs_ts):
        """
        Helper used during Phase-1 to supervise Time2Vec.
        abs_ts must be the same tensor given to forward().
        Returns: [B, T, 1] values ∈ [0,1]
        """
        t_abs = self.time2vec_abs(abs_ts)      # [B, T, k]
        logits = self.time_head(t_abs)         # [B, T, 1]
        logits = logits + self.time_skip(abs_ts.unsqueeze(-1))
        return torch.sigmoid(logits)

    def predict_dt(self, abs_ts):
        """
        Purpose: Phase-1 two-head Δt supervision — gate + magnitude.
        Method: Runs time2vec on abs_ts, returns a gate logit (P(Δt>0)) and
                a raw magnitude logit (target: log1p(Δt_hours)).

        Args:
            abs_ts (FloatTensor): [B, T] normalised absolute timestamps (hours/336).

        Returns:
            gate_logit (FloatTensor): [B, T, 1] raw logit for BCE gate loss.
            mag_logit  (FloatTensor): [B, T, 1] raw logit for log1p magnitude MSE.
        """
        t_abs      = self.time2vec_abs(abs_ts)                         # [B, T, k]
        gate_logit = self.dt_gate_head(t_abs)                          # [B, T, 1]
        mag_logit  = self.time_head(t_abs) + self.time_skip(abs_ts.unsqueeze(-1))  # [B, T, 1]
        return gate_logit, mag_logit

    def predict_dt_from_seq(self, seq, abs_ts):
        """
        Sequence-aware Δt prediction for Phase-1.
        Uses event embedding (seq) + Time2Vec(abs_ts) to predict P(Δt>0) and log1p(Δt_hours).
        seq is expected to be detached from the main BCE computation graph so Δt
        gradients don't double-flow through the token embeddings.

        Args:
            seq     (FloatTensor): [B, T, D] event embeddings (detached).
            abs_ts  (FloatTensor): [B, T] absolute timestamps (hours/336).

        Returns:
            gate_logit (FloatTensor): [B, T, 1]
            mag_logit  (FloatTensor): [B, T, 1]
        """
        t_abs    = self.time2vec_abs(abs_ts)                         # [B, T, k]
        combined = torch.cat([seq, t_abs], dim=-1)                   # [B, T, D+k]
        gate_logit = self.dt_gate_head_seq(combined)                 # [B, T, 1]
        mag_logit  = self.dt_mag_head_seq(combined)                  # [B, T, 1]
        return gate_logit, mag_logit


    def forward(self, parent_raw_ids, concept_ids, value_ids, position_ids,
                 abs_ts, patient_contexts, return_mask=False, detach_time=False):
        """
        Build time-aware event embeddings for a full sequence.

        Args:
            parent_raw_ids (LongTensor):  [B, T, P] (P=max_parents)
            concept_ids (LongTensor):     [B, T]
            value_ids (LongTensor):       [B, T]
            position_ids (LongTensor):    [B, T]
            abs_ts (FloatTensor):         [B, T]
            patient_contexts (FloatTensor): [B, ctx_dim]
            return_mask (bool): Whether to return an attention mask
            detach_time (bool): If True, stop gradients through time embeddings

        Returns:
                    seq:  [B, T, D]  (Event embeddings)
                    cond: [B, D]     (Patient context for AdaLN blocks of the transformer)
                    mask: [B, T]     (Padding mask, if return_mask=True)
        """
        # --- Token lookups ---
        
        # parent_raw_ids: [B, T, P]
        p_emb = self.raw_concept_embed(parent_raw_ids)   # [B, T, P, D]

        # mask PAD parents (PAD id == tokenizer.pad_token_id)
        mask = (parent_raw_ids != self.padding_idx).unsqueeze(-1)  # [B, T, P, 1]
        p_emb = p_emb * mask

        den = mask.sum(dim=2).clamp_min(1.0)            # [B, T, 1]
        r_emb = p_emb.sum(dim=2) / den                  # [B, T, D]
        
        c_emb = self.concept_embed(concept_ids)     # [B, T, D]
        v_emb = self.value_embed(value_ids)
        p_emb = self.position_embed(position_ids)

        # --- Time encoding ---
        t_abs = self.time2vec_abs(abs_ts)           # [B, T, k] (k = time2vec_dim)
        t_emb = self.time_proj(t_abs)               # [B, T, D]
        if detach_time:
            t_emb = t_emb.detach()

        # --- Combine all token-wise pieces ---
        combined = torch.cat([r_emb, c_emb, v_emb, p_emb, t_emb], dim=-1)  # [B, T, 5D]
        ev_vec = self.final_proj(combined)                                 # [B, T, D]

        ev_vec = self.dropout(ev_vec) / self.scale                         # [B, T, D]

        # --- Sequence Norm ---
        seq = self.layernorm(ev_vec) # [B, T, D]

        # --- Patient Context (Separated for AdaLN) ---
        cond = self.context_proj(patient_contexts)      # [B, D]

        if return_mask:
            pad_mask = (position_ids != self.padding_idx)
            return seq, cond, pad_mask

        return seq, cond
    
    def forward_with_decoder(self, batch: dict, context_dropout_prob=0.15):
        """
        Runs full forward pass + decoding (for training phase 1).
        Unpack tuple (ignore cond for simple decoding task, or use it if decoder depended on it)
        In Phase 1, we just predict tokens from embeddings. 
        We are using the context projection in order to bias the embeddings with patient context.
        Context Dropout is applied here to improve robustness when patient context is missing (inference from phase-1 to phase-2).

        Returns:
            logits: [B, T, vocab_size] — scores for next-token prediction
        """
        # detach_time=False: allow BCE gradient to flow through Time2Vec so it gets a
        # strong training signal from the primary loss (previously only Δt auxiliary trained it).
        seq, cond = self.forward(
        parent_raw_ids=batch["parent_raw_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=batch["position_ids"],
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False,
        detach_time=False,
        )  # → returns only [B,T,D]

        # Context Dropout (Crucial for Phase 2 compatibility)
        # Keep tensor-side branch for stable, vectorized execution.
        if self.training:
            B = cond.size(0)
            drop = (torch.rand(B, 1, device=cond.device) < context_dropout_prob)  # [B, 1] per-sample
            cond_for_addition = torch.where(drop, torch.zeros_like(cond), cond)
        else:
            cond_for_addition = cond

        # Additive Interaction
        # Broadcast context [B, D] -> [B, 1, D] and add to sequence.
        # This biases the event representations based on patient static data.
        combined_embedding = seq + cond_for_addition.unsqueeze(1)

        return self.decoder(combined_embedding), seq  # [B,T,V] logits + [B,T,D] embeddings

    def save(self, epoch, best_val, optimizer, scheduler, path, lambda_schedule_state=None, bad_epochs=0, training_settings=None):
        ckpt = {
            "epoch": epoch,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "lambda_schedule_state": lambda_schedule_state,
            "model_state": self.state_dict(),
            "config": {
                "ctx_dim": self.context_proj.in_features,
                "time2vec_dim": self.time2vec_abs.freq.out_features + 1,
                "embed_dim": self.output_dim,
                "dropout": self.dropout.p,
                "vocab_size": self.position_embed.num_embeddings,
            },
            # Keep a copy of training settings used to produce this checkpoint.
            "training_settings": copy.deepcopy(training_settings),
        }
        from kineret.intervene.utils import save_checkpoint
        save_checkpoint(ckpt, path)
    
    @classmethod
    def load(cls, path, tokenizer, map_location=None):
        """
        Load EMREmbedding from a checkpoint.

        Args:
            path (str or Path): Path to checkpoint file.
            tokenizer (EMRTokenizer): Tokenizer to verify against saved model config.
            map_location: Device map for torch.load.  Defaults to CUDA when available,
                          CPU otherwise.  Training paths pass map_location=device explicitly.

        Returns:
            model: Loaded EMREmbedding instance
            epoch (int): Last epoch saved in checkpoint
            best_val (float): Best validation loss
            optimizer_state (dict): State dict for optimizer
            scheduler_state (dict): State dict for LR scheduler
        """
        import torch as _t
        if map_location is None:
            map_location = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        config = ckpt.get("config")
        if config is None:
            raise ValueError(
                "[EMREmbedding.load] Invalid checkpoint: missing 'config'."
            )

        # === Safety check: tokenizer vocab consistency ===
        expected_vocab = config["vocab_size"]
        actual_vocab = len(tokenizer.token2id)
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[EMREmbedding.load] Tokenizer vocab size mismatch: "
                f"expected {expected_vocab}, got {actual_vocab}"
            )

        model = cls(
            tokenizer=tokenizer,
            ctx_dim=config["ctx_dim"],
            time2vec_dim=config["time2vec_dim"],
            embed_dim=config["embed_dim"],
            dropout=config["dropout"]
        )
        state = ckpt["model_state"]
        unexpected = set(state.keys()) - set(model.state_dict().keys())
        if unexpected:
            raise RuntimeError(f"[EMREmbedding.load] Unexpected keys: {sorted(unexpected)}")
        model.load_state_dict(state, strict=False)
        model.to(map_location)   # move to target device after state dict load

        # Helpful metadata for callers that want to fully restore prior training settings.
        model.checkpoint_model_config = copy.deepcopy(config)
        model.checkpoint_training_settings = copy.deepcopy(ckpt.get("training_settings"))

        return (
            model,
            ckpt.get("epoch", 0),
            ckpt.get("best_val", float("inf")),
            ckpt.get("optim_state"),
            ckpt.get("scheduler_state"),
            ckpt.get("lambda_schedule_state"),
            ckpt.get("bad_epochs", 0),
        )


@logger
def train_embedder(embedder, train_loader, val_loader, resume=True, checkpoint_path=PHASE1_CHECKPOINT,
                   training_settings=TRAINING_SETTINGS):
    """
    Trains an EMREmbedding model using temporal multi-hot BCE and Δt regression.
    Total Loss = λ1 * BCE(temporal multi-hot) + λ_dt * (gate BCE + magnitude MSE)

    Legality-aware BCE notes:
     - We compute BCE per element (reduction="none") and then mask out illegal classes.
     - Illegal classes are excluded from both numerator and denominator, so they produce no gradient.
     - We also zero targets at illegal indices to avoid any accidental positive supervision there.
     - pos_weight is still applied element-wise by BCEWithLogits; masking happens AFTER the per-element loss,
       so class balancing remains intact but is computed only over the allowed set at each (b,t).
     - The final normalization divides by weights.sum() (the count of allowed elements), not B*T*V.

    Because masking happens after per-element BCE, class rebalancing via pos_weight is preserved but applied 
    only over the legal subset at each (b, t). This prevents the model from learning to suppress tokens merely because 
    they were illegal at that step, aligning Phase-1 supervision with Phase-2 decoding constraints.

    Args:
        embedder (EMREmbedding): The embedding model with decoder.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        resume (bool): Resume from last checkpoint if available.
        checkpoint_path (str): Path to save the best model and state.
        training_settings (dict): A settings dictionary, imported from model_config.

    Returns:
        Tuple: (trained model, train_losses, val_losses)
    """
    set_seed(SEED)  # reproducible Phase-1 dataloader shuffle / sampler draws / dropout
    # Create global training lookup Tensors once the tokenizer is available and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    luts = build_luts(embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k,v in luts.items()}
    embedder.to(device)
    from kineret.intervene.utils import autocast_dtype
    use_amp, amp_dtype = autocast_dtype(device)
    from kineret.intervene.utils import StepTimer
    _timer = StepTimer()

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    # If resuming, prefer checkpoint-saved settings to avoid config mismatch.
    if resume and ckpt_last.exists():
        pre_ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=True)
        if pre_ckpt.get("training_settings") is not None:
            training_settings = pre_ckpt["training_settings"]

    # ----- Loss & Optimizer -----
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=training_settings["phase1_learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=embedder.tokenizer.token_weights.to(device), reduction="none")

    # All outcome + terminal token ids — used for 1-hot override in multi-hot targets
    _all_outcome_names = sorted(set(OUTCOMES + TERMINAL_OUTCOMES))
    p1_outcome_ids = torch.tensor(
        [embedder.tokenizer.token2id[n] for n in _all_outcome_names if n in embedder.tokenizer.token2id],
        dtype=torch.long, device=device,
    )

    # ----- Resume logic -----
    train_losses, val_losses = [], []
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0
    
    lambda_schedule_state = None
    if resume and ckpt_last.exists():
        print(f"[Phase-1] Resuming from checkpoint: {ckpt_last}")
        embedder, start_epoch, best_val, optim_state, scheduler_state, lambda_schedule_state, bad_epochs = EMREmbedding.load(ckpt_last, tokenizer=embedder.tokenizer, map_location=device)
        embedder.to(device)
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        # Prefer checkpoint-saved settings to avoid resume/config mismatch issues.
        if getattr(embedder, "checkpoint_training_settings", None) is not None:
            training_settings = embedder.checkpoint_training_settings
        start_epoch += 1

    schedule_controller = LambdaScheduleController(
        schedule_config=training_settings["phase1_scheduler"],
        start_epoch=start_epoch
    )
    if lambda_schedule_state is not None:
        schedule_controller.load_state_dict(lambda_schedule_state)

    # ----- Epoch function -----
    def run_epoch(loader, epoch, train_flag=False):
        """
        Returns weighted and raw component losses for logging and scheduler calibration.
        """
        embedder.train() if train_flag else embedder.eval()
        total_loss, total_bce, total_dt = 0.0, 0.0, 0.0
        total_dt_raw = 0.0

        for batch in _timer.wrap(tqdm(loader, desc="Training" if train_flag else "Validation", leave=False, mininterval=5.0, miniters=10, dynamic_ncols=True)):
            batch = {k: v.to(device) for k, v in batch.items()}

            if train_flag:
                optimizer.zero_grad(set_to_none=True)

            # BCE Logits + Loss (also returns seq for seq-aware Δt head)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                bce_logits, seq = embedder.forward_with_decoder(batch)  # [B, T, V], [B, T, D]
            bce_logits = bce_logits.float()
            seq = seq.float()

            # build legality with teacher forcing (same LUTs as phase-2)
            illegal = compute_legality_masks_tf(
                position_ids=batch["position_ids"],
                is_start=luts["is_start"],
                is_end=luts["is_end"],
                base_id=luts["base_id"],
                start_ids_per_base=luts["start_ids_per_base"],
                end_ids_per_base=luts["end_ids_per_base"],
                meal_rank=luts["meal_rank"],
                meal_pred_rank=luts["meal_pred_rank"],
                K_meals=luts["K_meals"],
                conflict_mat=luts["conflict_mat"],
                predict_block=luts["predict_block"],   # PAD/MASK/CTX only
            )

            # Temporal: all tokens within phase1_bce_window_hours
            _ABS_TS_SCALE = 336.0
            _P1_WIN = training_settings.get("phase1_bce_window_hours", 3.0) / _ABS_TS_SCALE
            # next_token_ids for phase-1: query positions cover full T, so shift by 1
            # and pad the last column with padding_idx (no next token after last position).
            _pos_ids = batch["position_ids"]
            _B = _pos_ids.size(0)
            _pad_col = torch.full((_B, 1), embedder.padding_idx, dtype=torch.long, device=_pos_ids.device)
            _next_tok = torch.cat([_pos_ids[:, 1:], _pad_col], dim=1)  # [B, T]
            multi_hot_targets = get_temporal_multi_hot_targets(
                target_ids=_pos_ids,
                all_abs_ts=batch["abs_ts"],
                query_abs_ts=batch["abs_ts"],
                padding_idx=embedder.padding_idx,
                vocab_size=bce_logits.size(-1),
                window_size=_P1_WIN,
                outcome_ids=p1_outcome_ids,
                next_token_ids=_next_tok,
            )
            multi_hot_targets = multi_hot_targets.masked_fill(illegal, 0.0)

            raw = loss_fn(bce_logits, multi_hot_targets)       # [B, T, V], Per-element loss
            weights = ((~illegal) & (batch["position_ids"] != embedder.padding_idx).unsqueeze(-1)).float()  # [B,T,V]
            den = weights.sum().clamp_min(1.0)
            bce_loss = (raw * weights).sum() / den

            lambdas = schedule_controller.get_lambdas(epoch)

            # Δt: two-head temporal supervision matching Phase-2 structure.
            # Gate BCE (all non-pad pairs): P(Δt>0) — gives signal on simultaneous events.
            # Magnitude MSE (non-simultaneous pairs only): log1p(Δt_hours) — meaningful scale.
            nonpad_src  = (batch["position_ids"][:, :-1] != embedder.padding_idx)  # [B, T-1]
            nonpad_tgt  = (batch["position_ids"][:, 1:]  != embedder.padding_idx)  # [B, T-1]
            nonpad_pair = nonpad_src & nonpad_tgt

            dt_vals_hrs = (batch["abs_ts"][:, 1:] - batch["abs_ts"][:, :-1]) * 336.0  # hours
            dt_nonzero  = dt_vals_hrs > 1e-3
            nonpad_nz   = nonpad_pair & dt_nonzero

            # Sequence-aware Δt head: event embedding + time2vec predicts inter-event gap.
            # Gradient flows through seq so embeddings learn temporal spacing structure
            # (complementary to BCE which trains token sequence prediction).
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                gate_logit, mag_logit = embedder.predict_dt_from_seq(
                    seq[:, :-1], batch["abs_ts"][:, :-1]
                )  # [B, T-1, 1]
            gate_logit = gate_logit.squeeze(-1).float()  # [B, T-1]
            mag_logit  = mag_logit.squeeze(-1).float()   # [B, T-1]

            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit[nonpad_pair],
                dt_nonzero[nonpad_pair].float(),
                reduction="mean",
            )
            if nonpad_nz.any():
                mag_loss = F.mse_loss(
                    mag_logit[nonpad_nz],
                    torch.log1p(dt_vals_hrs[nonpad_nz]),  # log1p(hours): 1hr→0.69, 6hr→1.95
                    reduction="mean",
                )
            else:
                mag_loss = gate_logit.new_tensor(0.0)

            dt_raw = gate_loss + mag_loss
            time_loss = dt_raw * lambdas["dt"]
            loss = bce_loss + time_loss

            if train_flag:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += bce_loss.item()
            total_dt   += time_loss.item()
            total_dt_raw += dt_raw.item()

        n = len(loader)
        return (
            total_loss / n,
            total_bce / n,
            total_dt / n,
            total_dt_raw / n,
        )

    # ----- Training loop -----
    for epoch in range(start_epoch, training_settings["phase1_n_epochs"] + 1):
        tr_tot, tr_bce, tr_dt, tr_dt_raw = run_epoch(train_loader, epoch=epoch, train_flag=True)
        vl_tot, vl_bce, vl_dt, _ = run_epoch(val_loader, epoch=epoch, train_flag=False)

        # Update auxiliary scheduler:
        #   vl_total → plateau detection
        #   tr_main  → calibration denominator (training BCE)
        #   dt       → calibration numerator (training raw aux loss)
        schedule_events = schedule_controller.update(
            epoch=epoch,
            vl_total=vl_tot,
            tr_main=tr_bce,
            dt=tr_dt_raw,
        )
        for msg in schedule_events:
            print(msg)

        # Step the plateau scheduler on the validation total
        scheduler.step(vl_tot)

        # Collect losses
        train_losses.append(tr_tot)
        val_losses.append(vl_tot)

        print(f"""[Phase-1] Epoch {epoch:03d}
            --> Train={tr_tot:.4f} (BCE={tr_bce:.4f}  Δt={tr_dt:.4f})
            --> Val={vl_tot:.4f} (BCE={vl_bce:.4f}  Δt={vl_dt:.4f})
            --> RawTrain dt={tr_dt_raw:.7f}""")
        print("   " + _timer.report())
        _timer.reset()

        # Save best model only after aux-scheduler warmup is complete.
        warmup_gate = schedule_controller.current_warmup_end_epoch()

        min_delta_rel = training_settings.get("early-stop-min-delta-rel", 1e-3)
        if (vl_tot < best_val * (1.0 - min_delta_rel)) and (epoch >= warmup_gate):
            best_val = vl_tot
            bad_epochs = 0
            embedder.save(epoch, best_val, optimizer, scheduler, ckpt_path,
                          lambda_schedule_state=schedule_controller.state_dict(), bad_epochs=bad_epochs,
                          training_settings=training_settings)
            print("[Phase-1]: Current best model saved.")
        elif epoch >= warmup_gate:
            bad_epochs += 1
            if bad_epochs >= training_settings["early-stop-patience"]:
                # Save last checkpoint before stopping
                embedder.save(epoch, best_val, optimizer, scheduler, ckpt_last,
                              lambda_schedule_state=schedule_controller.state_dict(), bad_epochs=bad_epochs,
                              training_settings=training_settings)
                print("[Phase-1]: Early stopping triggered.")
                break
        # else: warmup not complete — no best update, no early-stop counter,
        # but still fall through to save ckpt_last so a short run / interrupted
        # training leaves a usable checkpoint on disk.

        # Save last checkpoint at end of every epoch regardless of warmup state.
        embedder.save(epoch, best_val, optimizer, scheduler, ckpt_last,
                      lambda_schedule_state=schedule_controller.state_dict(), bad_epochs=bad_epochs,
                      training_settings=training_settings)
    
    plot_losses(train_losses, val_losses,
                save_path=str(ckpt_path.parent / "phase1_loss.png"),
                title="Phase 1 - embedder")
    return embedder, train_losses, val_losses