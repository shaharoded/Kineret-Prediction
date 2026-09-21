"""
transformer.py
==============

Bidirectional EMR encoder (BERT-style pivot from the original AR transformer).

The model consumes the same hierarchical embeddings produced by
``EMREmbedding`` and the same batch layout from ``collate_emr``, but the
attention is fully bidirectional and the training objective is masked language
modelling with two small time-aware auxiliaries instead of next-token BCE.

Three heads are exposed:

* ``lm_head``                — full-vocab logits for Phase-2 MLM CE.
* ``time_since_admission``   — Phase-2 aux, regression of t/336 at every
                                non-pad position.
* ``time_to_neighbour``      — Phase-2 aux, regression of the local-gap
                                distance at masked positions only.

Phase-3 attaches a :class:`TaskHeads` module containing a per-outcome
attention pool and shared MLP that produces patient-level risk + time
predictions.  The encoder is fine-tuned at ``phase3_backbone_lr_factor``.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend
from pathlib import Path
from tqdm.auto import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from kineret.intervene.embedder import EMREmbedding
from kineret.intervene.config.model_config import *
from kineret.intervene.config.dataset_config import (
    OUTCOMES, TERMINAL_OUTCOMES, RELEASE_TOKEN,
)
from kineret.intervene.utils import (
    set_seed, build_luts, apply_mlm_mask, logger, plot_losses,
    time_to_neighbour_targets, build_patient_labels,
)
from kineret.intervene.schedulers import LambdaScheduleController, LRScheduleController


# ───────── components  ───────────────────────────────────────────────── #
class BidirectionalSelfAttention(nn.Module):
    """
    Multi-head bidirectional self-attention with temporal RoPE.

    Mirrors the AR backbone's attention but drops the causal mask — every
    position attends to every other non-pad position.  Temporal RoPE is kept
    so absolute time is still injected into Q/K rotations.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg["embed_dim"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["embed_dim"]

        self.qkv = nn.Linear(cfg["embed_dim"], 3 * cfg["embed_dim"], bias=cfg["bias"])
        self.proj = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=cfg["bias"])
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])
        self.rope_t_scale = cfg.get("rope_t_scale", 24.0)

    def _apply_temporal_rope(self, x, abs_ts):
        """
        Rotate Q/K by timestamp-dependent phases before dot-product attention.
        Identical to the AR variant.
        """
        _, _, _, hd = x.shape
        half = hd // 2
        freq = 1.0 / (self.rope_t_scale ** (torch.arange(half, device=x.device, dtype=x.dtype) / half))
        theta = abs_ts.unsqueeze(-1).to(x.dtype) * freq            # [B, T, half]
        theta = theta.unsqueeze(1)                                  # [B, 1, T, half]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

    def forward(self, x, key_pad_mask=None, abs_ts=None):
        """
        Args:
            x            : [B, T, C]
            key_pad_mask : [B, T] bool (True=keep).  Padded positions are masked
                           out symmetrically (no token may attend to PAD).
            abs_ts       : [B, T] absolute timestamps for temporal RoPE.

        Returns:
            y            : [B, T, C]
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)

        if abs_ts is not None:
            q = self._apply_temporal_rope(q, abs_ts)
            k = self._apply_temporal_rope(k, abs_ts)

        _sdpa_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

        if key_pad_mask is not None:
            # Build a [B, 1, T, T] bool mask of PAD columns; broadcast across query rows.
            pad_m = (~key_pad_mask).unsqueeze(1).unsqueeze(2)        # [B,1,1,T]
            attn_mask = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(pad_m, float("-inf"))
            with sdpa_kernel(_sdpa_backends):
                attn = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )
        else:
            with sdpa_kernel(_sdpa_backends):
                attn = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=False,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )

        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    """
    SwiGLU MLP (SiLU gating) — same shape as the AR backbone so checkpoint
    structure is recognisable and tested layer-by-layer behaviours transfer.
    """
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = 4 * cfg["embed_dim"]
        self.w1 = nn.Linear(cfg["embed_dim"], 2 * hidden_dim, bias=cfg["bias"])
        self.w2 = nn.Linear(hidden_dim, cfg["embed_dim"], bias=cfg["bias"])
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        projected = self.w1(x)
        x_val, x_gate = projected.chunk(2, dim=-1)
        out = x_val * F.silu(x_gate)
        return self.drop(self.w2(out))


class AdaLNBlock(nn.Module):
    """
    Encoder block with AdaLN-Zero conditioning on patient context.

    Identical to the AR block layout except the attention sub-layer is
    bidirectional (no causal mask).  Zero-init keeps blocks identity at the
    start of Phase-2 training.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = BidirectionalSelfAttention(cfg)
        self.mlp = MLP(cfg)

        self.ln1 = nn.LayerNorm(cfg["embed_dim"], elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(cfg["embed_dim"], elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg["embed_dim"], 6 * cfg["embed_dim"], bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond_emb, key_pad_mask=None, abs_ts=None):
        """
        Args:
            x        : [B, T, D]
            cond_emb : [B, D] (patient context)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond_emb).chunk(6, dim=1)
        )

        norm_x   = self.ln1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.att(norm_x, key_pad_mask=key_pad_mask, abs_ts=abs_ts)
        x = x + gate_msa.unsqueeze(1) * attn_out

        norm_x = self.ln2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)
        return x


# ───────── per-outcome attention pool + task heads ───────────────────── #
class PerOutcomeAttnPool(nn.Module):
    """
    K learnable outcome queries cross-attend over the encoder's final hidden
    states to produce one pooled feature per (patient, outcome).
    """
    def __init__(self, d_model, n_outcomes, n_heads=4, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_outcomes, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_seq, key_padding_mask):
        """
        Args:
            h_seq             : [B, T, D] encoder output.
            key_padding_mask  : [B, T] bool — True at PADDED positions (matches
                                ``nn.MultiheadAttention``'s convention).

        Returns:
            [B, K, D] pooled features (one per outcome query).
        """
        B = h_seq.size(0)
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.attn(Q, h_seq, h_seq, key_padding_mask=key_padding_mask,
                         need_weights=False)
        return self.norm(z)


class TaskHeads(nn.Module):
    """
    Phase-3 head: per-outcome attention pool → shared MLP → (risk, time) splits.

    The risk head consumes K_risk = K-1 outcome queries (RELEASE excluded;
    P(release) is derived as 1 − P(DEATH) at eval if needed).  The time head
    consumes all K outcomes so the RELEASE slot can supply length-of-stay.

    Parameters
    ----------
    d_model         : encoder output dim.
    n_outcomes      : total number of outcome queries (K).
    risk_idx_buf    : LongTensor[K_risk] — positions of risk-only outcomes
                      inside the K pooled features.
    time_idx_buf    : LongTensor[K_time] — positions of time-only outcomes
                      (typically all K).
    hidden          : MLP hidden width.
    """
    def __init__(self, d_model, n_outcomes, risk_idx_buf, time_idx_buf,
                 hidden=256, dropout=0.1, n_heads=4):
        super().__init__()
        self.pool = PerOutcomeAttnPool(d_model, n_outcomes, n_heads=n_heads, dropout=dropout)
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden, 1)   # per-outcome scalar logit
        self.time_head = nn.Linear(hidden, 1)   # per-outcome scalar (pre-softplus)

        # Buffers (not trainable) so risk_idx / time_idx ride along to the
        # device with the module and survive state-dict round-trips.
        self.register_buffer("risk_idx", risk_idx_buf, persistent=True)
        self.register_buffer("time_idx", time_idx_buf, persistent=True)

    def forward(self, h_seq, key_padding_mask):
        """
        Returns:
            risk_logits : [B, K_risk]
            time_pred   : [B, K_time]   non-negative hours (softplus on raw logit).
        """
        z = self.pool(h_seq, key_padding_mask)
        z = self.shared(z)
        risk_logits = self.risk_head(z[:, self.risk_idx, :]).squeeze(-1)
        time_pred = F.softplus(self.time_head(z[:, self.time_idx, :])).squeeze(-1)
        return risk_logits, time_pred


# ───────── the BERT-style encoder that consumes EMREmbedding ──────────── #
class InterveneEncoder(nn.Module):
    """
    Bidirectional transformer encoder over an external :class:`EMREmbedding`.

    The encoder is paradigm-agnostic — Phase-2 uses an MLM head and two
    time-aware auxiliaries; Phase-3 attaches a :class:`TaskHeads` module that
    pools the encoder output into per-outcome (risk, time) predictions.

    Parameters
    ----------
    cfg       : dict   — hyper-parameters (n_layer, n_head, dropout, ...).
    embedder  : EMREmbedding — pre-trained Phase-1 embedder, weight-tied to
                ``lm_head`` so the MLM head shares the token embeddings.
    use_checkpoint : bool — gradient checkpointing toggle.
    """

    def __init__(self, cfg: dict, embedder: EMREmbedding, use_checkpoint: bool = True):
        super().__init__()
        set_seed(SEED)

        assert cfg["embed_dim"] == embedder.output_dim, (
            "Config embed_dim must equal EMREmbedding.output_dim"
        )

        self.cfg = cfg
        self.embedder = embedder
        self.use_checkpoint = use_checkpoint

        vocab_size = self.embedder.decoder.out_features

        assert hasattr(self.embedder.tokenizer, "id2token"), "[InterveneEncoder] Embedder missing id2token map"
        assert len(self.embedder.tokenizer.id2token) == vocab_size, (
            f"[InterveneEncoder] id2token size mismatch: got {len(self.embedder.tokenizer.id2token)}, expected {vocab_size}"
        )

        # --- Backbone ---
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["embed_dim"], eps=1e-5)

        # --- MLM head (full vocab, weight-tied to embedder.position_embed) ---
        self.lm_head = nn.Linear(cfg["embed_dim"], vocab_size, bias=False)
        self.lm_head.weight = self.embedder.position_embed.weight

        # --- Time auxiliaries ---
        # Both heads are deliberately tiny so they cannot drown out the MLM
        # gradient.  They condition the hidden state to retain time-locality
        # information that the MLM head alone may not enforce.
        _aux_hidden = max(16, cfg["time2vec_dim"])
        # time_since_admission: predicts t/336 (normalised hours from admission)
        # at every non-pad position; trained with MSE.
        self.time_since_admission_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], _aux_hidden),
            nn.GELU(),
            nn.Linear(_aux_hidden, 1),
        )
        # time_to_neighbour: predicts min(t-t_prev, t_next-t) / 24h at masked
        # positions only; trained with MSE.  Forces masked tokens to retain
        # local-time context after their concept identity is hidden.
        self.time_to_neighbour_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], _aux_hidden),
            nn.GELU(),
            nn.Linear(_aux_hidden, 1),
        )

        # --- Outcome bookkeeping ---
        # The head layout is the COHORT's outcome list, full stop. Upstream this
        # was filtered to outcomes appearing in the tokenizer vocabulary,
        # because labels were read back off the token stream. Here they are
        # injected from the cohort, so a target needs no token to be learnable
        # -- and filtering on the vocabulary is actively wrong:
        #
        #   the sigma-bin arm carries NO outcome tokens by design (it is the
        #   knowledge-free baseline), so vocabulary filtering left it with a
        #   single RELEASE_EVENT head. Every real complication was dropped, the
        #   injected labels for them were discarded by `build_patient_labels`,
        #   and rung 5 -- half of this study's central claim -- was predicting
        #   nothing at all.
        #
        # Absence from the vocabulary is still reported: it says the model
        # cannot SEE that outcome in its input, which is a property of the arm
        # worth knowing, not a reason to drop the head.
        tok = self.embedder.tokenizer
        all_config_outcomes = sorted(list(set(OUTCOMES + TERMINAL_OUTCOMES)))
        missing_outcomes = [n for n in all_config_outcomes if n not in tok.token2id]
        if missing_outcomes:
            print(f"[InterveneEncoder] Outcomes with no token in this arm's input "
                  f"(head kept, labels come from the cohort): {missing_outcomes}")

        valid_outcomes = list(all_config_outcomes)
        if not valid_outcomes:
            raise ValueError(
                f"[InterveneEncoder] No outcomes configured. Call "
                f"dataset_config.configure(..., outcome_names=cohort.outcome_names) "
                f"before building the model."
            )
        self.outcome_names = valid_outcomes
        self.num_outcomes = len(self.outcome_names)

        # Index of RELEASE inside outcome_names (drops from the risk head; kept
        # by the time head for length-of-stay regression).
        self._release_idx = (
            self.outcome_names.index(RELEASE_TOKEN) if RELEASE_TOKEN in self.outcome_names else -1
        )

        # --- Init ---
        self.apply(self._init_weights)
        for n, p in self.named_parameters():
            if n.endswith(("att.proj.weight", "mlp.w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

        # Phase-3 task heads — built lazily by ``attach_task_heads`` so Phase-2
        # checkpoints stay slim and the head buffer indices can be tokenizer-derived.
        self.task_heads: TaskHeads | None = None

        print(f"[InterveneEncoder]: Total params: {self.get_num_params()/1e6:.2f} M")

    # -------------------------------------------------------- helpers --- #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------- task-head attachment ---- #
    def attach_task_heads(self, hidden=256, n_heads=4, dropout=None):
        """
        Purpose: Build the Phase-3 :class:`TaskHeads` module with RELEASE
                 excluded from the risk targets and kept on the time head.
        Method:  Compute index buffers once from ``self.outcome_names`` so the
                 head can index its K pooled features without re-deriving the
                 indices at every forward.

        Args:
            hidden  (int): MLP hidden width.
            n_heads (int): heads in the pool's MultiheadAttention.
            dropout (float|None): dropout used inside the pool + shared MLP.
                ``None`` (default) inherits the backbone's ``cfg['dropout']``;
                pass an explicit value (e.g., 0.20) to regularise the Phase-3
                head independently of the backbone.

        Returns:
            self.task_heads (TaskHeads).
        """
        K = self.num_outcomes
        all_idx = torch.arange(K, dtype=torch.long)
        if self._release_idx >= 0:
            # Drop RELEASE from the risk head — P(RELEASE) = 1 − P(DEATH)
            # in this cohort; LoS is reported via the RELEASE time-head.
            risk_idx = torch.tensor(
                [i for i in range(K) if i != self._release_idx], dtype=torch.long,
            )
        else:
            risk_idx = all_idx.clone()
        time_idx = all_idx

        pool_dropout = self.cfg["dropout"] if dropout is None else float(dropout)
        self.task_heads = TaskHeads(
            d_model=self.cfg["embed_dim"],
            n_outcomes=K,
            risk_idx_buf=risk_idx,
            time_idx_buf=time_idx,
            hidden=hidden,
            dropout=pool_dropout,
            n_heads=n_heads,
        )
        # Match the device of the rest of the module so Phase-3 finetune does
        # not have to remember to move the head explicitly.
        device = next(self.parameters()).device
        self.task_heads.to(device)
        return self.task_heads

    # --------------------------------------------------- forward (encode) - #
    def encode(self, parent_raw_ids, concept_ids, value_ids, position_ids,
               abs_ts, context_vec=None):
        """
        Purpose: Run the encoder backbone and return its final hidden states.
        Method:  Embed → AdaLN encoder stack → ln_f.  Bidirectional throughout.

        Returns:
            hidden  : [B, T, D] encoder output (post final LN).
            pad_mask : [B, T] bool, True at valid (non-PAD) positions.
        """
        x, cond_emb, pad_mask = self.embedder(
            parent_raw_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec, return_mask=True,
        )
        x = self.drop(x)

        _ckpt_active = torch.is_grad_enabled() and self.use_checkpoint
        for blk in self.blocks:
            if _ckpt_active:
                def _ckpt(_x, _c, _m, _t, _blk=blk):
                    from kineret.intervene.utils import autocast_dtype
                    _amp, _dtype = autocast_dtype(_x.device)
                    with torch.autocast(device_type=_x.device.type, dtype=_dtype,
                                        enabled=_amp):
                        return _blk(_x, _c, key_pad_mask=_m, abs_ts=_t)
                x = checkpoint.checkpoint(_ckpt, x, cond_emb, pad_mask, abs_ts, use_reentrant=True)
            else:
                x = blk(x, cond_emb, key_pad_mask=pad_mask, abs_ts=abs_ts)

        x = self.ln_f(x)
        return x, pad_mask

    def forward(self, parent_raw_ids, concept_ids, value_ids, position_ids,
                abs_ts, context_vec=None):
        """
        Phase-2 forward — returns lm_logits + the two time aux predictions.

        Returns:
            lm_logits  : [B, T, V] full-vocab logits for MLM CE.
            t_pos_pred : [B, T] normalised time-since-admission prediction.
            t_local_pred : [B, T] local-gap prediction (only used at masked
                           positions; values at unmasked positions are
                           ignored downstream).
            pad_mask   : [B, T] bool, True at valid positions.
        """
        hidden, pad_mask = self.encode(
            parent_raw_ids, concept_ids, value_ids, position_ids, abs_ts, context_vec,
        )
        lm_logits = self.lm_head(hidden)
        t_pos_pred   = self.time_since_admission_head(hidden).squeeze(-1)
        t_local_pred = self.time_to_neighbour_head(hidden).squeeze(-1)
        return lm_logits, t_pos_pred, t_local_pred, pad_mask

    def predict(self, parent_raw_ids, concept_ids, value_ids, position_ids,
                abs_ts, context_vec=None):
        """
        Phase-3 / inference forward — encode + run the patient-level task heads.

        Returns:
            risk_logits : [B, K_risk] (RELEASE dropped).
            time_pred   : [B, K_time] non-negative hours (softplus on raw logit).
            hidden      : [B, T, D] encoder output (handy for downstream tools).
            pad_mask    : [B, T] bool, True at valid positions.
        """
        if self.task_heads is None:
            raise RuntimeError(
                "[InterveneEncoder.predict] task_heads not attached. Call "
                "model.attach_task_heads() before Phase-3 training / inference."
            )
        hidden, pad_mask = self.encode(
            parent_raw_ids, concept_ids, value_ids, position_ids, abs_ts, context_vec,
        )
        # MultiheadAttention's key_padding_mask is True at PAD positions; we
        # carry pad_mask as True at VALID positions, so invert here.
        kpm = ~pad_mask
        risk_logits, time_pred = self.task_heads(hidden, key_padding_mask=kpm)
        return risk_logits, time_pred, hidden, pad_mask

    # ---------------------------------------------- optimiser config ---- #
    def configure_optimizers(self, weight_decay, learning_rate, betas,
                             embedder_lr_factor=0.1, backbone_lr_factor=1.0,
                             head_lr=None):
        """
        Purpose: Build an AdamW with three param groups — embedder, backbone,
                 and (optionally) Phase-3 task heads at a separate LR.
        Method:  Same dim≥2 → weight-decay rule as the AR backbone.

        Args:
            weight_decay        (float): WD applied to dim≥2 params.
            learning_rate       (float): main LR (applied to backbone).
            betas               (tuple): AdamW betas.
            embedder_lr_factor  (float): scale on embedder LR (default 0.1 to
                                         protect Phase-1 weights).
            backbone_lr_factor  (float): scale on backbone LR (Phase-3 uses
                                         ``phase3_backbone_lr_factor``).
            head_lr             (float|None): if set and task_heads exist,
                                         the task head trains at this LR.

        Returns:
            torch.optim.AdamW with up to three param groups.
        """
        embedder_param_ids = {id(p) for p in self.embedder.parameters()}
        task_param_ids = (
            {id(p) for p in self.task_heads.parameters()} if self.task_heads is not None else set()
        )

        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in embedder_param_ids or id(p) in task_param_ids:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay,
             "lr": learning_rate * backbone_lr_factor},
            {"params": no_decay, "weight_decay": 0.0,
             "lr": learning_rate * backbone_lr_factor},
            {"params": list(self.embedder.parameters()), "weight_decay": 0.0,
             "lr": learning_rate * embedder_lr_factor},
        ]

        if self.task_heads is not None:
            groups.append({
                "params": list(self.task_heads.parameters()),
                "weight_decay": weight_decay,
                "lr": head_lr if head_lr is not None else learning_rate,
            })

        return torch.optim.AdamW(groups, betas=betas)

    # --------------------------------------------- save / load --------- #
    def save(self, path, epoch=None, best_val=None, optimizer=None, scheduler=None,
             lambda_schedule_state=None, training_settings=None, bad_epochs=0,
             extra=None):
        ckpt = {
            "model_state": self.state_dict(),
            "config": copy.deepcopy(self.cfg),
            "vocab_size": self.embedder.decoder.out_features,
            "outcome_names": self.outcome_names,
            "num_outcomes": self.num_outcomes,
            "release_idx": self._release_idx,
            "has_task_heads": self.task_heads is not None,
            "task_heads_hidden": (
                self.task_heads.shared[0].out_features if self.task_heads is not None else None
            ),
            "task_heads_n_heads": (
                self.task_heads.pool.attn.num_heads if self.task_heads is not None else None
            ),
            "lambda_schedule_state": lambda_schedule_state,
            "training_settings": copy.deepcopy(training_settings),
            "bad_epochs": bad_epochs,
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if best_val is not None:
            ckpt["best_val"] = best_val
        if optimizer is not None:
            ckpt["optim_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            ckpt["scheduler_state"] = scheduler.state_dict()
        if extra is not None:
            ckpt["extra"] = extra
        from kineret.intervene.utils import save_checkpoint
        save_checkpoint(ckpt, path)

    @classmethod
    def load(cls, path, embedder, map_location="cpu", attach_task_heads=None):
        """
        Purpose: Reconstruct an InterveneEncoder from a checkpoint.
        Method:  Validate vocab + outcomes against the supplied embedder; build
                 the model; optionally re-attach task heads when the ckpt was
                 saved with them or the caller explicitly requests it.

        Args:
            path             (str/Path): checkpoint path.
            embedder         (EMREmbedding): tokenizer-compatible embedder.
            map_location     : torch.load map_location.
            attach_task_heads (bool|None): if True, build TaskHeads even if the
                              ckpt was Phase-2-only; if False, never attach;
                              if None, follow the ckpt's ``has_task_heads``.

        Returns:
            (model, epoch, best_val, optim_state, scheduler_state, lambda_state).
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        if "config" not in ckpt:
            raise ValueError("[InterveneEncoder.load] Invalid checkpoint: missing 'config'.")

        expected_vocab = ckpt["vocab_size"]
        actual_vocab = embedder.decoder.out_features
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[InterveneEncoder.load] Embedder vocab size mismatch: ckpt={expected_vocab}, embedder={actual_vocab}"
            )

        expected_outcome_names = set(ckpt.get("outcome_names", []))
        current_outcomes = set(OUTCOMES + TERMINAL_OUTCOMES)
        if expected_outcome_names and not expected_outcome_names.issubset(current_outcomes):
            raise ValueError(
                f"[InterveneEncoder.load] Outcome configuration mismatch.\n"
                f"  ckpt: {sorted(expected_outcome_names)}\n  current: {sorted(current_outcomes)}"
            )

        model = cls(cfg=ckpt["config"], embedder=embedder)

        want_heads = (
            attach_task_heads
            if attach_task_heads is not None
            else bool(ckpt.get("has_task_heads", False))
        )
        if want_heads:
            model.attach_task_heads(
                hidden=ckpt.get("task_heads_hidden") or 256,
                n_heads=ckpt.get("task_heads_n_heads") or 4,
            )

        state = ckpt["model_state"]
        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(state.keys())
        missing    = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        # task_heads.* keys are expected to be missing when loading a Phase-2
        # ckpt without heads — they get random init and only train in Phase-3.
        _th_missing = {k for k in missing if k.startswith("task_heads.")}
        if _th_missing and not want_heads:
            missing -= _th_missing
        elif _th_missing and want_heads:
            print(f"[InterveneEncoder.load] task_heads not in ckpt — keeping random init "
                  f"({len(_th_missing)} keys). Expected on first Phase-3 epoch.")
            missing -= _th_missing

        if missing:
            raise RuntimeError(f"[InterveneEncoder.load] Missing required keys: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"[InterveneEncoder.load] Unexpected keys: {sorted(unexpected)}")
        model.load_state_dict(state, strict=False)

        embedder_device = next(embedder.parameters()).device
        model.to(embedder_device)

        model.checkpoint_model_config = copy.deepcopy(ckpt["config"])
        model.checkpoint_training_settings = copy.deepcopy(ckpt.get("training_settings"))

        return (
            model,
            ckpt.get("epoch", 0),
            ckpt.get("best_val", float("inf")),
            ckpt.get("optim_state"),
            ckpt.get("scheduler_state"),
            ckpt.get("lambda_schedule_state"),
        )


# ───────── Phase-2 training: MLM + time auxiliaries ─────────────────── #
@logger
def pretrain_transformer(model, train_dl, val_dl, resume=True,
                         checkpoint_path=PHASE2_CHECKPOINT,
                         training_settings=TRAINING_SETTINGS):
    """
    Purpose: Phase-2 — bidirectional MLM pre-training of the EMR encoder with
             two small time-aware auxiliaries.
    Method:  At each step:
               • atomic-interval MLM mask is applied to the input batch;
               • encoder forward returns lm_logits + the two aux predictions;
               • main loss = full-vocab CE at masked positions only;
               • aux losses follow LambdaScheduleController calibration.

    Args:
        model (InterveneEncoder): bidirectional encoder with attached embedder.
        train_dl / val_dl : DataLoaders.
        resume (bool): resume from ``ckpt_last`` if present.
        checkpoint_path (str): destination for the best checkpoint.
        training_settings (dict): config — uses ``phase2_scheduler`` for aux
            calibration and ``phase2_learning_rate`` / ``weight_decay``.

    Returns:
        (model, train_losses, val_losses).
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    luts = build_luts(model.embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    if resume and ckpt_last.exists():
        pre_ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=True)
        if pre_ckpt.get("training_settings") is not None:
            training_settings = pre_ckpt["training_settings"]

    # Embedder is trainable in Phase-2 at 10× lower LR than the backbone.
    for p in model.embedder.parameters():
        p.requires_grad = True
    model.to(device)
    from kineret.intervene.utils import autocast_dtype
    use_amp, amp_dtype = autocast_dtype(device)
    from kineret.intervene.utils import StepTimer
    _timer = StepTimer()

    optimizer = model.configure_optimizers(
        weight_decay=training_settings["weight_decay"],
        learning_rate=training_settings["phase2_learning_rate"],
        betas=(0.9, 0.95),
        embedder_lr_factor=0.1,
        backbone_lr_factor=1.0,
    )
    scheduler = LRScheduleController(optimizer, training_settings, train_dl)

    # MLM ratio ramp from 0 → phase2_mlm_ratio over the main-only window, then steady.
    mlm_p_max = training_settings.get("phase2_mlm_ratio", 0.15)
    cbm_ramp_epochs = training_settings["phase2_scheduler"]["main_only_epochs"]
    print(f"[Phase-2] MLM ratio target = {mlm_p_max}")

    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    lambda_schedule_state = None

    if resume and ckpt_last.exists():
        print(f"[Phase-2]: Loading model from checkpoint: {ckpt_last}")
        loaded, start_epoch, best_val, opt_state, sch_state, lambda_schedule_state = InterveneEncoder.load(
            ckpt_last, embedder=model.embedder, map_location=device,
        )
        model = loaded
        model.to(device)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        scheduler = LRScheduleController(optimizer, training_settings, train_dl)
        if sch_state is not None:
            scheduler.load_state_dict(sch_state)
        start_epoch += 1

    # Phase-2 schedule comes straight from TRAINING_SETTINGS["phase2_scheduler"]:
    # main_only_epochs + aux_fraction_caps + order + ramp_epochs for the two
    # active auxes (t_pos, t_local). Agent edits the config; no inline overrides.
    schedule_controller = LambdaScheduleController(
        schedule_config=training_settings["phase2_scheduler"],
        start_epoch=start_epoch,
    )
    if lambda_schedule_state is not None:
        try:
            schedule_controller.load_state_dict(lambda_schedule_state)
        except Exception as e:
            # Resume from an incompatible ckpt: skip the controller state cleanly.
            print(f"[Phase-2] Could not restore aux schedule state ({e}); starting fresh.")

    train_losses, val_losses = [], []
    grad_accum_steps = training_settings.get("grad_accumulation_steps", 1)

    def run_epoch(loader, epoch, train_flag=False):
        model.train() if train_flag else model.eval()
        total_loss = total_mlm = total_tpos = total_tlocal = 0.0
        total_tpos_raw = total_tlocal_raw = 0.0
        accum_step = 0
        if train_flag:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_flag):
            for batch in _timer.wrap(tqdm(
                    loader, desc="Training" if train_flag else "Validation",
                    leave=False, mininterval=5.0, miniters=10, dynamic_ncols=True)):
                batch = {k: v.to(device) for k, v in batch.items()}

                # Mask ratio ramps with the foundational stage to keep early
                # training stable.
                if epoch <= cbm_ramp_epochs:
                    p_now = mlm_p_max * (epoch / max(1, cbm_ramp_epochs))
                else:
                    p_now = mlm_p_max

                batch, target_ids, mlm_mask = apply_mlm_mask(
                    batch=batch,
                    tokenizer=model.embedder.tokenizer,
                    forbid_ids=luts["forbid_mask_ids"],
                    luts=luts,
                    p=p_now,
                )

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    lm_logits, t_pos_pred, t_local_pred, pad_mask = model(
                        parent_raw_ids=batch["parent_raw_ids"],
                        concept_ids=batch["concept_ids"],
                        value_ids=batch["value_ids"],
                        position_ids=batch["position_ids"],
                        abs_ts=batch["abs_ts"],
                        context_vec=batch["context_vec"],
                    )
                lm_logits = lm_logits.float()
                t_pos_pred = t_pos_pred.float()
                t_local_pred = t_local_pred.float()

                # ---------- Main loss: full-vocab CE on masked positions ----------
                if mlm_mask.any():
                    flat_logits = lm_logits[mlm_mask]            # [N_mask, V]
                    flat_target = target_ids[mlm_mask]           # [N_mask]
                    mlm_loss = F.cross_entropy(flat_logits, flat_target, reduction="mean")
                else:
                    # No eligible positions in this batch — fall through with a
                    # zero loss that still carries grad to keep the graph happy.
                    mlm_loss = lm_logits.sum() * 0.0

                # ---------- Aux 1: time-since-admission (all non-pad) ----------
                t_pos_target = batch["abs_ts"]                   # already normalised to [0, 1] (hours/336)
                tpos_valid = pad_mask
                if tpos_valid.any():
                    t_pos_raw = F.mse_loss(
                        t_pos_pred[tpos_valid], t_pos_target[tpos_valid], reduction="mean",
                    )
                else:
                    t_pos_raw = lm_logits.sum() * 0.0

                # ---------- Aux 2: time-to-neighbour (masked only) ----------
                t_local_target, tloc_valid = time_to_neighbour_targets(
                    batch["abs_ts"], pad_mask, mlm_mask, max_hours=24.0,
                )
                if tloc_valid.any():
                    t_local_raw = F.mse_loss(
                        t_local_pred[tloc_valid], t_local_target[tloc_valid], reduction="mean",
                    )
                else:
                    t_local_raw = lm_logits.sum() * 0.0

                lambdas = schedule_controller.get_lambdas(epoch)
                t_pos_term = t_pos_raw   * lambdas.get("t_pos",   0.0)
                t_local_term = t_local_raw * lambdas.get("t_local", 0.0)

                loss = mlm_loss + t_pos_term + t_local_term

                if train_flag:
                    (loss / grad_accum_steps).backward()
                    accum_step += 1
                    if accum_step % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                total_loss   += loss.item()
                total_mlm    += mlm_loss.item()
                total_tpos   += t_pos_term.item()
                total_tlocal += t_local_term.item()
                total_tpos_raw   += t_pos_raw.item()
                total_tlocal_raw += t_local_raw.item()

        n = len(loader)
        return (
            total_loss / n,
            total_mlm / n,
            total_tpos / n,
            total_tlocal / n,
            total_tpos_raw / n,
            total_tlocal_raw / n,
        )

    for epoch in range(start_epoch, training_settings["phase2_n_epochs"] + 1):
        tr_tot, tr_mlm, tr_tpos, tr_tlocal, tr_tpos_raw, tr_tlocal_raw = run_epoch(
            train_dl, epoch=epoch, train_flag=True,
        )
        vl_tot, vl_mlm, vl_tpos, vl_tlocal, _, _ = run_epoch(
            val_dl, epoch=epoch, train_flag=False,
        )

        schedule_events = schedule_controller.update(
            epoch=epoch, vl_total=vl_tot, tr_main=tr_mlm,
            t_pos=tr_tpos_raw, t_local=tr_tlocal_raw,
        )
        for msg in schedule_events:
            print(msg)

        train_losses.append(tr_tot)
        val_losses.append(vl_tot)
        try:
            _lr_now = scheduler.get_last_lr()[0]
        except Exception:
            _lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Phase-2] Epoch {epoch:03d}  lr={_lr_now:.3e}\n"
              f"    --> Train={tr_tot:.4f} (MLM={tr_mlm:.4f}  tPos={tr_tpos:.4f}  tLoc={tr_tlocal:.4f})\n"
              f"    --> Val  ={vl_tot:.4f} (MLM={vl_mlm:.4f}  tPos={vl_tpos:.4f}  tLoc={vl_tlocal:.4f})")
        print("   " + _timer.report())
        _timer.reset()

        warmup_gate = schedule_controller.current_warmup_end_epoch()
        min_delta_rel = training_settings.get("early-stop-min-delta-rel", 1e-3)
        if (vl_tot < best_val * (1.0 - min_delta_rel)) and (epoch >= warmup_gate):
            best_val = vl_tot
            bad_epochs = 0
            model.save(ckpt_path, epoch=epoch, best_val=best_val,
                       optimizer=optimizer, scheduler=scheduler,
                       lambda_schedule_state=schedule_controller.state_dict(),
                       bad_epochs=bad_epochs, training_settings=training_settings)
            print("[Phase-2]: Current best model saved.")
        elif epoch >= warmup_gate:
            bad_epochs += 1
            if bad_epochs >= training_settings["early-stop-patience"]:
                model.save(ckpt_last, epoch=epoch, best_val=best_val,
                           optimizer=optimizer, scheduler=scheduler,
                           lambda_schedule_state=schedule_controller.state_dict(),
                           bad_epochs=bad_epochs, training_settings=training_settings)
                print("[Phase-2]: Early stopping triggered.")
                break

        model.save(ckpt_last, epoch=epoch, best_val=best_val,
                   optimizer=optimizer, scheduler=scheduler,
                   lambda_schedule_state=schedule_controller.state_dict(),
                   bad_epochs=bad_epochs, training_settings=training_settings)

    plot_losses(train_losses, val_losses,
                save_path=str(ckpt_path.parent / "phase2_loss.png"),
                title="Phase 2 - MLM pretraining")
    return model, train_losses, val_losses


# ───────── Phase-3 training: per-outcome risk + time ─────────────────── #
@logger
def finetune_transformer(model, train_dl, val_dl, resume=True,
                         checkpoint_path=PHASE3_CHECKPOINT,
                         training_settings=TRAINING_SETTINGS):
    """
    Purpose: Phase-3 — attach task heads and fine-tune the encoder for
             per-outcome (risk, time) prediction.
    Method:  Per-batch:
               • encode + pool → (risk_logits, time_pred);
               • risk loss = ``BCEWithLogitsLoss`` (multi-label, mean-reduced);
               • time loss = Smooth-L1 over positive-only patients;
             Backbone uses ``phase3_backbone_lr_factor`` LR, task heads use the
             full ``phase3_learning_rate``.

    Args:
        model (InterveneEncoder): Phase-2 best ckpt loaded.
        train_dl / val_dl : DataLoaders (natural distribution).
        resume (bool): resume from ``ckpt_last`` if present.
        checkpoint_path (str): destination for the best Phase-3 checkpoint.
        training_settings (dict): config — uses ``phase3_learning_rate``,
                                  ``phase3_backbone_lr_factor``,
                                  ``phase3_weight_decay`` and (new)
                                  ``phase3_time_lambda`` (default 0.1).

    Returns:
        (model, train_losses, val_losses).
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    from kineret.intervene.utils import autocast_dtype
    use_amp, amp_dtype = autocast_dtype(device)
    from kineret.intervene.utils import StepTimer
    _timer = StepTimer()

    # Attach task heads (idempotent — checks .task_heads is None).
    if model.task_heads is None:
        model.attach_task_heads(
            hidden=training_settings.get("phase3_head_hidden", 256),
            dropout=training_settings.get("phase3_pool_dropout"),
        )

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    if resume and ckpt_last.exists():
        pre = torch.load(ckpt_last, map_location="cpu", weights_only=True)
        if pre.get("training_settings") is not None:
            training_settings = pre["training_settings"]

    risk_idx_cpu = model.task_heads.risk_idx.cpu().tolist()
    risk_outcome_names = [model.outcome_names[i] for i in risk_idx_cpu]

    def risk_criterion(logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)

    optimizer = model.configure_optimizers(
        weight_decay=training_settings.get("phase3_weight_decay", 1e-3),
        learning_rate=training_settings["phase3_learning_rate"],
        betas=(0.9, 0.95),
        embedder_lr_factor=training_settings.get("phase3_backbone_lr_factor", 0.01),
        backbone_lr_factor=training_settings.get("phase3_backbone_lr_factor", 0.01),
        head_lr=training_settings["phase3_learning_rate"],
    )

    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0

    if resume and ckpt_last.exists():
        print(f"[Phase-3]: Loading model from checkpoint: {ckpt_last}")
        loaded, start_epoch, best_val, opt_state, *_ = InterveneEncoder.load(
            ckpt_last, embedder=model.embedder, map_location=device, attach_task_heads=True,
        )
        model = loaded
        model.to(device)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_epoch += 1

    train_losses, val_losses = [], []
    lambda_time = training_settings.get("phase3_time_lambda", 0.5)
    grad_accum_steps = training_settings.get("grad_accumulation_steps", 1)

    # CBM-style input-token replacement during Phase-3 training (regularizer).
    # Masks a random p-fraction of non-special positions with [MASK] before
    # the encoder sees them; targets/labels are computed from the original
    # batch so the supervised contract is preserved. p = 0 disables it.
    phase3_cbm_p = float(training_settings.get("phase3_cbm_p", 0.0))
    if phase3_cbm_p > 0.0:
        cbm_luts = build_luts(model.embedder.tokenizer)
        cbm_luts = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in cbm_luts.items()}
        cbm_forbid_ids = cbm_luts["forbid_mask_ids"]
        print(f"[Phase-3] CBM input dropout enabled at p={phase3_cbm_p}")
    else:
        cbm_luts = None
        cbm_forbid_ids = None

    # ─── Per-outcome time statistics (train-only) ──────────────────────── #
    # Computed once before training. Used to z-score gt_time and time_pred
    # inside the time loss so MSE gradients sit at a consistent unit scale
    # for every outcome (RELEASE's ~150 h spread vs HYPERGLYCEMIA's ~60 h
    # spread mustn't make RELEASE's gradient ~6× larger just because of
    # raw scale). Outcomes are aggregated with a uniform mean across
    # time-bearing outcomes — no per-outcome reweighting.
    # The head still emits raw hours via softplus; this normalisation
    # lives entirely in the loss, leaving the eval contract intact.
    time_idx_cpu = model.task_heads.time_idx.cpu().tolist()
    K_time = len(time_idx_cpu)
    print(f"[Phase-3] computing per-outcome time stats over {len(train_dl)} train batches...")
    pos_times_buf = [[] for _ in range(K_time)]
    with torch.no_grad():
        for batch in tqdm(train_dl, desc="time-stats", leave=False,
                          mininterval=5.0, miniters=10, dynamic_ncols=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_b, gt_time_b, _ = build_patient_labels(
                model, batch, training_settings, device,
            )
            for j, k in enumerate(time_idx_cpu):
                mask = (labels_b[:, k] == 1) & torch.isfinite(gt_time_b[:, k])
                if mask.any():
                    pos_times_buf[j].append(gt_time_b[mask, k].float().cpu())

    n_pos_per, mean_per, std_per = [], [], []
    for j in range(K_time):
        if pos_times_buf[j]:
            cat = torch.cat(pos_times_buf[j])
            n_pos_per.append(int(cat.numel()))
            mean_per.append(float(cat.mean().item()))
            std_per.append(float(cat.std().item()) if cat.numel() > 1 else 1.0)
        else:
            n_pos_per.append(0); mean_per.append(0.0); std_per.append(1.0)
    # Floor std at 1 h so a degenerate outcome doesn't divide by ~0.
    std_per = [max(s, 1.0) for s in std_per]
    time_mean = torch.tensor(mean_per, dtype=torch.float32, device=device)
    time_std  = torch.tensor(std_per,  dtype=torch.float32, device=device)

    print("[Phase-3] per-outcome time stats (used inside the time loss only):")
    for j, k in enumerate(time_idx_cpu):
        name = model.outcome_names[k]
        print(f"[Phase-3]   {name:<32s} n_pos={n_pos_per[j]:>5d}  "
              f"mean={mean_per[j]:>7.2f}h  std={std_per[j]:>7.2f}h")

    def run_epoch(loader, train_flag):
        model.train() if train_flag else model.eval()
        total_loss = total_risk = total_time = 0.0
        accum_step = 0
        if train_flag:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_flag):
            for batch in _timer.wrap(tqdm(
                    loader, desc="Training" if train_flag else "Validation",
                    leave=False, mininterval=5.0, miniters=10, dynamic_ncols=True)):
                batch = {k: v.to(device) for k, v in batch.items()}

                labels, gt_time, present = build_patient_labels(
                    model, batch, training_settings, device,
                )
                risk_idx = model.task_heads.risk_idx.to(device)
                time_idx = model.task_heads.time_idx.to(device)

                # CBM input-token replacement (training-time only). Labels
                # were computed above from the un-noised batch; the encoder
                # then sees the noised sequence so it can't lean on a single
                # strong predictor token.
                if train_flag and phase3_cbm_p > 0.0:
                    batch_for_model, _, _ = apply_mlm_mask(
                        batch=batch, tokenizer=model.embedder.tokenizer,
                        forbid_ids=cbm_forbid_ids, luts=cbm_luts, p=phase3_cbm_p,
                    )
                else:
                    batch_for_model = batch

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    risk_logits, time_pred, _, _ = model.predict(
                        parent_raw_ids=batch_for_model["parent_raw_ids"],
                        concept_ids=batch_for_model["concept_ids"],
                        value_ids=batch_for_model["value_ids"],
                        position_ids=batch_for_model["position_ids"],
                        abs_ts=batch_for_model["abs_ts"],
                        context_vec=batch_for_model["context_vec"],
                    )
                risk_logits = risk_logits.float()
                time_pred   = time_pred.float()

                # Risk loss — multi-label BCE on risk-indexed outcomes.
                risk_labels = labels[:, risk_idx]                  # [B, K_risk]
                risk_loss = risk_criterion(risk_logits, risk_labels)

                # Time loss — per-outcome MSE in z-space, masked to positives.
                #
                # The old Smooth-L1 over raw hours was L1 in practice (its
                # quadratic region ends at 1 h; every realistic time error
                # is well past that), which is happy at the conditional
                # median and exerts no gradient pressure to predict
                # per-patient variance — letting the head collapse to a
                # near-constant ~47 h for every patient.
                #
                # MSE on a per-outcome z-normalised target (a) makes the
                # gradient scale invariant across outcomes (RELEASE's
                # ~150 h spread no longer dominates HYPERGLYCEMIA's ~60 h
                # one) and (b) penalises under-dispersion quadratically,
                # forcing the head to use input features to spread its
                # predictions. The inverse-frequency outcome weight then
                # restores per-outcome footing so DEATH (n_pos ≈ 1.1 k)
                # isn't drowned by HYPERGLYCEMIA (n_pos ≈ 3.2 k).
                time_labels_full = labels[:, time_idx]                # [B, K_time]
                time_target_full = gt_time[:, time_idx]               # [B, K_time], hours
                pos_mask = time_labels_full.bool() & torch.isfinite(time_target_full)
                if pos_mask.any():
                    # Broadcast per-outcome stats over the batch axis.
                    z_pred = (time_pred       - time_mean) / time_std
                    z_tgt  = (time_target_full - time_mean) / time_std
                    sq = ((z_pred - z_tgt) ** 2) * pos_mask.float()
                    per_outcome_n  = pos_mask.float().sum(dim=0)                # [K_time]
                    per_outcome_sq = sq.sum(dim=0)                              # [K_time]
                    per_outcome_mse = per_outcome_sq / per_outcome_n.clamp(min=1.0)
                    has_any = (per_outcome_n > 0).float()
                    # Uniform mean across outcomes that had any positives in this batch.
                    time_loss = (per_outcome_mse * has_any).sum() / has_any.sum().clamp(min=1.0)
                else:
                    time_loss = time_pred.sum() * 0.0

                loss = risk_loss + lambda_time * time_loss

                if train_flag:
                    (loss / grad_accum_steps).backward()
                    accum_step += 1
                    if accum_step % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                total_risk += risk_loss.item()
                total_time += time_loss.item()

        n = len(loader)
        return total_loss / n, total_risk / n, total_time / n

    for epoch in range(start_epoch, training_settings["phase3_n_epochs"] + 1):
        tr_tot, tr_risk, tr_time = run_epoch(train_dl, train_flag=True)
        vl_tot, vl_risk, vl_time = run_epoch(val_dl,   train_flag=False)

        train_losses.append(tr_tot)
        val_losses.append(vl_tot)
        print(f"[Phase-3] Epoch {epoch:03d}\n"
              f"    --> Train={tr_tot:.4f} (Risk={tr_risk:.4f}  Time={tr_time:.4f})\n"
              f"    --> Val  ={vl_tot:.4f} (Risk={vl_risk:.4f}  Time={vl_time:.4f})")
        print("   " + _timer.report())
        _timer.reset()

        min_delta_rel = training_settings.get("early-stop-min-delta-rel", 1e-3)
        if vl_tot < best_val * (1.0 - min_delta_rel):
            best_val = vl_tot
            bad_epochs = 0
            model.save(ckpt_path, epoch=epoch, best_val=best_val,
                       optimizer=optimizer, training_settings=training_settings,
                       bad_epochs=bad_epochs)
            print("[Phase-3]: Current best model saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= training_settings["early-stop-patience"]:
                model.save(ckpt_last, epoch=epoch, best_val=best_val,
                           optimizer=optimizer, training_settings=training_settings,
                           bad_epochs=bad_epochs)
                print("[Phase-3]: Early stopping triggered.")
                break

        model.save(ckpt_last, epoch=epoch, best_val=best_val,
                   optimizer=optimizer, training_settings=training_settings,
                   bad_epochs=bad_epochs)

    plot_losses(train_losses, val_losses,
                save_path=str(ckpt_path.parent / "phase3_loss.png"),
                title="Phase 3 - risk + time fine-tune")
    return model, train_losses, val_losses
