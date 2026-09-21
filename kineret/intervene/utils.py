"""
utils.py
==============

General util functions for the package
"""
import sys
import os
import tempfile
import datetime
import functools
import inspect
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Optional

# ───────── local code ─────────────────────────────────────────────────── #
from kineret.intervene.config.dataset_config import (
    ADMISSION_TOKEN, TERMINAL_OUTCOMES, OUTCOMES, MEAL_TOKENS
)
from kineret.intervene.schedulers import linear_schedule


def set_seed(seed: int):
    """Seed all RNGs (python/numpy/torch/cuda) for reproducible runs.

    Called at the start of every model constructor and every training-phase
    entry point so that a fixed config `SEED` makes init + dataloader shuffle +
    sampler draws + dropout reproducible across runs, and varying `SEED`
    produces independent runs (used by the multi-seed confidence study).
    Not bitwise-deterministic on GPU (we do not force deterministic kernels, to
    avoid erroring on ops without deterministic impls) — but removes the
    init/training-stochasticity confound that otherwise dominates run-to-run
    variance.
    """
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TeeStream:
    """Wraps sys.stdout so every write goes to both the terminal and a log file."""

    def __init__(self, log_path, original_stream):
        self._original = original_stream
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, text):
        try:
            self._original.write(text)
        except UnicodeEncodeError:
            # Kineret tweak: the training banners contain Greek and box-drawing
            # characters that a legacy Windows console (cp1252) cannot encode.
            # A log line is not worth killing a training run over -- transliterate
            # what the terminal cannot show. The utf-8 log file keeps the original.
            encoding = getattr(self._original, "encoding", None) or "ascii"
            self._original.write(text.encode(encoding, errors="replace").decode(encoding))
        self._file.write(text)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Proxy everything else (isatty, fileno, etc.) to the real stream so tqdm
    # and other tools that inspect stdout continue to work correctly.
    def __getattr__(self, name):
        return getattr(self._original, name)


_active_tee: Optional[_TeeStream] = None  # shared across all decorated training functions


def _ensure_tee_active():
    global _active_tee
    if _active_tee is not None:
        return
    # Tee log goes to the system temp dir (local fs) — the workspace mfs has intermittent
    # OSError [Errno 5] that crashes training mid-run. stdout redirect already captures everything.
    # Per-uid filename so a stale root-owned training.log from a prior run on a shared Linux pod
    # does not block this process with PermissionError. getuid is POSIX-only — fall back to pid
    # on Windows so local validation runs.
    _uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    log_path = os.path.join(tempfile.gettempdir(), f"training_{_uid}.log")
    _active_tee = _TeeStream(log_path, sys.stdout)
    sys.stdout = _active_tee
    print(f"[Logger] Logging to: {log_path}")


def logger(func):
    """
    Decorator for training functions.  On first call it activates the stdout
    tee (append mode).  Every call prints a timestamped header containing the
    function name, model config, and training settings so the log is
    self-contained and searchable across runs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ensure_tee_active()

        # Pull training_settings out of the call if present
        try:
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            ts = bound.arguments.get("training_settings")
        except Exception:
            ts = None

        try:
            from kineret.intervene.config.model_config import MODEL_CONFIG
        except Exception:
            MODEL_CONFIG = None

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  {func.__name__}  |  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if MODEL_CONFIG:
            print(f"  model config      : {MODEL_CONFIG}")
        if ts:
            print(f"  training settings : {ts}")
        print(sep)

        return func(*args, **kwargs)
    return wrapper






def get_temporal_soft_targets(
    target_ids: torch.Tensor,
    all_abs_ts: torch.Tensor,
    query_abs_ts: torch.Tensor,
    padding_idx: int,
    vocab_size: int,
    tau: torch.Tensor,
    horizon: float,
) -> torch.Tensor:
    """
    Soft-kernel LM-head BCE targets — exp59's two-tier window replaced by a
    learnable per-token-class decay constant.

    For each query step t and each token id v in the vocabulary:
        target[b, t, v] = clamp_{0..1}( sum_{s : 0 < dt(t,s) <= horizon}
                                        exp(-dt(t,s) / tau[v])
                                        * 1[target_ids[b, s] == v] )

    Matches the formula already used by `get_future_outcome_targets` for the
    outcome head, extended to the full LM-head vocabulary. Implementation
    uses scatter_add along the V dimension to avoid materialising a
    [B, T, V] one-hot intermediate — memory cost is the same as the binary
    version's [B, T_q, V] target tensor, plus a [B, T_q, T] decay matrix.

    NOTE: NOT decorated with @torch.no_grad(): gradient flows through tau so
    the learnable `log_tau_lm` parameter trains end-to-end with the LM-head
    BCE. The binary `get_temporal_multi_hot_targets` path remains
    non-differentiable (boolean → float cast) so removing its no_grad would
    be a no-op.

    Args:
        target_ids:    [B, T] token ids whose occurrences mark positives.
        all_abs_ts:    [B, T] absolute timestamps (normalised, same units as horizon).
        query_abs_ts:  [B, T_q] query timestamps.
        padding_idx:   PAD id. PAD positions in target_ids contribute 0, and
                       target[..., padding_idx] is zeroed.
        vocab_size:    V.
        tau:           [V] per-token-class decay constants (positive, same
                       normalised units as horizon). Differentiable.
        horizon:       Hard horizon — positives at dt > horizon contribute 0.

    Returns:
        FloatTensor [B, T_q, V] with soft targets in [0, 1].
    """
    B, T_all = target_ids.shape
    T_q = query_abs_ts.size(1)
    V = vocab_size
    device = target_ids.device

    # Per-source-position tau: tau_per_s[b, s] = tau[target_ids[b, s]].
    # Clamp index to V-1 so the lookup is safe even if rare ids land outside.
    safe_ids = target_ids.clamp(0, V - 1)
    tau_per_s = tau[safe_ids]  # [B, T_all], differentiable

    # Δt: [B, T_q, T_all]. Positive = future.
    dt = all_abs_ts.unsqueeze(1) - query_abs_ts.unsqueeze(2)
    in_horizon = (dt > 0) & (dt <= horizon)

    # decay[b, t, s] = exp(-Δt / tau_per_s[b, s]) inside the horizon, else 0.
    # Clamp tau ≥ 1e-6 to keep the division finite if any tau gets pushed to 0.
    decay = torch.exp(-dt / tau_per_s.unsqueeze(1).clamp(min=1e-6))
    decay = decay.masked_fill(~in_horizon, 0.0)

    # Source-side PAD mask: PAD never contributes a positive.
    if 0 <= padding_idx < V:
        nonpad_src = (target_ids != padding_idx).to(decay.dtype)
        decay = decay * nonpad_src.unsqueeze(1)

    # Scatter-add into the V dim using target_ids as the column index.
    target = torch.zeros(B, T_q, V, device=device, dtype=decay.dtype)
    idx = target_ids.unsqueeze(1).expand(B, T_q, T_all)
    target.scatter_add_(2, idx, decay)
    target = target.clamp(0.0, 1.0)

    if 0 <= padding_idx < V:
        pad_mask = torch.ones(V, device=device, dtype=target.dtype)
        pad_mask[padding_idx] = 0.0
        target = target * pad_mask

    return target


@torch.no_grad()
def get_future_outcome_targets(
    target_ids: torch.Tensor,      # [B, T] token ids
    outcome_ids: list,        # [K] list of outcome token IDs
    all_abs_ts: Optional[torch.Tensor] = None,  # [B, T] absolute timestamps
    query_abs_ts: Optional[torch.Tensor] = None, # [B, T_q] query timestamps
    tau = None,        # scalar OR [K] tensor — decay constant in normalised units
    horizon: Optional[float] = None,  # max lookahead horizon (hours / 336)
) -> torch.Tensor:
    """
    Builds time-aware outcome targets for auxiliary prediction head.

    Two modes:
    
    1. **Binary mode** (all_abs_ts=None): 
       target[b, t, k] = 1 if outcome_ids[k] appears in position_ids[b, t+1:] (anywhere after t).
       
    2. **Time-decayed mode** (all_abs_ts provided):
       target[b, t, k] = soft score ∈ [0, 1]:
           sum_s { exp(-dt(t,s) / tau) * 1[token_s == outcome_k] }.clamp(0, 1)
       where dt(t,s) is filtered by 0 < dt <= horizon.
       Maximum signal for outcomes very soon; decays exponentially to zero.

    Args:
        target_ids: [B, T] token sequence to search for outcomes.
        outcome_ids: [K] list of outcome token IDs.
        all_abs_ts: [B, T] absolute timestamps aligned with target_ids. If provided, enables time decay.
        query_abs_ts: [B, T_q] query timestamps. If omitted, uses all_abs_ts (same shape as target_ids).
        tau: Decay time constant (same units as timestamps). Only used if all_abs_ts is provided.
        horizon: Max lookahead window (same units as timestamps). Only used if all_abs_ts is provided.

    Returns:
        FloatTensor [B, T_q, K] with outcome probabilities (0/1 for binary, soft [0..1] for decayed).
    """
    # B, T = target_ids.shape
    K = len(outcome_ids)
    device = target_ids.device
    
    # Build outcome match matrix: [B, T, K]
    # matches[b, t, k] = 1.0 if target_ids[b, t] == outcome_ids[k]
    out_tensor = torch.tensor(outcome_ids, device=device, dtype=torch.long).view(1, 1, K)
    matches = (target_ids.unsqueeze(-1) == out_tensor).float()  # [B, T, K]

    # Binary mode: no time information
    if all_abs_ts is None:
        # Shift matches to get "future presence": target[t] = any match at s > t
        future_matches = torch.zeros_like(matches)
        future_matches[:, :-1, :] = matches[:, 1:, :]
        future_presence = future_matches.flip(dims=[1]).cummax(dim=1).values.flip(dims=[1])
        return future_presence

    # Time-decayed mode
    if query_abs_ts is None:
        query_abs_ts = all_abs_ts
    
    if tau is None or horizon is None:
        raise ValueError(
            "tau and horizon must be provided when all_abs_ts is given. "
            "Use model_config.TRAINING_SETTINGS for defaults."
        )

    # T_q = query_abs_ts.size(1)
    # Compute time differences: [B, T_q, T]
    dt = all_abs_ts.unsqueeze(1) - query_abs_ts.unsqueeze(2)

    # Horizon filtering: 0 < dt <= horizon
    in_horizon = (dt > 0) & (dt <= horizon)

    # tau may be a scalar or a per-outcome tensor of shape [K]. Per-outcome tau lets
    # each outcome adapt its own decay timescale — RELEASE-type events have different
    # dynamics from clinical complications, and a single scalar undersupervises one.
    is_per_k_tau = torch.is_tensor(tau) and tau.dim() == 1 and tau.numel() == K

    if not is_per_k_tau:
        decay_weights = torch.exp(-dt / tau).masked_fill(~in_horizon, 0.0)  # [B, T_q, T]
        outcome_targets = torch.bmm(decay_weights, matches).clamp(0.0, 1.0)
    else:
        # Per-outcome decay: bmm in a loop over K to keep memory bounded.
        # Out-of-horizon entries are suppressed with masked_fill, NOT multiplication —
        # exp(-dt/tau) can overflow to inf on padded/out-of-horizon dt values, and
        # 0 * inf would produce NaN. masked_fill replaces those entries cleanly.
        outcome_cols = []
        for k_idx in range(K):
            decay_k = torch.exp(-dt / tau[k_idx]).masked_fill(~in_horizon, 0.0)  # [B,T_q,T]
            col = torch.bmm(decay_k, matches[..., k_idx:k_idx + 1]).clamp(0.0, 1.0)  # [B,T_q,1]
            outcome_cols.append(col)
        outcome_targets = torch.cat(outcome_cols, dim=-1)

    return outcome_targets


def set_embedder_frozen(model, freeze: bool):
    for p in model.embedder.parameters():
        p.requires_grad = not freeze
    model.embedder.eval() if freeze else model.embedder.train()


def _build_interval_partner_pos(pos_ids, is_s_lut, is_e_lut, base_lut):
    """
    For every interval-endpoint position in the batch, compute its partner-
    endpoint position (same batch row, same base id, opposite role, k-th
    occurrence paired with k-th occurrence in temporal order).

    Fully vectorised — no Python loop over batch or bases. Two argsorts +
    one cummax over the flattened interval positions of the whole batch.

    Args:
        pos_ids   : LongTensor [B, T]   — per-position token ids.
        is_s_lut  : BoolTensor [V]      — luts["is_start"].
        is_e_lut  : BoolTensor [V]      — luts["is_end"].
        base_lut  : LongTensor [V]      — luts["base_id"]; -1 for non-interval tokens.

    Returns:
        partner_pos : LongTensor [B, T] — for each interval endpoint, the
            time-index of its matching partner in the same batch row. -1 for
            non-interval positions or unmatched endpoints.
    """
    B, T = pos_ids.shape
    device = pos_ids.device

    is_s_at = is_s_lut[pos_ids]        # [B, T]
    is_e_at = is_e_lut[pos_ids]
    base_at = base_lut[pos_ids]        # [B, T], -1 if non-interval
    is_int  = (is_s_at | is_e_at) & (base_at >= 0)

    partner_pos = torch.full((B, T), -1, dtype=torch.long, device=device)
    if not is_int.any():
        return partner_pos

    # Flatten interval positions only.
    flat_b = torch.arange(B, device=device).unsqueeze(1).expand(B, T).reshape(-1)
    flat_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T).reshape(-1)
    is_int_flat = is_int.reshape(-1)

    int_b    = flat_b[is_int_flat]                        # [N_int]
    int_t    = flat_t[is_int_flat]
    int_base = base_at.reshape(-1)[is_int_flat]
    int_is_s = is_s_at.reshape(-1)[is_int_flat]

    NB = int(int_base.max().item()) + 1                   # number of base slots

    # ---- Pass 1: assign a temporal rank within each (b, base, is_start) group ----
    # Sort the interval positions by (b, base, is_start, t) using a single
    # composite key. After this sort, positions sharing (b, base, is_start) are
    # consecutive in temporal order.
    sort_key1 = (((int_b * NB + int_base) * 2) + int_is_s.long()) * T + int_t
    order1 = torch.argsort(sort_key1)
    s_b    = int_b[order1]
    s_t    = int_t[order1]
    s_base = int_base[order1]
    s_is_s = int_is_s[order1]

    # cumcount within each (b, base, is_start) group via cummax of boundary positions
    group_id   = (s_b * NB + s_base) * 2 + s_is_s.long()
    positions  = torch.arange(group_id.numel(), device=device)
    boundary   = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        group_id[1:] != group_id[:-1],
    ])
    last_bnd   = torch.cummax(positions * boundary.long(), dim=0)[0]
    rank_in_group = positions - last_bnd                  # 0, 1, 2, … within each group

    # ---- Pass 2: re-sort by (b, base, rank) so START_k and END_k of the same
    # (b, base) become adjacent; with a stable argsort they keep is_start order
    # within ties. Adjacent items that share (b, base, rank) and differ in
    # is_start are a pair.
    sort_key2 = (s_b * NB + s_base) * (T + 1) + rank_in_group
    order2    = torch.argsort(sort_key2, stable=True)
    r_b    = s_b[order2]
    r_t    = s_t[order2]
    r_base = s_base[order2]
    r_rank = rank_in_group[order2]
    r_is_s = s_is_s[order2]

    if r_b.numel() < 2:
        return partner_pos

    same_b    = r_b[1:]    == r_b[:-1]
    same_base = r_base[1:] == r_base[:-1]
    same_rank = r_rank[1:] == r_rank[:-1]
    diff_role = r_is_s[1:] != r_is_s[:-1]
    is_pair_lo = same_b & same_base & same_rank & diff_role   # [N_int - 1]

    if is_pair_lo.any():
        lo = is_pair_lo.nonzero(as_tuple=True)[0]
        hi = lo + 1
        partner_pos[r_b[lo], r_t[lo]] = r_t[hi]
        partner_pos[r_b[hi], r_t[hi]] = r_t[lo]

    return partner_pos


def apply_cbm(batch, tokenizer, forbid_ids, luts, p=0.25):
    """
    Curriculum by Masking — atomic-interval variant.

    For interval tokens (those with `*_START` / `*_END` suffixes), the START and
    matching END carry the state in their name (e.g.
    GLUCOSE_STATE_High_START / _END). If only one endpoint of an interval is
    masked the interval is half-broken; masking both leaves the model with a
    clean "this measurement is missing" signal. We sample p% of eligible
    positions and, when a sampled position is an interval endpoint, mask its
    matching partner in the same batch row atomically. Standalone (non-
    interval) eligible tokens are masked individually as before.

    Args:
        batch       : dict of [B, T] tensors (position_ids, parent_raw_ids,
                       concept_ids, value_ids, …).
        tokenizer   : EMRTokenizer.
        forbid_ids  : LongTensor of token ids that must never be masked
                       (PAD/MASK/CTX/ADMISSION/TERMINALS/OUTCOMES/MEALS).
                       NOTE: must NOT include START/END ids since the fix —
                       those are handled atomically.
        luts        : dict from `_compute_token_lookups`; must contain
                       `is_start`, `is_end`, `base_id`.
        p           : masking ratio of eligible positions.

    Returns:
        batch (modified in place at the same keys).

    Cost: two argsorts and one cummax over the flattened interval positions;
    no Python loop over the batch dimension.
    """
    pos_ids = batch["position_ids"]
    device  = pos_ids.device
    B, T    = pos_ids.shape

    mask_tok = tokenizer.mask_token_id
    pad_id   = tokenizer.pad_token_id

    # Eligible positions: not in forbid list and not PAD.
    forbid = torch.zeros(tokenizer.token_weights.numel(), dtype=torch.bool, device=device)
    forbid[pad_id]  = True
    forbid[mask_tok] = True
    forbid[forbid_ids] = True
    eligible = ~forbid[pos_ids]                            # [B, T]

    # Sample exactly p * (#eligible) positions.
    E = int(eligible.sum().item())
    N = int(round(p * E))
    if N == 0:
        return batch

    idx  = eligible.nonzero(as_tuple=False)                # [E, 2]
    perm = torch.randperm(E, device=device)[:N]
    pick = idx[perm]                                       # [N, 2]
    to_mask = torch.zeros_like(eligible)
    to_mask[pick[:, 0], pick[:, 1]] = True

    # Atomic-interval pair completion. Precompute partner-position table
    # once (vectorised), then a single gather to flip the partners of all
    # picked endpoints.
    partner_pos = _build_interval_partner_pos(
        pos_ids,
        luts["is_start"].to(device),
        luts["is_end"].to(device),
        luts["base_id"].to(device),
    )
    partner_t = partner_pos[pick[:, 0], pick[:, 1]]        # [N], -1 if no partner
    has_partner = partner_t >= 0
    to_mask[pick[has_partner, 0], partner_t[has_partner]] = True

    # Apply the mask.
    pos_ids_out = pos_ids.clone()
    raw_ids     = batch["parent_raw_ids"].clone()
    con_ids     = batch["concept_ids"].clone()
    val_ids     = batch["value_ids"].clone()
    pos_ids_out[to_mask]    = mask_tok
    raw_ids[to_mask, :]     = mask_tok
    con_ids[to_mask]        = mask_tok
    val_ids[to_mask]        = mask_tok

    batch["position_ids"]   = pos_ids_out
    batch["parent_raw_ids"] = raw_ids
    batch["concept_ids"]    = con_ids
    batch["value_ids"]      = val_ids
    return batch


def apply_mlm_mask(batch, tokenizer, forbid_ids, luts, p=0.15):
    """
    Purpose: BERT-style masked language modelling masker for the EMR encoder.
    Method:  Builds on apply_cbm — samples p% of eligible (b, t) positions and
             atomically masks both endpoints of any picked interval token via
             the precomputed partner-position table.  Unlike apply_cbm, the
             replacement preserves interval structure on the input side and
             the function also returns:
               • target_ids — the ORIGINAL position_ids (used as MLM targets)
               • mlm_mask   — bool [B, T], True at positions to score in CE

    Replacement rule (per position):
        original *_START          → [MASK_INTERVAL_START]
        original *_END            → [MASK_INTERVAL_END]
        any other maskable token  → [MASK]

    All four hierarchical input streams (parent_raw_ids, concept_ids,
    value_ids, position_ids) are replaced with the same mask id so the
    embedder cannot leak the original concept through the lower hierarchy.

    Args:
        batch       (dict):   keys ['position_ids', 'parent_raw_ids',
                              'concept_ids', 'value_ids', ...] all on the same
                              device, position_ids shaped [B, T].
        tokenizer   (EMRTokenizer): must expose pad_token_id, mask_token_id,
                              mask_interval_start_id, mask_interval_end_id.
        forbid_ids  (LongTensor): tokens never masked (PAD/MASK/CTX/ADMISSION/
                              TERMINALS/OUTCOMES/MEALS) — same as apply_cbm.
        luts        (dict):   from build_luts(); requires 'is_start', 'is_end',
                              'base_id'.
        p           (float):  fraction of eligible positions to sample.

    Returns:
        batch     (dict): masked in place; same keys as input.
        target_ids (LongTensor): [B, T] copy of the ORIGINAL position_ids
                                (before masking) — feed into MLM CE.
        mlm_mask   (BoolTensor): [B, T] True at every masked position
                                (including partner endpoints).
    """
    pos_ids = batch["position_ids"]
    device  = pos_ids.device
    B, T    = pos_ids.shape

    target_ids = pos_ids.clone()  # original token ids preserved for the CE target

    mask_tok           = tokenizer.mask_token_id
    mask_int_start_tok = tokenizer.mask_interval_start_id
    mask_int_end_tok   = tokenizer.mask_interval_end_id
    pad_id             = tokenizer.pad_token_id

    # Eligible positions — same filter as apply_cbm.
    forbid = torch.zeros(tokenizer.token_weights.numel(), dtype=torch.bool, device=device)
    forbid[pad_id] = True
    forbid[mask_tok] = True
    forbid[mask_int_start_tok] = True
    forbid[mask_int_end_tok] = True
    forbid[forbid_ids] = True
    eligible = ~forbid[pos_ids]  # [B, T]

    mlm_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    E = int(eligible.sum().item())
    N = int(round(p * E))
    if N == 0:
        return batch, target_ids, mlm_mask

    idx  = eligible.nonzero(as_tuple=False)
    perm = torch.randperm(E, device=device)[:N]
    pick = idx[perm]
    mlm_mask[pick[:, 0], pick[:, 1]] = True

    # Atomic-interval completion — pair the partner endpoint of any picked
    # interval token so the model never sees half an interval.
    partner_pos = _build_interval_partner_pos(
        pos_ids,
        luts["is_start"].to(device),
        luts["is_end"].to(device),
        luts["base_id"].to(device),
    )
    partner_t = partner_pos[pick[:, 0], pick[:, 1]]
    has_partner = partner_t >= 0
    if has_partner.any():
        mlm_mask[pick[has_partner, 0], partner_t[has_partner]] = True

    # Determine which mask token to use per masked position based on the
    # ORIGINAL token role (interval START / END / plain).
    is_s_at = luts["is_start"][pos_ids]
    is_e_at = luts["is_end"][pos_ids]

    # Non-interval replacement: generic [MASK]. (Hierarchical/HEART-family
    # replacement was tested in experiment i1-hier and DISCARDED.)
    non_interval_repl = torch.full_like(pos_ids, mask_tok)

    repl_pos = torch.where(
        is_s_at & mlm_mask, torch.full_like(pos_ids, mask_int_start_tok),
        torch.where(
            is_e_at & mlm_mask, torch.full_like(pos_ids, mask_int_end_tok),
            non_interval_repl,
        ),
    )

    pos_ids_out = pos_ids.clone()
    raw_ids     = batch["parent_raw_ids"].clone()
    con_ids     = batch["concept_ids"].clone()
    val_ids     = batch["value_ids"].clone()

    pos_ids_out[mlm_mask] = repl_pos[mlm_mask]
    con_ids[mlm_mask]     = repl_pos[mlm_mask]
    val_ids[mlm_mask]     = repl_pos[mlm_mask]
    # parent_raw_ids is [B, T, P] — broadcast the per-token replacement across P.
    raw_ids[mlm_mask, :]  = repl_pos[mlm_mask].unsqueeze(-1)

    batch["position_ids"]   = pos_ids_out
    batch["parent_raw_ids"] = raw_ids
    batch["concept_ids"]    = con_ids
    batch["value_ids"]      = val_ids
    return batch, target_ids, mlm_mask


def mix_with_predictions(
        gt_ids: torch.LongTensor,
        pred_ids: torch.LongTensor,
        epoch: int,
        warmup_epochs: int,
        protected_ids: torch.BoolTensor,
        max_rate: float=0.3
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Utility to mix ground-truth and predicted tokens in-batch,
    while never replacing protected token IDs (e.g., START/END,
    MEAL, OUTCOME, NULL, PAD, ADMISSION, CTX).
    Will increase replacement ratio during warmup.

    Args:
      gt_ids        : [B, T] LongTensor of ground-truth token IDs
      pred_ids      : [B, T] LongTensor of argmax predictions
      epoch         : int [0, max_epochs], Current epochs in training process
      warmup_epochs : int, Number of warmup epochs in training process
      protected_ids : [V] BoolTensor, True for tokens to keep from GT
      max_rate      : float, Max ratio of real predictions in Teacher's Forcing batch.


    Returns:
      mixed_ids : [B, T] LongTensor
      mix_mask  : [B, T] BoolTensor, True where pred_ids replaced GT
    NOTE: Currently not used in the codebase.
    """    
    device = gt_ids.device
    B, T = gt_ids.shape
    ss_rate = linear_schedule(epoch, 0, warmup_epochs, max_rate)

    # 1) Random swap mask
    rand_mask = torch.rand(B, T, device=device) < ss_rate
    # 2) Build safe-to-mix mask
    safe_mask = ~protected_ids[gt_ids]
    # 3) Final mix mask
    mix_mask = rand_mask & safe_mask
    mixed_ids = torch.where(mix_mask, pred_ids, gt_ids)
    return mixed_ids, mix_mask


def build_luts(tokenizer):
    """
    Pre-compute all LUTs (lookup tensors) needed for:
      • legality masks (intervals + meals + value-conflict)
      • CBM masking forbid list
      • Inference legality

    Returns
    -------
    luts : dict
        {
        # per-token
        "is_start"          : Bool[V]
        "is_end"            : Bool[V]
        "base_id"           : Long[V]   (-1 if not interval token)
        "meal_rank"         : Long[V]   (-1 non-meal, else 0..K-1)
        "meal_pred_rank"    : Long[V]   (-1 non-meal)

        # per-base (nb = #interval bases)
        "start_ids_per_base": Long[nb]  id of *_START  (-1 if missing)
        "end_ids_per_base"  : Long[nb]  id of *_END    (-1 if missing)
        "conflict_mat"      : Bool[nb, nb]  (same concept & different value)

        # misc
        "start_ids"         : Long[*]   all start ids (unordered)
        "end_ids"           : Long[*]   all end ids   (unordered)
        "K_meals"           : Long[]    scalar
        "forbid_mask_ids"   : Long[*]   tokens we never CBM-mask
        "predict_block"     : Long[*]   tokens we never predict (PAD/MASK/CTX) - Forbids at all steps.
        }
    """
    V = len(tokenizer.token2id)
    device = torch.device("cpu")  # keep CPU; move to GPU later where needed

    # --- per-token LUTs ------------------------------------------------------
    is_start   = torch.zeros(V, dtype=torch.bool, device=device)
    is_end     = torch.zeros(V, dtype=torch.bool, device=device)
    base_id    = torch.full((V,), -1, dtype=torch.long, device=device)
    tok2concept= torch.full((V,), -1, dtype=torch.long, device=device)
    tok2value  = torch.full((V,), -1, dtype=torch.long, device=device)

    # ---------- detect START/END tokens, map bases, and fill per‑token LUTs ----------
    base2idx = {}
    start_ids_list, end_ids_list = [], []

    for tok, tid in tokenizer.token2id.items():
        # Strip ONLY the suffix for interval tokens
        if tok.endswith("_START"):
            core = tok[:-6]
            is_start[tid] = True
        elif tok.endswith("_END"):
            core = tok[:-4]
            is_end[tid] = True
        else:
            core = tok

        # ----- concept & value ids -----
        parts = core.split("_") # A_STATE_High, A_TREND_Dec, events
        # concept  = everything except the final value segment
        #           e.g.  A_STATE_Low   ->  A_STATE
        #                 A_TREND_inc   ->  A_TREND
        concept_key = "_".join(parts[:-1])
        value_key   = core # Will also represent events, contexts

        # Use position-id on V to mark the unifying mark of that hierarchy
        tok2concept[tid] = tokenizer.concept2id.get(concept_key, -1)
        tok2value[tid]   = tokenizer.value2id.get(value_key,   -1)

        # ----- interval base bookkeeping -----
        if is_start[tid] or is_end[tid]:
            base_idx = base2idx.setdefault(core, len(base2idx))
            base_id[tid] = base_idx
            if is_start[tid]:
                start_ids_list.append(tid)
            else:  # END
                end_ids_list.append(tid)

    # tensors of all *_START / *_END ids (unordered)
    start_ids = torch.tensor(start_ids_list, dtype=torch.long, device=device)
    end_ids   = torch.tensor(end_ids_list,   dtype=torch.long, device=device)

    # ---------- per‑base LUTs ----------
    nb = len(base2idx)
    start_ids_per_base = torch.full((nb,), -1, dtype=torch.long, device=device)
    end_ids_per_base   = torch.full((nb,), -1, dtype=torch.long, device=device)
    base_concept       = torch.full((nb,), -1, dtype=torch.long, device=device)
    base_value         = torch.full((nb,), -1, dtype=torch.long, device=device)

    # Fill per‑base arrays (first seen wins; START/END of same base share concept/value)
    for tid in range(V):
        b = base_id[tid].item()
        if b < 0:
            continue
        if is_start[tid]:
            start_ids_per_base[b] = tid
        elif is_end[tid]:
            end_ids_per_base[b] = tid
        if base_concept[b] < 0:
            base_concept[b] = tok2concept[tid]
        if base_value[b] < 0:
            base_value[b] = tok2value[tid]

    # ---------- conflict matrix: same concept, different value ----------
    if nb > 0:
        conf_mat = (base_concept[:, None] == base_concept[None, :]) & \
                (base_value[:,  None]  != base_value[None,  :])
    else:
        conf_mat = torch.zeros(0, 0, dtype=torch.bool, device=device)

    # --- meals ---------------------------------------------------------------
    meal_rank = torch.full((V,), -1, dtype=torch.long, device=device)
    for r, name in enumerate(MEAL_TOKENS):
        tid = tokenizer.token2id.get(name)
        if tid is not None:
            meal_rank[tid] = r
    K = int(meal_rank.max().item()) + 1 if (meal_rank >= 0).any() else 0

    meal_pred_rank = torch.full((V,), -1, dtype=torch.long, device=device)
    if K > 0:
        meal_mask = meal_rank >= 0
        meal_pred_rank[meal_mask] = (meal_rank[meal_mask] - 1) % K

    # ---- forbid list for CBM ----
    # NOTE: START/END interval markers are NOT forbidden. Both endpoints carry
    # the state in their token name (e.g. GLUCOSE_STATE_High_START / _END), so
    # protecting both leaves CBM with almost nothing to mask on this data. The
    # apply_cbm function pair-masks intervals atomically — when one endpoint is
    # sampled, its matching endpoint in the same sample is also masked, so the
    # interval is either fully present or fully absent, never half-broken.
    forbid = {
        tokenizer.pad_token_id,
        tokenizer.null_token_id,
        tokenizer.token2id.get(ADMISSION_TOKEN),
        *[tokenizer.token2id.get(t) for t in TERMINAL_OUTCOMES],
        *[tokenizer.token2id.get(t) for t in OUTCOMES],
        *[tokenizer.token2id.get(t) for t in MEAL_TOKENS],
    }
    forbid_mask_ids = torch.tensor([tid for tid in forbid if tid is not None],
                                   dtype=torch.long)
    
    # ---- forbid list for Decoder ----
    block_ids = {
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        # Interval mask tokens — never valid prediction targets.
        getattr(tokenizer, "mask_interval_start_id", None),
        getattr(tokenizer, "mask_interval_end_id", None),
    }
    block_ids = [tid for tid in block_ids if tid is not None]

    predict_block = torch.zeros(V, dtype=torch.bool, device=device)
    predict_block[torch.tensor(block_ids, dtype=torch.long, device=device)] = True

    return {
        # per-token
        "is_start": is_start,
        "is_end":   is_end,
        "base_id":  base_id,
        "meal_rank":      meal_rank,
        "meal_pred_rank": meal_pred_rank,
        "tok2concept": tok2concept,   # Long[V], -1 if no concept mapping
        "tok2value":   tok2value,     # Long[V], -1 if no value mapping

        # per-base
        "start_ids_per_base": start_ids_per_base,
        "end_ids_per_base":   end_ids_per_base,
        "conflict_mat":       conf_mat,

        # misc
        "start_ids": start_ids,
        "end_ids":   end_ids,
        "K_meals":   torch.tensor(K, dtype=torch.long, device=device),
        "forbid_mask_ids": forbid_mask_ids,
        "predict_block": predict_block
    }


def init_legality_state_batched(luts: dict, position_ids: torch.LongTensor):
    """
    Compute initial batched legality state from a seed sequence (may be padded).

    Mirrors the per-token scalar state in the old inference helpers but fully vectorized.
    Padding tokens (base_id == -1, meal_rank == -1) are ignored automatically.

    Args:
        luts: dict returned by build_luts(), already on the target device.
        position_ids: [B, T_seed] — may include right-padding with pad_token_id.

    Returns:
        open_counts  : LongTensor [B, nb]  — net open count per interval base.
        next_meal_rank: LongTensor [B]     — required next meal rank (-1 = no meal seen yet,
                                             any meal is allowed as the first one).
    """
    device = position_ids.device
    B, T  = position_ids.shape
    nb    = luts["start_ids_per_base"].numel()
    K     = int(luts["K_meals"].item())

    # ── interval open counts ──────────────────────────────────────────────────
    tok_base = luts["base_id"][position_ids]   # [B, T], -1 for non-interval / pad
    tok_s    = luts["is_start"][position_ids]  # [B, T]
    tok_e    = luts["is_end"  ][position_ids]  # [B, T]

    valid        = tok_base >= 0
    scatter_idx  = tok_base.clamp(min=0)       # avoid -1 index; gated by `valid`

    start_oh = torch.zeros(B, T, nb, dtype=torch.int32, device=device)
    end_oh   = torch.zeros(B, T, nb, dtype=torch.int32, device=device)
    b_idx    = torch.arange(B, device=device)[:, None]
    t_idx    = torch.arange(T, device=device)[None, :]
    start_oh[b_idx, t_idx, scatter_idx] = (tok_s & valid).to(torch.int32)
    end_oh  [b_idx, t_idx, scatter_idx] = (tok_e & valid).to(torch.int32)

    open_counts = (start_oh.sum(dim=1) - end_oh.sum(dim=1)).clamp(min=0).long()  # [B, nb]

    # ── meal cycle state ──────────────────────────────────────────────────────
    next_meal_rank = torch.full((B,), -1, dtype=torch.long, device=device)
    if K > 0:
        mr      = luts["meal_rank"][position_ids]          # [B, T], -1 for non-meal
        is_meal = mr >= 0                                   # [B, T]
        if is_meal.any():
            # Find the LAST meal position per batch item (vectorized argmax trick).
            t_range  = torch.arange(T, dtype=torch.float32, device=device)
            weighted = torch.where(is_meal, t_range.unsqueeze(0), torch.full_like(t_range, -1.0).unsqueeze(0))
            last_t   = weighted.argmax(dim=1)              # [B]
            has_meal = is_meal.any(dim=1)                  # [B]
            last_rank = mr[torch.arange(B, device=device), last_t]   # [B]
            next_meal_rank = torch.where(
                has_meal,
                (last_rank + 1) % K,
                torch.full((B,), -1, dtype=torch.long, device=device)
            )

    return open_counts, next_meal_rank


def build_illegal_mask_batched(luts: dict, open_counts: torch.LongTensor,
                                next_meal_rank: torch.LongTensor,
                                pad_id: int, mask_id: int) -> torch.BoolTensor:
    """
    Build a [B, V] illegal-token mask for the *next* generation step, given the
    current batched legality state.

    This is the batched, stateful counterpart to the scalar ``_build_illegal_mask``
    that was previously inlined in inference.py.

    Args:
        luts           : dict from build_luts(), on the correct device.
        open_counts    : LongTensor [B, nb] — net open-count per interval base.
        next_meal_rank : LongTensor [B]     — required meal rank (-1 = free choice).
        pad_id, mask_id: always-illegal special tokens.

    Returns:
        BoolTensor [B, V], True → token is illegal for that batch item.
    """
    device  = open_counts.device
    B       = open_counts.shape[0]
    V       = int(luts["is_start"].numel())
    nb      = int(luts["start_ids_per_base"].numel())
    K       = int(luts["K_meals"].item())

    illegal = torch.zeros(B, V, dtype=torch.bool, device=device)

    closed = open_counts <= 0   # [B, nb]
    opened = open_counts  > 0   # [B, nb]

    # 1) END illegal when its base is not open
    end_ids_pb  = luts["end_ids_per_base"]    # [nb]
    valid_end   = end_ids_pb >= 0             # [nb]
    if valid_end.any() and closed.any():
        mask           = closed & valid_end.unsqueeze(0)     # [B, nb]
        b_idx, nb_idx  = mask.nonzero(as_tuple=True)
        if b_idx.numel():
            illegal[b_idx, end_ids_pb[nb_idx]] = True

    # 2) START illegal when its base is already open (duplicate start)
    start_ids_pb = luts["start_ids_per_base"]   # [nb]
    valid_start  = start_ids_pb >= 0
    if valid_start.any() and opened.any():
        mask           = opened & valid_start.unsqueeze(0)   # [B, nb]
        b_idx, nb_idx  = mask.nonzero(as_tuple=True)
        if b_idx.numel():
            illegal[b_idx, start_ids_pb[nb_idx]] = True

    # 3) Conflict: START of another value of an already-open concept
    conf_mat = luts["conflict_mat"]   # [nb, nb]
    if valid_start.any() and opened.any() and conf_mat.any():
        oc              = opened.to(torch.float32)             # [B, nb]
        conflict_active = (oc @ conf_mat.to(torch.float32)) > 0   # [B, nb]
        mask            = conflict_active & valid_start.unsqueeze(0)
        b_idx, nb_idx   = mask.nonzero(as_tuple=True)
        if b_idx.numel():
            illegal[b_idx, start_ids_pb[nb_idx]] = True

    # 4) Meal cycle
    if K > 0:
        meal_rank    = luts["meal_rank"]        # [V]
        is_meal_tok  = meal_rank >= 0           # [V]
        if is_meal_tok.any():
            meal_v_ids   = is_meal_tok.nonzero(as_tuple=False).squeeze(-1)  # [n_meals]
            meal_v_ranks = meal_rank[meal_v_ids]                             # [n_meals]
            # Patients that have seen at least one meal must follow the cycle
            has_seen     = next_meal_rank >= 0                               # [B]
            if has_seen.any():
                required     = next_meal_rank[has_seen].unsqueeze(1)         # [B_act, 1]
                ok           = meal_v_ranks.unsqueeze(0) == required         # [B_act, n_meals]
                b_active     = has_seen.nonzero(as_tuple=False).squeeze(-1)  # [B_act]
                bad_b, bad_v = (~ok).nonzero(as_tuple=True)
                if bad_b.numel():
                    illegal[b_active[bad_b], meal_v_ids[bad_v]] = True

    # 5) Always block pad / mask tokens
    illegal[:, pad_id]  = True
    illegal[:, mask_id] = True

    return illegal


def update_legality_state_batched(luts: dict, next_token_ids: torch.LongTensor,
                                   open_counts: torch.LongTensor,
                                   next_meal_rank: torch.LongTensor,
                                   finished: torch.BoolTensor):
    """
    Update open_counts and next_meal_rank in-place after a batch generation step.

    Finished patients are skipped so their state stays frozen.

    Args:
        luts            : dict from build_luts(), on the correct device.
        next_token_ids  : LongTensor [B] — the token chosen for each batch item.
        open_counts     : LongTensor [B, nb] — mutated in-place.
        next_meal_rank  : LongTensor [B]     — mutated in-place.
        finished        : BoolTensor [B]     — True for already-finished patients.

    Returns:
        (open_counts, next_meal_rank) — same tensors, updated in-place.
    """
    K      = int(luts["K_meals"].item())
    device = next_token_ids.device
    B      = next_token_ids.shape[0]

    active     = ~finished
    active_idx = active.nonzero(as_tuple=False).view(-1)   # [B_act]
    if active_idx.numel() == 0:
        return open_counts, next_meal_rank

    active_toks = next_token_ids[active_idx]  # [B_act]

    # ── interval state ────────────────────────────────────────────────────────
    is_s   = luts["is_start"][active_toks]   # [B_act]
    is_e   = luts["is_end"  ][active_toks]   # [B_act]
    b_ids  = luts["base_id" ][active_toks]   # [B_act], -1 if not interval
    valid  = b_ids >= 0

    start_mask = is_s & valid
    if start_mask.any():
        bi = active_idx[start_mask]
        ba = b_ids[start_mask]
        open_counts.index_put_((bi, ba),
                               torch.ones(start_mask.sum(), dtype=open_counts.dtype, device=device),
                               accumulate=True)

    end_mask = is_e & valid
    if end_mask.any():
        bi = active_idx[end_mask]
        ba = b_ids[end_mask]
        open_counts.index_put_((bi, ba),
                               torch.full((end_mask.sum(),), -1, dtype=open_counts.dtype, device=device),
                               accumulate=True)
        open_counts.clamp_(min=0)

    # ── meal cycle ────────────────────────────────────────────────────────────
    if K > 0:
        mr       = luts["meal_rank"][active_toks]   # [B_act]
        is_meal  = mr >= 0
        if is_meal.any():
            meal_active = active_idx[is_meal]
            next_meal_rank[meal_active] = (mr[is_meal] + 1) % K

    return open_counts, next_meal_rank


def build_rep_penalty_batched(last_tokens_batch: list, V: int,
                               window: int = 5, strength: float = 0.6,
                               device=None) -> torch.Tensor:
    """
    Batched repetition-penalty vector for inference.

    Vectorised version of build_rep_penalty for a whole batch of patients.

    Args:
        last_tokens_batch : List[List[int]], one list per patient (newest last).
        V                 : Vocabulary size.
        window, strength  : Same semantics as build_rep_penalty.
        device            : Target device.

    Returns:
        FloatTensor [B, V].
    """
    B   = len(last_tokens_batch)
    rep = torch.zeros(B, V, device=device)
    if strength <= 0 or all(not t for t in last_tokens_batch):
        return rep

    decay = torch.linspace(1.0, 0.2, steps=window, device=device)  # newest=1.0

    for b, last_toks in enumerate(last_tokens_batch):
        if not last_toks:
            continue
        k   = min(window, len(last_toks))
        idx = torch.tensor(last_toks[-k:], dtype=torch.long, device=device)
        rep[b].index_add_(0, idx.flip(0), decay[:k])

    return rep * strength


def compute_legality_masks_tf(position_ids: torch.LongTensor,
                              is_start: torch.BoolTensor,
                              is_end:   torch.BoolTensor,
                              base_id:  torch.LongTensor,
                              start_ids_per_base: torch.LongTensor,
                              end_ids_per_base:   torch.LongTensor,
                              meal_rank: torch.LongTensor,
                              meal_pred_rank: torch.LongTensor,
                              K_meals: torch.Tensor,
                              conflict_mat: torch.BoolTensor,
                              predict_block: torch.BoolTensor):
    """
    Vectorized legality masks from GOLD prefix (teacher forcing).

    illegal[B,T,V]  True → forbid v at step t

    Terms:
    If token== 'GLUCOSE_TREND_inc_START', base(tok) == 'GLUCOSE_TREND_inc'
    If token== 'GLUCOSE_TREND_inc_START', concept(tok) == 'GLUCOSE_TREND'

    Interval logic (per base):
      • END is illegal if base not open yet (You can't see 'GLUCOSE_TREND_inc_END' before opening 
      it 'GLUCOSE_TREND_inc_START'). Enforced using base(tok).
      • START illegal if base already open 
      (You can't have 'GLUCOSE_TREND_inc_START' after 'GLUCOSE_TREND_inc_START' without seeing
        'GLUCOSE_TREND_inc_END'). Enforced using base(tok).
      • START of concept(tok) is illegal if concept(tok) is still open 
        (meaning you can't have 'GLUCOSE_TREND_inc_START' after 'GLUCOSE_TREND_dec_START' without seeing
        'GLUCOSE_TREND_inc_END'). Enforced using concept(tok).

    Meal logic:
            cyclic order; meal m illegal if predecessor rank not seen yet.
      The first meal is never illegal, only the following ones.

    All done without loops over T (only broadcast/cumsums).

    position_ids : [B,T]
    """
    device = position_ids.device
    B, T = position_ids.shape
    V    = is_start.numel()
    nb   = start_ids_per_base.numel()

    # Map tokens → base / start / end
    tok_base = base_id[position_ids]    # [B,T]
    tok_s    = is_start[position_ids]   # [B,T]
    tok_e    = is_end[position_ids]     # [B,T]

    # Build one-hots over bases
    valid = tok_base >= 0
    scatter_idx = tok_base.clone()
    scatter_idx[~valid] = 0
    start_oh = torch.zeros(B, T, nb, device=device, dtype=torch.int16)
    end_oh   = start_oh.clone()
    b_idx = torch.arange(B, device=device)[:,None]
    t_idx = torch.arange(T, device=device)[None,:]
    start_oh[b_idx, t_idx, scatter_idx] = (tok_s & valid).to(start_oh.dtype)
    end_oh  [b_idx, t_idx, scatter_idx] = (tok_e & valid).to(end_oh.dtype)

    # Cumulative sums to know "open count" at each t
    starts_cum = start_oh.cumsum(dim=1)
    ends_cum   = end_oh.cumsum(dim=1)

    # Build "open state before t" by shifting right
    prev_s = torch.zeros_like(starts_cum); prev_s[:,1:,:] = starts_cum[:,:-1,:]
    prev_e = torch.zeros_like(ends_cum);   prev_e[:,1:,:] = ends_cum[:,:-1,:]
    open_before = (prev_s - prev_e) > 0  # [B,T,nb]

    # Prepare legality matrix
    illegal = torch.zeros(B, T, V, device=device, dtype=torch.bool)

    # 1) END rules: illegal if not open_before
    end_ids = end_ids_per_base.view(1,1,nb).expand(B,T,nb)
    illegal.scatter_(2, end_ids, ~open_before)

    # 2) DUP-START: START when *that same* base was already open should be illegal
    #    clamp base_id to ≥0 so gather never errors
    base_idxs = tok_base.clamp(min=0).unsqueeze(-1)   # [B,T,1]
    was_open  = open_before.gather(2, base_idxs).squeeze(-1)  # [B,T]
    dup       = tok_s & (tok_base >= 0) & was_open
    if dup.any():
        b_d, t_d    = dup.nonzero(as_tuple=True)
        start_toks  = position_ids[b_d, t_d]    # the token‐IDs of those STARTs
        illegal[b_d, t_d, start_toks] = True

    # 3) CNF (conflicts): if any other value of same concept was open_before
    if nb > 0 and conflict_mat.any():
        # 3) CNF (conflicts): START of a value is illegal if any conflicting base is already open
        oc = open_before.to(torch.float32)                   # [B,T,nb]
        cm = conflict_mat.to(torch.float32).T                # [nb,nb]
        conflict_active = (oc @ cm) > 0                      # [B,T,nb]

        # only apply to actual START tokens of each base
        conflict_active &= start_oh.bool()                  # [B,T,nb]

        # OR‑add these into the illegal mask at the corresponding START token IDs
        ids = start_ids_per_base                           # shape [nb]
        illegal[:, :, ids] |= conflict_active              # in‑place OR

    # --- Meal logic: allow starting anywhere, then enforce absolute cycle order ---
    if K_meals > 0:
        # 1) build a one‑hot of *where* meals occur in the GOLD prefix
        mr   = meal_rank[position_ids]             # [B,T]
        mask = mr >= 0
        meal_oh = torch.zeros(B, T, K_meals, device=device, dtype=torch.bool)
        idx = mask.nonzero(as_tuple=False)
        if idx.numel():
            b_i, t_i = idx[:,0], idx[:,1]
            meal_oh[b_i, t_i, mr[mask]] = True

        # 2) compute “prefix_seen”: which ranks have appeared *before* time t
        #    (we shift the cumsum right by one)
        cum        = meal_oh.cumsum(dim=1) > 0          # [B,T,K_meals]
        prefix_seen = cum.clone()
        prefix_seen[:,1:,:] = cum[:,:-1,:]
        prefix_seen[:,0,:] = False

        # 3) decide legality:
        #    • if *no* meal has ever been seen in the prefix → any meal is OK
        #    • else → only the immediate successor (pred_rank) is OK
        mtok       = meal_rank >= 0                     # [V]
        if mtok.any():
            v_ids       = mtok.nonzero(as_tuple=False).squeeze(-1)  # meal token IDs
            pred_ranks  = meal_pred_rank[ mtok ]                  # [nv]
            # a) predecessor rule
            ok_pred     = prefix_seen[:,:,pred_ranks]             # [B,T,nv]
            # b) free‐pass if absolutely no meal seen yet
            any_seen    = prefix_seen.any(dim=2, keepdim=True)    # [B,T,1]
            ok_initial  = ~any_seen                                 # [B,T,1]
            ok          = ok_pred | ok_initial                     # [B,T,nv]

            # 4) apply to illegal mask
            illegal[:,:,v_ids] |= ~ok
    
    # ---- Specials never to be predicted ----
    illegal |= predict_block.view(1, 1, -1)   # broadcast [V] -> [B,T,V]

    return illegal


def masked_softmax(logits, allowed):
    """
    Softmax over allowed classes only; zeros on disallowed.
    Safe when a row is fully masked (no legal classes): returns zeros and stops grad.
    """
    allowed = allowed.to(torch.bool)

    # numeric stability: subtract row max
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    # push disallowed far down but finite
    shifted = shifted.masked_fill(~allowed, -1e9)

    # normal softmax on the shifted scores
    lse = torch.logsumexp(shifted, dim=-1, keepdim=True)     # [B,T,1]
    probs = torch.exp(shifted - lse) * allowed               # [B,T,V], zero on disallowed

    # handle rows where *all* entries are disallowed
    any_valid = allowed.any(dim=-1, keepdim=True)            # [B,T,1]
    probs = torch.where(any_valid, probs, torch.zeros_like(probs))
    # stop gradients on fully-masked rows (no learning signal there)
    probs = probs * any_valid.float()
    return probs


def apply_masks_to_logits(logits, illegal_mask):
    """
    logits: [B,T,V] *after* slicing (no [CTX])
    illegal_mask: Bool [B,T,V]

    Sets illegal-token logits to -1e9 so they are suppressed in both
    softmax and BCE without producing NaN gradients.

    Note: bonus boosting (+0.2 for legal-closure tokens) was removed.
    The bonus was a hardcoded nudge toward closing open intervals, but
    it was never calibrated and interfered with BCE learning the same
    signal organically. The illegal hard-mask is sufficient.
    """
    return logits.masked_fill(illegal_mask, -1e9)


def plot_losses(train_losses, val_losses, save_path=None, title=None):
    """
    Purpose: Record the train-vs-validation loss curve for a training phase.
    Method:  Kineret tweak -- the figure is WRITTEN, not shown, unless we are
             running inside a notebook.

             Upstream this ended in a bare `plt.show()`, which is correct in a
             notebook and fatal in a script: with any GUI backend it blocks the
             process on a window nobody is there to close, so a batch run of the
             grid would hang at the end of Phase 1 and look like a slow epoch.
             Writing a PNG next to the checkpoint keeps the diagnostic without
             the deadlock.

             The first epoch is dropped from the curve; its loss is dominated by
             initialisation transients and squashes the informative range.

    Args:
        train_losses (list[float]): Per-epoch training loss.
        val_losses   (list[float]): Per-epoch validation loss.
        save_path    (str|None):    Destination PNG. None skips writing.
        title        (str|None):    Plot title.

    Returns:
        str|None: The path written, or None.
    """
    train_losses = list(train_losses)[1:]
    val_losses = list(val_losses)[1:]
    if not train_losses:
        return None

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training vs. validation loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    written = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        written = save_path

    # Only a live notebook gets an inline figure; everything else closes the
    # figure so the process can exit and so a long sweep does not accumulate
    # hundreds of open figures.
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close(fig)
    return written


def build_rep_penalty(last_tokens, V, window=5, strength=0.6, device=None):
    """
    Soft repetition discourager on inference.
    last_tokens : list[int]   (already generated, newest at the end)
    V           : vocab size
    window      : how many recent tokens we look back
    strength    : scalar multiplier for the penalty (0..1 typical)
    Returns:
        rep_vec : [V] float tensor, 0 for unseen, higher for very recent repeats
    """
    if not last_tokens or strength <= 0:
        return torch.zeros(V, device=device)
    device = device or torch.device("cpu")
    k = min(window, len(last_tokens))
    # decay weights: newest gets 1.0, then 0.8, 0.6, ...
    decay = torch.linspace(1.0, 0.2, steps=window, device=device)[:k]
    idx = torch.tensor(last_tokens[-k:], device=device)

    rep_vec = torch.zeros(V, device=device)
    # reverse so newest aligns with decay[0]
    rep_vec.index_add_(0, idx.flip(0), decay)
    return rep_vec * strength


def compute_soft_outcome_labels(gen_abs_ts_hours, gt_df, outcome_names,
                                 tau_hours, horizon_hours, device):
    """
    Compute soft outcome labels for a single patient's generated trajectory,
    using the same time-decayed formula as Phase-2 training:

        target_k(t) = clamp( Σ_s exp(-dt(t,s)/τ) * 1[token_s == outcome_k], 0, 1 )

    where s iterates over ground-truth FUTURE events within `horizon_hours` of t.
    Used during phase-3 training (finetuning on generated trajectories) to provide a learning signal for outcomes,
    even when the exact outcome token is not generated at the exact time in the future.
    
    Parameters
    ----------
    gen_abs_ts_hours : 1-D tensor of absolute times (hours from admission) for
                       each generated step.
    gt_df            : full (untruncated) token DataFrame for this patient.
                       Expected to have a 'PositionToken' (or 'Token') column and
                       a 'TimePoint' column (normalised to [0,1] by /336).
    outcome_names    : list of outcome token strings (model.outcome_names).
    tau_hours        : decay constant in hours.
    horizon_hours    : max lookahead in hours (TRAINING_SETTINGS['outcome_horizon_hours']).
    device           : torch device for the returned tensor.

    Returns
    -------
    torch.Tensor of shape [T_gen, len(outcome_names)], dtype float32.
    """
    if gt_df is None or gen_abs_ts_hours.numel() == 0:
        return torch.zeros(0, len(outcome_names), device=device)

    T_gen   = gen_abs_ts_hours.shape[0]
    labels  = torch.zeros(T_gen, len(outcome_names))
    tok_col = 'PositionToken' if 'PositionToken' in gt_df.columns else 'Token'

    for k, name in enumerate(outcome_names):
        occ = gt_df[gt_df[tok_col] == name]
        if occ.empty:
            continue
        occ_times = torch.tensor(occ['TimePoint'].values * 336.0, dtype=torch.float32)
        for t_idx in range(T_gen):
            t      = gen_abs_ts_hours[t_idx].item()
            dt     = occ_times - t
            future = (dt > 0) & (dt <= horizon_hours)
            if not future.any():
                continue
            labels[t_idx, k] = torch.clamp(
                torch.exp(-dt[future] / tau_hours).sum(), 0.0, 1.0
            )

    return labels.to(device)


def audit_generated_stream(
        results_df: pd.DataFrame,
        tokenizer,
        token_w: int = 50,        # Max width of the “Token” column
    ) -> None:
    """
    Prints a step-by-step structural audit of a context+generation stream.
    Used for model manual tests after training, to validate poor results.

    After the row-by-row trace it prints the whole DataFrame (Token,
    IsInput, Note).  No return value.
    """
    # ------------- helpers ------------- #
    def _print(idx, tok, is_inp, problems, token_w):
        clipped = (tok[:token_w - 3] + "...") if len(tok) > token_w else tok
        note    = ";".join(problems)
        print(f"{idx:4d}  {clipped:<{token_w}}  {int(is_inp):>5d}  {note}")
    
    # ------------- look‑ups & runtime state ------------- #
    l = build_luts(tokenizer)
    is_start, is_end   = l['is_start'], l['is_end']
    base_id            = l['base_id']
    meal_rank          = l['meal_rank']
    conflict_mat       = l['conflict_mat']
    nb                 = int(l['start_ids_per_base'].numel())
    K                  = int(l['K_meals'].item())

    open_counts = torch.zeros(nb, dtype=torch.int16)
    seen_meals  = torch.zeros(K,  dtype=torch.bool) if K else None

    t2id = tokenizer.token2id
    notes = []                                            # per‑row notes

    # ------------- pretty header for live trace -------- #
    hdr = f"{'Idx':>4}  {'Token':<{token_w}}  IsInp  Note"
    print(hdr)
    print("-" * len(hdr))

    # ------------- iterate over stream ----------------- #
    for idx, (tok, is_inp) in enumerate(zip(results_df['Token'],
                                            results_df['IsInput'])):
        tid = t2id.get(tok, None)
        problems = []

        # UNK
        if tid is None:
            problems.append("UNK")
            notes.append(";".join(problems))
            _print(idx, tok, is_inp, problems, token_w)
            continue

        b = base_id[tid].item()
        s = bool(is_start[tid])
        e = bool(is_end[tid])
        r = meal_rank[tid].item()

        # interval FSM / DUP / CNF
        if e:
            if b < 0 or open_counts[b] == 0:
                problems.append("FSM")
            else:
                open_counts[b] -= 1
        elif s:
            if open_counts[b] > 0:
                problems.append("DUP")
            if conflict_mat.any() and (open_counts > 0).any():
                if (conflict_mat[b] & (open_counts > 0)).any():
                    problems.append("CNF")
            open_counts[b] += 1

        # meal order
        if r >= 0:
            if 'next_meal' not in locals():               # first meal we ever see
                next_meal = (r + 1) % K                   # set expectation
            else:
                if r != next_meal:                        # wrong meal → violation
                    problems.append("MEAL")
                next_meal = (r + 1) % K                   # advance expectation

        notes.append(";".join(problems))
        
    # ------------- final full table -------------------- #
    df_out = results_df.copy()
    df_out['Note'] = notes

    pd.set_option("display.max_rows", None)
    print("\n=== Full trajectory with annotations ===")
    print(df_out[['Token', 'IsInput', 'Note']]
          .to_string(index=True, col_space={'Token': token_w}))


# Phase-1 BCE target builder. Phase 2 has migrated to a learnable per-class soft
# kernel (`get_temporal_soft_targets`); Phase 1 still consumes this hard-window
# multi-hot version because the embedder works better with a static 3 h window
# (soft kernel destabilised Time2Vec — confirmed in exp53/exp70).
@torch.no_grad()
def get_temporal_multi_hot_targets(
    target_ids: torch.Tensor,
    all_abs_ts: torch.Tensor,
    padding_idx: int,
    vocab_size: int,
    window_size: float,
    query_abs_ts: Optional[torch.Tensor] = None,
    outcome_ids: Optional[torch.Tensor] = None,
    next_token_ids: Optional[torch.Tensor] = None,
    wide_token_ids: Optional[torch.Tensor] = None,
    wide_window_size: Optional[float] = None,
    wide_tiers: Optional[list] = None,
) -> torch.Tensor:
    """
    Build temporal multi-hot targets over a future time window using GPU-efficient
    searchsorted + prefix-sum approach.

    For each query step t, marks token ids that appear at any future step s such that:
        0 < (all_abs_ts[s] - query_abs_ts[t]) <= window_size

    Outcome override: when ``outcome_ids`` and ``next_token_ids`` are both provided,
    any query position whose immediate next token is an outcome/terminal token has its
    entire multi-hot row replaced with a 1-hot on just that token. This eliminates
    gradient dilution at exactly the positions where precise supervision matters most.

    IMPORTANT: This function assumes all_abs_ts is NON-DECREASING (sorted) per batch.
    The dataset MUST maintain this ordering. See dataset.py for sorting guarantees.

    Args:
        target_ids: [B, T_all] token ids whose occurrences will be marked as positives.
        all_abs_ts: [B, T_all] absolute timestamps, MUST be non-decreasing per batch.
        padding_idx: Token id used for PAD. PAD is excluded from targets.
        vocab_size: Vocabulary size V for output shape.
        window_size: Future window size (same normalized units as ``all_abs_ts``).
        query_abs_ts: [B, T_q] optional query timestamps. If omitted, uses ``all_abs_ts``.
        outcome_ids: 1-D LongTensor of outcome + terminal token ids. When provided
            together with ``next_token_ids``, enables the 1-hot override.
        next_token_ids: [B, T_q] immediate next token at each query position.
            Use ``padding_idx`` where no next token exists (e.g. last position in phase-1).

    Returns:
        FloatTensor [B, T_q, V] with 0/1 multi-hot labels.
    """
    if query_abs_ts is None:
        query_abs_ts = all_abs_ts

    B, T_all = target_ids.shape
    T_q = query_abs_ts.size(1)

    # GPU-friendly searchsorted + prefix-sum approach (O(B * T * log T) instead of O(B * T^2)).
    # Assumes timestamps are sorted; violation will produce incorrect results silently.
    all_abs_ts = all_abs_ts.contiguous()
    query_abs_ts = query_abs_ts.contiguous()
    left_idx = torch.searchsorted(all_abs_ts, query_abs_ts, right=True)
    right_idx = torch.searchsorted(all_abs_ts, query_abs_ts + window_size, right=True)

    oh = F.one_hot(target_ids.clamp(min=0), num_classes=vocab_size).to(torch.float32)  # [B, T_all, V]
    csum = oh.cumsum(dim=1)

    # Prefix a zero row so window sum is prefix[right] - prefix[left].
    prefix = torch.cat(
        [torch.zeros(B, 1, vocab_size, device=target_ids.device, dtype=csum.dtype), csum],
        dim=1,
    )  # [B, T_all+1, V]

    left = left_idx.clamp(0, T_all).unsqueeze(-1).expand(B, T_q, vocab_size)
    right = right_idx.clamp(0, T_all).unsqueeze(-1).expand(B, T_q, vocab_size)

    future_counts = prefix.gather(1, right) - prefix.gather(1, left)
    multi_hot = (future_counts > 0).to(torch.float32)

    if 0 <= padding_idx < vocab_size:
        multi_hot[..., padding_idx] = 0.0

    # Optionally apply wider future windows to specific subsets of tokens. Two entry
    # points are supported:
    #   - single-tier: pass wide_token_ids + wide_window_size
    #   - multi-tier:  pass wide_tiers = [(ids, window), ...]
    # Each tier reuses the same prefix sums — one extra searchsorted + gather per
    # tier. The wider window provides denser positive signal for rare/critical
    # tokens (e.g. terminals), so the LM head learns to assign higher logits at
    # pre-event positions instead of only the immediate-next position.
    _wide_specs = []
    if wide_token_ids is not None and wide_window_size is not None and wide_token_ids.numel() > 0 and wide_window_size > window_size:
        _wide_specs.append((wide_token_ids, wide_window_size))
    if wide_tiers is not None:
        for _tier_ids, _tier_w in wide_tiers:
            if _tier_ids is not None and _tier_w is not None and _tier_ids.numel() > 0 and _tier_w > window_size:
                _wide_specs.append((_tier_ids, _tier_w))
    for _wide_ids, _w_size in _wide_specs:
        right_wide_idx = torch.searchsorted(all_abs_ts, query_abs_ts + _w_size, right=True)
        right_wide = right_wide_idx.clamp(0, T_all).unsqueeze(-1).expand(B, T_q, vocab_size)
        future_counts_wide = prefix.gather(1, right_wide) - prefix.gather(1, left)
        wide_multi_hot = (future_counts_wide > 0).to(torch.float32)
        if 0 <= padding_idx < vocab_size:
            wide_multi_hot[..., padding_idx] = 0.0
        wide_ids = _wide_ids.to(target_ids.device)
        multi_hot[:, :, wide_ids] = wide_multi_hot[:, :, wide_ids]

    # Outcome 1-hot override: replace the broad window multi-hot with a strict 1-hot at
    # positions where the immediate next token is an outcome or terminal. Outcome tokens
    # are never illegal so the legality mask does not conflict with this override.
    if outcome_ids is not None and next_token_ids is not None and outcome_ids.numel() > 0:
        assert next_token_ids.shape == (B, T_q), (
            f"next_token_ids shape {next_token_ids.shape} must match (B={B}, T_q={T_q})"
        )
        oids = outcome_ids.to(target_ids.device)
        # is_outcome_next[b, q] = True iff next_token_ids[b, q] is in outcome_ids
        is_outcome_next = (next_token_ids.unsqueeze(-1) == oids.view(1, 1, -1)).any(-1)  # [B, T_q]
        if is_outcome_next.any():
            bi, qi = is_outcome_next.nonzero(as_tuple=True)
            tok = next_token_ids[bi, qi]   # [N] the specific outcome token ids
            multi_hot[bi, qi] = 0.0        # wipe window
            multi_hot[bi, qi, tok] = 1.0   # replace with 1-hot

    return multi_hot


# ---------------------------------------------------------------------------
# Tensor helpers shared by the Phase-2 / Phase-3 training loops + diagnose
# ---------------------------------------------------------------------------

def time_to_neighbour_targets(abs_ts, pad_mask, mlm_mask, max_hours=24.0,
                              min_gap_hours=2.0 / 3600.0):
    """
    Purpose: Build per-position local-gap targets for the time_to_neighbour aux.
    Method:  For each masked position, target = time gap (hours) to the nearest
             UNMASKED *non-adjacent* event, i.e. the nearest unmasked event whose
             absolute time differs by more than ``min_gap_hours`` (default 2 s —
             chosen empirically by cutoff sweep: target std plateaus at ~0.109
             from 2 s onward [vs 0.094 at 0 s, degenerate floor 0.02]; 2 s skips
             exact-simultaneous labs + the small sub-2 s TAK-ordering offset tail
             while keeping the local temporal rhythm). Rescaled to hours/max_hours.

             Rationale: the previous definition used the immediately-adjacent
             unmasked event (min of prev/next gap). EMR events cluster at near-
             identical timestamps, so that target was near-degenerate (std≈0.02)
             — a constant predictor solved it and the aux carried no signal.
             Excluding sub-30-min neighbours restores a non-trivial target
             (std≈0.1–0.2), giving the backbone real local-temporal structure to
             learn. (MSE form and /max_hours rescale unchanged.)

             Implementation: vectorised O(T²) masked-min over pairwise |Δt|.
             T is bounded and this is the same order as attention, computed once
             per batch.

    Args:
        abs_ts        (FloatTensor): [B, T] normalised abs timestamps (t / 336 h).
        pad_mask      (BoolTensor):  [B, T] True at non-PAD positions.
        mlm_mask      (BoolTensor):  [B, T] True at MLM-masked positions.
        max_hours     (float):       rescale denominator.
        min_gap_hours (float):       exclude unmasked neighbours closer than this
                                     (the "non-adjacent" cutoff).

    Returns:
        target  (FloatTensor): [B, T] local gap normalised to [0, ~5].
        valid   (BoolTensor):  [B, T] masked, non-PAD positions that have at
                              least one unmasked neighbour > min_gap_hours away.
    """
    unmasked = pad_mask & (~mlm_mask)                                  # [B, T]
    # Pairwise absolute time gaps in hours.
    dt_hours = (abs_ts.unsqueeze(2) - abs_ts.unsqueeze(1)).abs() * 336.0   # [B, T, T]
    # Candidate neighbours j: unmasked and strictly more than min_gap_hours away.
    cand = unmasked.unsqueeze(1) & (dt_hours > min_gap_hours)          # [B, T, T]
    masked_dt = torch.where(cand, dt_hours, dt_hours.new_full((), float("inf")))
    gap_hours, _ = masked_dt.min(dim=2)                               # [B, T]

    has_nb = torch.isfinite(gap_hours)
    valid = pad_mask & mlm_mask & has_nb
    gap_hours = torch.where(has_nb, gap_hours, torch.zeros_like(gap_hours))
    target = (gap_hours / max_hours).clamp(0.0, 5.0)
    return target, valid


def build_patient_labels(model, batch, training_settings, device):
    """
    Purpose: Build patient-level risk labels and per-outcome time targets for
             the Phase-3 fine-tune loss.
    Method:  For each outcome k, label[b, k] = 1 iff the outcome appears
             anywhere in the GT trajectory of patient b. gt_time[b, k] is the
             first occurrence in hours from sequence start, clamped to the
             training horizon. When the batch carries the un-truncated
             ``full_position_ids`` / ``full_abs_ts`` (Phase-3 input-truncated
             view), labels are derived from those so the prediction horizon
             matches evaluation.py; otherwise they come from the same
             ``position_ids`` / ``abs_ts`` the model consumed (Phase 1/2,
             eval).

    Args:
        model              : InterveneEncoder (uses ``outcome_names`` and the
                             tokenizer for the outcome → token-id mapping).
        batch              : dict with ``position_ids`` [B, T] and ``abs_ts``
                             [B, T] (normalised by 336 h). May also carry
                             ``full_position_ids`` / ``full_abs_ts`` (Phase 3).
        training_settings  : config — uses ``outcome_horizon_hours_p3`` for
                             the label clip (default 336 h).
        device             : torch device.

    Returns:
        labels   (FloatTensor): [B, K] {0., 1.} multi-label outcome flags.
        gt_time  (FloatTensor): [B, K] hours from seq start to first
                                occurrence (filled with 0.0 where label==0).
        present  (BoolTensor):  [B, K] True iff outcome present.
    """
    # ── Kineret addition: cohort-supplied supervision ────────────────────
    # When the dataset carries `patient_labels` (built by kineret.cohort), use
    # them verbatim instead of re-reading outcomes off the token stream. This is
    # what guarantees INTERVenE-Enc, STraTS and LogReg are fit against identical
    # targets: the token path would silently diverge whenever the Mediator
    # emitted an outcome as an interval (`X_EVENT_START` / `_END`) rather than
    # an instantaneous `X_EVENT`, or attached a Value other than "True".
    if "patient_labels" in batch:
        # Read the layout from the tokenizer first. `finetune_transformer`
        # rebuilds the model object when it resumes from `ckpt_last`, which
        # drops any attribute set on the original instance -- but the tokenizer
        # is passed through the embedder and survives, so that is where the
        # column order has to live.
        tok_columns = getattr(model.embedder.tokenizer, "injected_label_columns", None)
        columns = getattr(model, "injected_label_columns", None) or tok_columns
        if columns is None:
            raise RuntimeError(
                "[build_patient_labels] Batch carries `patient_labels` but no "
                "`injected_label_columns` was found on the model or its "
                "tokenizer. Set `tokenizer.injected_label_columns = "
                "dataset.label_columns` before training so the label matrix can "
                "be permuted into the model's own outcome order."
            )
        perm, missing = [], []
        for name in model.outcome_names:
            if name in columns:
                perm.append(columns.index(name))
            else:
                perm.append(-1)
                missing.append(name)
        if missing and not getattr(model, "_warned_missing_labels", False):
            print(f"[build_patient_labels] No cohort labels for {missing} -- "
                  f"those head slots are trained as all-negative.")
            model._warned_missing_labels = True

        raw_labels = batch["patient_labels"].to(device)             # [B, L]
        raw_times  = batch["patient_gt_time"].to(device)            # [B, L]
        B = raw_labels.shape[0]
        idx = torch.tensor(perm, dtype=torch.long, device=device)
        gather = idx.clamp(min=0)
        labels  = raw_labels[:, gather]
        gt_time = raw_times[:, gather]
        keep = (idx >= 0).view(1, -1).to(labels.dtype)
        labels, gt_time = labels * keep, gt_time * keep

        horizon = float(training_settings.get("outcome_horizon_hours_p3", 336.0))
        present = labels > 0.5
        gt_time = torch.where(present, gt_time.clamp(0.0, horizon),
                              torch.zeros_like(gt_time))
        return labels, gt_time, present

    tok = model.embedder.tokenizer
    outcome_ids = torch.tensor(
        [tok.token2id[n] for n in model.outcome_names], dtype=torch.long, device=device,
    )                                                              # [K]
    K = outcome_ids.numel()

    # Prefer the un-truncated trajectory (Phase-3 input-truncated batches
    # carry it explicitly). Falls back to the input sequence otherwise so
    # Phase 1/2 and eval-time call sites are unchanged.
    if "full_position_ids" in batch and "full_abs_ts" in batch:
        pos_ids = batch["full_position_ids"]                       # [B, T_full]
        abs_ts  = batch["full_abs_ts"]                             # [B, T_full] normalised
    else:
        pos_ids = batch["position_ids"]                            # [B, T]
        abs_ts  = batch["abs_ts"]                                  # [B, T] normalised
    pad_mask = pos_ids != tok.pad_token_id                          # [B, T]

    # match[b, t, k] = 1 iff position (b, t) is the k-th outcome token, non-pad.
    match = (pos_ids.unsqueeze(-1) == outcome_ids.view(1, 1, K)) & pad_mask.unsqueeze(-1)

    # Restrict positives to events occurring AFTER the input observation
    # window (default 48 h). Events inside the seed are observed tokens the
    # model receives directly — treating them as positive labels trivially
    # leaks the answer and inflates risk AUPRC, especially for HYPER /
    # SEVERE_HYPER which often fire within 0–48 h. ss-STraTS already
    # use this post-window-only convention in their preprocess, so this
    # restores a fair head-to-head label definition.
    t_hours = abs_ts.unsqueeze(-1) * 336.0                          # [B, T, 1]
    min_event_t = float(training_settings.get("outcome_min_event_hours_p3", 48.0))
    match = match & (t_hours > min_event_t)                         # [B, T, K]
    # Kineret addition: the label window is closed at the top --
    # (K*24, HORIZON_END_DAYS*24] -- so an outcome that only fires past the
    # horizon is a negative, exactly as it is for STraTS and LogReg.
    max_event_t = training_settings.get("outcome_max_event_hours_p3")
    if max_event_t is not None:
        match = match & (t_hours <= float(max_event_t))

    present = match.any(dim=1)                                     # [B, K]
    labels  = present.float()

    # First occurrence time per (b, k): mask non-matches with +inf, then min.
    huge = torch.full_like(t_hours, float("inf"))
    t_masked = torch.where(match, t_hours, huge)
    first_t, _ = t_masked.min(dim=1)                                # [B, K]
    horizon = float(training_settings.get("outcome_horizon_hours_p3", 336.0))
    gt_time = torch.where(present, first_t.clamp(0.0, horizon), torch.zeros_like(first_t))

    return labels, gt_time, present


def save_checkpoint(payload, path) -> None:
    """
    Purpose: Write a checkpoint so an interrupted save cannot destroy the old one.
    Method:  Serialise to a sibling `.tmp` file, flush it to disk, then rename
             over the target -- `os.replace` is atomic on POSIX and on Windows.

             `torch.save` writes in place, so a process killed midway through
             (an OOM, a hard reboot of a wedged VM) leaves a truncated file
             where the checkpoint used to be, and the next resume raises instead
             of continuing. Every phase here rewrites `ckpt_last.pt` once per
             epoch, so over a multi-day run that window is hit often enough to
             matter.

    Args:
        payload (dict):     What `torch.save` would have written.
        path    (str|Path): Destination.
    """
    path = str(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Leave no partial file behind, and let the failure surface: the
        # previous checkpoint is still on disk and still loadable.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def autocast_dtype(device):
    """
    Purpose: Decide whether mixed precision is a speed-up on THIS card.
    Method:  bfloat16 needs Ampere (compute capability >= 8.0). Turing and older
             can allocate a bf16 tensor but have no hardware path for it, and
             `torch.cuda.is_bf16_supported()` returns True anyway -- it falls
             back to "can I make one of these", which succeeds. Gating on that
             turned autocast on for a Tesla T4 and made every op SLOWER than
             plain fp32: measured on the study's own card, a 4096x4096 matmul
             took 34.3 ms in fp32 and 62.4 ms under bf16 autocast.

             So gate on the capability, not on the allocator. Pre-Ampere runs
             fp32, which is the fastest thing available there without adding a
             GradScaler (fp16 would be faster still, but needs loss scaling to
             train stably, and that is a change to training dynamics rather than
             to throughput).

    Args:
        device (torch.device): The device training will run on.

    Returns:
        tuple[bool, torch.dtype]: (enabled, dtype) for `torch.autocast`.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, torch.float32
    major, _ = torch.cuda.get_device_capability(device if device.index is not None else None)
    if major >= 8:
        return True, torch.bfloat16
    return False, torch.float32


class StepTimer:
    """
    Purpose: Split an epoch's wall clock into "waiting for the dataloader" and
             "computing", and report peak memory alongside it.
    Method:  Wrap the loader in a generator. Time before a batch arrives is
             data; time between yielding it and asking for the next is compute.
             CUDA work is synchronised before the compute clock stops --
             without that, kernels queue asynchronously and their cost is
             misattributed to the NEXT data wait, inverting the answer.

             This exists because the alternative -- attaching `py-spy` to the
             live process -- needs a shell, and a notebook kernel busy training
             cannot give you one.

    Read it as:
        data >> compute  -> dataloader-bound (workers, collate, RAM pressure)
        compute >> data  -> GPU-bound (precision, model size, sequence length)
    """

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.reset()

    def reset(self):
        """Purpose: Clear the accumulators at the start of an epoch."""
        self.data_s = 0.0
        self.compute_s = 0.0
        self.batches = 0

    def wrap(self, loader):
        """
        Purpose: Iterate `loader`, attributing time to data vs compute.

        Args:
            loader (iterable): Any batch iterator (tqdm-wrapped is fine).

        Yields:
            Whatever the loader yields.
        """
        import time
        t_wait = time.perf_counter()
        for batch in loader:
            self.data_s += time.perf_counter() - t_wait
            t_compute = time.perf_counter()
            yield batch
            if self.sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.compute_s += time.perf_counter() - t_compute
            self.batches += 1
            t_wait = time.perf_counter()

    def report(self) -> str:
        """
        Purpose: One line naming the bottleneck.

        Returns:
            str: Timing split, throughput and peak memory.
        """
        total = self.data_s + self.compute_s
        if total <= 0 or self.batches == 0:
            return "[timing] no batches"
        share = self.data_s / total
        verdict = ("DATALOADER-BOUND" if share > 0.6
                   else "compute-bound" if share < 0.3 else "mixed")
        rss = ""
        try:                                    # Linux; absent on Windows
            import resource
            peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
            rss = f"  peak RSS={peak_gb:.1f} GB"
        except Exception:                       # noqa: BLE001
            pass
        gpu = ""
        if torch.cuda.is_available():
            gpu = (f"  gpu={torch.cuda.max_memory_allocated()/1024**3:.1f}"
                   f"/{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        return (f"[timing] {self.batches} batches in {total/60:.1f} min "
                f"({total/self.batches:.2f} s/batch)  "
                f"data={share:.0%} compute={1-share:.0%}  -> {verdict}{rss}{gpu}")
