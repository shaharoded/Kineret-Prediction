import copy
import os

from kineret.config import paths
from kineret.config import data_config as _shared

PROJECT_ROOT = paths.PROJECT_ROOT

# Checkpoint paths. `configure_checkpoints()` re-roots these per sweep cell so
# a K=4 run never resumes from a K=7 checkpoint.
CHECKPOINT_PATH = paths.CHECKPOINT_DIR
PHASE1_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'ckpt_best.pt')
PHASE2_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'ckpt_best.pt')
PHASE3_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase3', 'ckpt_best.pt')


def configure_checkpoints(context_days: int, use_qa: bool, root: str = None):
    """
    Purpose: Give each (K, QA-arm) cell its own checkpoint tree.
    Method:  Re-root CHECKPOINT_PATH under `<root>/k<K>_<arm>/` and rebuild the
             three phase paths, then push the new values into every
             already-imported `kineret.intervene.*` module -- the package pulls
             this config in with `import *`, so rebinding here alone would not
             reach the modules that already copied the old values.

    Args:
        context_days (int):      K -- context window in days.
        use_qa       (bool):     QA ablation arm.
        root         (str|None): Checkpoint root. Defaults to `checkpoints/`.
                                 The smoke test passes its own so a 1-epoch
                                 toy-width run can never be resumed by, or
                                 poison, a real one.

    Returns:
        dict: The rebound checkpoint paths.
    """
    import sys

    global CHECKPOINT_PATH, PHASE1_CHECKPOINT, PHASE2_CHECKPOINT, PHASE3_CHECKPOINT
    CHECKPOINT_PATH = os.path.join(root or paths.CHECKPOINT_DIR,
                                   f"k{context_days}_{'qa' if use_qa else 'noqa'}")
    PHASE1_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'ckpt_best.pt')
    PHASE2_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'ckpt_best.pt')
    PHASE3_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase3', 'ckpt_best.pt')
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    bound = {
        'CHECKPOINT_PATH': CHECKPOINT_PATH,
        'PHASE1_CHECKPOINT': PHASE1_CHECKPOINT,
        'PHASE2_CHECKPOINT': PHASE2_CHECKPOINT,
        'PHASE3_CHECKPOINT': PHASE3_CHECKPOINT,
    }
    for name, module in list(sys.modules.items()):
        if not name.startswith('kineret.intervene') or module is None:
            continue
        for key, value in bound.items():
            if hasattr(module, key):
                setattr(module, key, value)
    return bound


def configure_windows(context_days: int):
    """
    Purpose: Align Phase-3 supervision with the shared label contract.
    Method:  Mutate TRAINING_SETTINGS in place (it is a module-level dict, so
             every `import *` consumer already holds the same object) to set the
             label window to (K*24, HORIZON_END_DAYS*24] and the soft-label
             horizon to match.

    Args:
        context_days (int): K -- context window in days.

    Returns:
        dict: The keys that were set.
    """
    lo = float(context_days) * 24.0
    hi = float(_shared.HORIZON_END_DAYS) * 24.0
    TRAINING_SETTINGS['outcome_min_event_hours_p3'] = lo
    TRAINING_SETTINGS['outcome_max_event_hours_p3'] = hi
    TRAINING_SETTINGS['outcome_horizon_hours_p3'] = hi
    TRAINING_SETTINGS['phase3_input_days'] = context_days
    return {k: TRAINING_SETTINGS[k] for k in
            ('outcome_min_event_hours_p3', 'outcome_max_event_hours_p3',
             'outcome_horizon_hours_p3', 'phase3_input_days')}

# Global RNG seed — applied via utils.set_seed() in every model constructor and
# training-phase entry point so runs are reproducible.
SEED = _shared.SEED

MODEL_CONFIG = {
      "time2vec_dim": 32,
      "embed_dim": 128,   # M-128 starting point (head_dim=64, n_head=2); grow during the size sweep.
      "n_head": 2,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True,
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 100,
    "phase2_n_epochs": 100,
    "phase3_n_epochs": 100,
    "sample": None,

    # Phase-2 optimizer LR warmup (OneCycleLR pct_start).
    # This controls optimizer step size ramp-up, not auxiliary-loss lambda warmup.
    "lr_warmup_epochs": 5,
    "early-stop-patience": 10,
    "early-stop-min-delta-rel": 1e-3,  # relative improvement threshold (0.1%)

    "phase1_learning_rate":       3e-4,
    "phase2_learning_rate":       3e-4,
    "phase3_learning_rate":       1e-4,
    "phase3_backbone_lr_factor":  0.1,   # was 0.01 (near-frozen); unfreeze so P3 backbone adapts  # backbone LR = phase3_lr * factor (1e-6); 0.0 = fully frozen
    "phase3_weight_decay":        1e-3,  # weight decay for outcome_head in P3 (matches backbone)
    "weight_decay":               1e-3,

    # DataLoader worker processes. 0 = load in the training process, which is
    # the only setting that keeps ONE copy of the dataset in memory; each worker
    # forks and copy-on-write breaks on the per-sample DataFrames. Raise it only
    # if the [timing] line says DATALOADER-BOUND *and* peak RSS has headroom.
    "dataloader_workers": 0,

    "batch_size": 16, # Number of patients processed concurrently (effective batch=64 via grad accumulation)
    "grad_accumulation_steps": 4, # Accumulate gradients over N steps before optimizer.step(), memory-friendly way to get effective batch size > GPU batch size.
    "phase1_bce_window_hours": 3.0,
    # Soft-kernel horizon for the Phase-2 LM-head BCE. The kernel decay constant
    # tau is learnable per token class (model.log_tau_lm); this value is both the
    # init for terminal tokens and the hard outer horizon beyond which the kernel
    # contribution is zero.
    "phase2_terminal_bce_window_hours": 168.0,

    # Phase-1 auxiliary scheduler.
    # Main loss = per-window outcome BCE. Single stage: the `dt` (Δt MSE) aux
    # activates after `main_only_epochs` epochs of main-loss-only training. The
    # lambda max is calibrated ONCE from train losses at the first active epoch
    # (λ = aux_fraction_cap × tr_main / tr_aux) and then kept fixed. The
    # weighted contribution is capped to `aux_fraction_caps[name]` of tr_main.
    "phase1_scheduler": {
        "main_only_epochs": 3,     # epochs of BCE-only training before dt activates
        "aux_fraction_caps": {
            "dt":  0.40,           # Δt MSE capped at 40% of BCE at calibration
        },
        "order": [["dt"]],         # single stage with one aux
        "ramp_epochs": {
            "dt":  0,              # no ramp; jump straight to λ_max once unlocked
        },
        # Uncap dt (remove the global hard clamp here too). Numerically a no-op —
        # dt's natural λ≈0.033 (raw dt ≫ main BCE) is far below 10 — but keeps the
        # fraction-cap rule the sole governor, per the no-hard-clamp preference.
        "max_lambda": {"dt": 400.0},
    },

    # Phase-2 auxiliary scheduler.
    # Main loss = MLM cross-entropy. Single stage: `t_pos` (time-since-admission
    # MSE at every non-pad position) and `t_local` (time-to-neighbour MSE at
    # masked positions only) activate together after `main_only_epochs` of
    # MLM-only training. Lambda max is calibrated ONCE from training losses at
    # the first active epoch (λ = aux_fraction_cap × tr_main / tr_aux) and then
    # kept fixed. `main_only_epochs` doubles as LRScheduleController's
    # OneCycleLR pct_start anchor.
    "phase2_scheduler": {
        "main_only_epochs": 4,     # epochs of MLM-only training before t_pos/t_local activate
        "aux_fraction_caps": {
            "t_pos":   0.40,       # time-since-admission MSE capped at 40% of MLM CE
            "t_local": 0.15,       # trimmed 0.30->0.15: reduce early MLM competition (t_pos uncapped adds pressure)
        },
        "order": [["t_pos", "t_local"]],   # single stage, both auxes unlock together
        "ramp_epochs": {
            "t_pos":   0,          # no ramp; jump to λ_max at unlock
            "t_local": 0,
        },
        # Per-aux λ_max ceiling (overrides the global hard clamp of 10). The
        # global clamp is a safety against a tiny-magnitude aux getting a runaway
        # λ from the fraction rule (λ = fraction_cap × MLM/aux). Both time auxes
        # are now uncapped so the fraction-cap rule alone governs their weight:
        #   t_local → λ≈61 (0.30 share);  t_pos → λ≈41 (0.40 share).
        # (Phase-1 `dt` keeps the default-10 safety; it's tiny and never binds.)
        "max_lambda": {"t_local": 400.0, "t_pos": 400.0},
    },

    # Outcome head — time-decayed soft labels.
    # For each position t the target for outcome k is:
    # sum_s { exp(-dt(t,s) / tau_k) * 1[token_s == outcome_k] }.clamp(0, 1)
    # tau_k is a per-outcome learnable parameter (model.outcome_log_tau), initialised
    # at log(12 / 336). outcome_horizon_hours hard-zeros any contribution beyond that
    # horizon (kept in sync with the eval window family).
    "outcome_horizon_hours": 48.0,

    # P4 — patient-level attention pool head (Phase 3 only).
    # Per-outcome learnable query embeddings cross-attend over the backbone's
    # final hidden states to produce one pooled feature per (patient, outcome).
    # A scalar projection turns each pooled feature into a patient-level
    # logit; BCE against patient_label[b, k] = "outcome k appears anywhere in
    # the non-pad GT trajectory". λ_pool calibrated once at the end of
    # Phase-3 epoch 1, capped at this fraction of raw outcome BCE — same
    # regime as ranking. Pool head trains at Phase-3 head LR; its gradient
    # flows through the hidden-state stash into the backbone at
    # backbone_lr_factor=0.01, protecting the outcome head from patient-level
    # coarseness.
    "phase3_pool_fraction_cap": 0.05,   # I2 P4-tight: lowered 0.20 -> 0.05

    # --- BERT-style Phase-2 settings ---
    # MLM ratio applied per batch (BERT-default 15%; ramped from 0 over main_only_epochs).
    "phase2_mlm_ratio": 0.15,
    # Atomic-interval mask replacement: three generic tokens
    # ([MASK], [MASK_INTERVAL_START/END]); hierarchical/HEART-family masking
    # was tested (i1-hier) and DISCARDED — removed from the codebase.
    # Phase-2 aux-loss fraction caps live inside `phase2_scheduler`
    # ("aux_fraction_caps": {"t_pos": …, "t_local": …}) — see above.

    # --- Phase-3 settings ---
    # phase3_time_lambda — weight of the per-outcome z-MSE time loss vs the
    # multi-label BCE risk loss. 0.5 keeps the time head on near-equal
    # footing with risk (matches ss-STraTS's LoS-loss weight).
    "phase3_time_lambda": 0.1,
    "phase3_head_hidden": 256,
    # phase3_cbm_p — Curriculum-by-Masking input-token replacement during
    # Phase-3 training. Masks `p` fraction of non-special positions with
    # [MASK] (BERT-style), targets/labels computed from the un-noised
    # batch. Forces the model to use multi-source signal, helps rare
    # outcomes. 0 disables.
    "phase3_cbm_p": 0.0,
    # phase3_pool_dropout — dropout inside the Phase-3 attention pool +
    # shared MLP. ``None`` inherits the backbone's MODEL_CONFIG["dropout"].
    # 0.20 matches STraTS's attention_dropout.
    "phase3_pool_dropout": 0.20,

    # --- Kineret label window ---
    # Phase-3 positives are outcomes in (min, max] hours from admission.
    # `configure_windows(K)` overwrites both from the shared config; the values
    # here are only the fallback for an ad-hoc run.
    "outcome_min_event_hours_p3": _shared.EVAL_CONTEXT_DAYS * 24.0,
    "outcome_max_event_hours_p3": _shared.HORIZON_END_DAYS * 24.0,
    "outcome_horizon_hours_p3":   _shared.HORIZON_END_DAYS * 24.0,
    "phase3_input_days":          _shared.EVAL_CONTEXT_DAYS,
}


# ---------------------------------------------------------------------------
# Per-run isolation.
#
# `intervene.train.run` applies its overrides with `MODEL_CONFIG.update(...)` and
# `TRAINING_SETTINGS.update(...)`, which mutate THESE dicts in place and never
# put them back. That is fine for one run and wrong for a sequence of them:
#
#   smoke_test() runs every arm with SMOKE_OVERRIDES -- embed_dim 32, n_layer 1,
#   one epoch per phase -- and those values then persist in this module for the
#   rest of the kernel. Running the wiring check and then the real ladder in one
#   session would train the whole study as a 32-dim toy for one epoch, complete
#   every arm, pass the label-agreement check, and write plausible-looking
#   predictions. Nothing would announce it.
#
# So each run resets to the values this file declares before applying its own
# overrides. Editing a default here still reaches every arm; an override reaches
# only the run that asked for it.
_DEFAULT_MODEL_CONFIG = copy.deepcopy(MODEL_CONFIG)
_DEFAULT_TRAINING_SETTINGS = copy.deepcopy(TRAINING_SETTINGS)


def reset_to_defaults():
    """
    Purpose: Undo any previous run's overrides.
    Method:  Restore both dicts in place -- in place, because the package pulls
             them in with `import *` and rebinding the names here would leave
             every importing module holding the old objects.

    Returns:
        tuple[dict, dict]: (MODEL_CONFIG, TRAINING_SETTINGS), restored.
    """
    MODEL_CONFIG.clear()
    MODEL_CONFIG.update(copy.deepcopy(_DEFAULT_MODEL_CONFIG))
    TRAINING_SETTINGS.clear()
    TRAINING_SETTINGS.update(copy.deepcopy(_DEFAULT_TRAINING_SETTINGS))
    return MODEL_CONFIG, TRAINING_SETTINGS
