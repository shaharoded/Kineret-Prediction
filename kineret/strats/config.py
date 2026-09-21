"""
ss-STraTS model configuration.

Ported from `med-transformers-baseline/src/config.py`, reduced to what this
study uses. The MIMIC-specific event-derivation rules are gone -- labels come
from `kineret.cohort`, so every model is supervised on byte-identical targets --
and so is the self-supervised forecasting stage: this cohort is small and is the
only data available, so the study compares against **ss-STraTS**, the
supervised-only variant from the original paper.

What remains is this model's own hyperparameters. Shared study decisions
(windows, targets, date range, QA aggregation) are NOT here -- they live in
`kineret/config/data_config.py` and are set from the notebook.
"""

import os

from kineret.config import paths

PROCESSED_DIR = paths.PROCESSED_DIR


def dataset_name(use_qa: bool) -> str:
    """
    Purpose: Canonical pickle stem for one QA arm.
    Method:  The arm is in the name so a stale pickle from the other arm cannot
             be silently reused. There is no K in the name: training is
             augmented across all training windows and evaluation happens at a
             single window, so one pickle covers the whole arm.

    Args:
        use_qa (bool): QA ablation arm.

    Returns:
        str: "kineret_qa" or "kineret_noqa".
    """
    return f"kineret_{'qa' if use_qa else 'noqa'}"


def dataset_pkl(use_qa: bool) -> str:
    """Purpose: Path to the input pickle for one QA arm."""
    return os.path.join(PROCESSED_DIR, dataset_name(use_qa) + ".pkl")


# ---------------------------------------------------------------------------
# Input-vocabulary filter. 0.0 keeps every observed concept, so ss-STraTS sees
# the same token surface the Mediator-driven models do. Set > 0 to drop concepts
# below that sample-level support.
# ---------------------------------------------------------------------------
CONCEPT_SUPPORT_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Architecture + optimisation. MIMIC-III defaults from the official
# `run_main.sh` -- the closer comparator to a cohort of this size.
# ---------------------------------------------------------------------------
STRATS_SETTINGS = {
    "hid_dim": 64,
    "num_layers": 2,
    "num_heads": 16,
    "dropout": 0.2,
    "attention_dropout": 0.2,
    "max_obs": 880,
    "lr": 5e-4,
    "max_epochs": 50,
    "patience": 10,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "los_loss_weight": 1.0,
    # Weight of the per-outcome onset-time MSE against the risk BCE. Matches
    # INTERVenE's `phase3_time_lambda` so the two models trade risk against
    # timing identically.
    "time_loss_weight": 0.1,
}

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
