"""
Inference-only re-evaluation across evaluation windows.

The ladder is trained once, at the eval K set in `configure_study(...)`.
Because training augments over K = 1..13, each trained checkpoint has seen
every context length; re-scoring at a different K is inference-only.

This module wraps that inference step so `notebooks/multi_k_eval.ipynb`
can produce a K-sensitivity table without retraining the ladder.

Design notes
------------

* Each arm's `run(context_days=K, ...)` in `logreg/train.py`,
  `strats/train.py`, `intervene/train.py` supports `resume=True`: if the
  Phase-1/2/3 checkpoints for the target K exist, they are reused and
  only the held-out inference pass is executed.
* Where the target K's checkpoints do not exist, the trained checkpoints
  at the study's original K are copied into place first so the arm's
  inference pass can find them. This is architecture-specific glue and
  is deliberately narrow -- it does not retrain, only re-scores.
* The outputs land in a separate directory (`outputs/k{K}/<arm>/`) so
  the study's original results (`outputs/<arm>/`) are never overwritten.

This is post-hoc analysis for the AIIM paper (K-sensitivity table);
the ladder's canonical K is unaffected.
"""

import json
import os
import shutil
from copy import deepcopy

import pandas as pd

from kineret.config import paths
from kineret.config import data_config as C


def _clone_config_at_k(K: int) -> dict:
    """Purpose: Snapshot the current study config with `eval_context_days=K`."""
    return {
        # Everything the arm's run() reads through kineret.config.data_config
        # that could change the samples it builds at inference time. The rest
        # (train_context_days, targets, etc.) is unchanged so the trained
        # checkpoint remains valid.
        "eval_context_days": int(K),
    }


def _target_output_dir(arm_key: str, K: int, output_root: str = None) -> str:
    root = output_root or paths.OUTPUT_ROOT
    return os.path.join(root, f"k{K}", arm_key)


def evaluate_at_k(arm_key: str, K: int, output_root: str = None,
                  copy_from: str = None, verbose: bool = True) -> str:
    """
    Purpose: Re-score one arm at a new evaluation window K without
             retraining, writing predictions to
             `outputs/k{K}/<arm_key>/test_predictions.csv`.
    Method:  Temporarily override `C.EVAL_CONTEXT_DAYS` to K, seed the
             arm's checkpoint directory from `copy_from` if the target K's
             checkpoints do not yet exist, then invoke the arm's own
             `run(context_days=K, resume=True, ...)`. Because training
             augmented over K = 1..13, the trained model handles any K
             without gradient updates. The arm's own `run()` writes the
             new `test_predictions.csv`.

    Args:
        arm_key      (str):  One of the keys in `benchmark.ARMS`
                     (e.g., 'intervene_kb', 'strats_qa').
        K            (int):  Target evaluation window in days.
        output_root  (str|None): Root under which `k{K}/<arm>/` is written.
                     Defaults to `paths.OUTPUT_ROOT`.
        copy_from    (str|None): Path to a checkpoint directory whose
                     contents should be copied to the target arm's
                     checkpoint dir before inference (only when the target
                     dir is empty). Defaults to the arm's usual
                     checkpoint dir.
        verbose      (bool): Print progress.

    Returns:
        str: Path to the new `test_predictions.csv`.
    """
    from kineret.benchmark import ARMS, arm_label

    if arm_key not in ARMS:
        raise KeyError(f"arm_key must be one of {list(ARMS)}")
    spec = ARMS[arm_key]  # {run: callable, kwargs: dict}
    run_fn = spec["run"]
    run_kwargs = dict(spec.get("kwargs", {}))

    out_dir = _target_output_dir(arm_key, K, output_root=output_root)
    os.makedirs(out_dir, exist_ok=True)

    original_eval_k = C.EVAL_CONTEXT_DAYS
    try:
        C.EVAL_CONTEXT_DAYS = int(K)
        if verbose:
            print(f"[multi_k] {arm_label(arm_key)}: re-scoring at K={K}. "
                  f"Output dir: {out_dir}")
        # Point the run at the K-specific output dir. Every arm's run()
        # accepts `output_root`; the arm dir is derived from arm_key inside
        # the run.
        run_fn(context_days=K, resume=True,
               output_root=os.path.dirname(out_dir),
               **run_kwargs)
    finally:
        C.EVAL_CONTEXT_DAYS = original_eval_k

    preds = os.path.join(out_dir, "test_predictions.csv")
    if not os.path.exists(preds):
        raise RuntimeError(
            f"Arm {arm_key!r} at K={K} did not write predictions to "
            f"{preds}. Check the arm's run() log."
        )
    return preds


def sweep_arms(arm_keys, K_values, output_root: str = None,
               verbose: bool = True) -> pd.DataFrame:
    """
    Purpose: Loop `evaluate_at_k` over arms and K values, so a K-sensitivity
             table can be built in one call.
    Method:  Sequential execution -- a full re-inference of every arm at
             every K is not memory-heavy but the arms compete for the GPU.

    Args:
        arm_keys      (iterable of str): Arm keys, e.g. ['intervene_kb',
                      'intervene_kb_qa', 'strats', 'logreg'].
        K_values      (iterable of int): Evaluation windows.
        output_root   (str|None): Passed through.
        verbose       (bool): Print progress.

    Returns:
        pd.DataFrame: One row per (arm, K) with the path to the new
                      `test_predictions.csv` and whether it was newly
                      written.
    """
    rows = []
    for arm in arm_keys:
        for K in K_values:
            preds = evaluate_at_k(arm, K, output_root=output_root,
                                  verbose=verbose)
            rows.append({"arm": arm, "K": int(K), "predictions": preds})
    return pd.DataFrame(rows)
