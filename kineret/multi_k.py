"""Frozen-model K sensitivity, managed from notebooks/benchmark.ipynb."""
from pathlib import Path
from kineret.experiments.workflow import evaluate, analyze, require_predictions


def evaluate_at_k(arm_key, K, cohort, output_root, source_outputs=None,
                  checkpoint_root=None, **kwargs):
    """Re-score at K with unchanged heads/scalers/checkpoints; never train."""
    destination = Path(output_root) / f"k{int(K)}"
    predictions, summary, status = evaluate(
        cohort, destination, arms=[arm_key], k=K,
        source_outputs=source_outputs, checkpoint_root=checkpoint_root, **kwargs)
    if arm_key not in predictions:
        raise RuntimeError(status.to_string(index=False))
    return str(destination / arm_key / "test_predictions.csv")


def sweep_arms(arm_keys, K_values, cohort, output_root, **kwargs):
    import pandas as pd
    tables = []
    for k in K_values:
        directory = Path(output_root)/f"k{k}"
        predictions, table, status = evaluate(cohort, directory,
                                   arms=arm_keys, k=k, include_validation=False, **kwargs)
        require_predictions(predictions, arm_keys)
        analyze(predictions, directory, n_resamples=kwargs.get("n_resamples", 2000), do_calibration=False)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
