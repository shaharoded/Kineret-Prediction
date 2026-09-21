"""
One scorer for all models.

Every training script writes the same artefact:

    outputs/<run_name>/test_predictions.csv
        PatientId,
        label_<outcome>...,     did it occur in (K*24, N*24]
        prob_<outcome>...,      predicted risk
        time_true_<outcome>..., observed onset hours (NaN when it never occurred)
        time_pred_<outcome>..., predicted onset hours
        los_true_hours, los_pred_hours
    outputs/<run_name>/run_meta.json
        model, context_days (K), use_qa, seed, outcome_names, n_test, ...

so the comparison across {INTERVenE-Enc, ss-STraTS, LogReg} x
{K = 2..7} x {QA on, QA off} reduces to reading those files. Metrics and the
patient-level bootstrap are ported from the med-transformers baseline's
`src/evaluator.py` so numbers stay directly comparable with the MIMIC-IV runs.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, roc_auc_score,
)

METRIC_NAMES = ("auroc", "auprc", "best_f1", "f1_0_5", "minrp")

# Below this many positive patients a bootstrap interval stops describing model
# uncertainty and starts describing which of a handful of patients got drawn.
# At n=1 it collapses to a point, which reads as precision and is the opposite
# of it. Outcomes under the floor are flagged `ci_reliable=False` rather than
# hidden -- the estimate is still the best available, it just must not be
# quoted on its own.
MIN_POSITIVES_FOR_CI = 5


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------

def _drop_nonfinite(y_true, y_pred):
    """Purpose: Keep only rows with a finite prediction so sklearn cannot choke
    on a freshly-initialised model that emits NaN for edge-case patients."""
    mask = np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _safe_auc_pair(y_true, y_pred):
    """Purpose: AUROC/AUPRC, returning (nan, nan) when a resample is single-class."""
    y_true, y_pred = _drop_nonfinite(y_true, y_pred)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)


def _f1_triplet(y_true, y_pred):
    """
    Purpose: Threshold-family metrics for one outcome.
    Method:  Sweep the PR curve for the best achievable F1, evaluate F1 at the
             fixed 0.5 operating point, and take min(precision, recall) at its
             maximum -- the balanced-operating-point summary the STraTS
             benchmark reports as `minRP`.

    Args:
        y_true (np.ndarray): Binary labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        tuple[float, float, float]: (best_f1, f1_at_0.5, minrp).
    """
    y_true, y_pred = _drop_nonfinite(y_true, y_pred)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    f1_curve = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    return (float(np.nanmax(f1_curve)),
            float(f1_score(y_true, y_pred >= 0.5)),
            float(np.minimum(precision, recall).max()))


def per_outcome_metrics(labels: np.ndarray, probs: np.ndarray,
                        outcome_names: list) -> pd.DataFrame:
    """
    Purpose: Point-estimate table, one row per outcome.
    Method:  Patient-level pairs -- each (patient, outcome) contributes exactly
             one (probability, did-it-occur-in-the-label-window) pair.

    Args:
        labels        (np.ndarray): [N, K] binary ground truth.
        probs         (np.ndarray): [N, K] predicted probabilities.
        outcome_names (list[str]):  Column order for both matrices.

    Returns:
        pd.DataFrame: outcome, n_pos, prevalence, auroc, auprc, best_f1,
                      f1_0_5, minrp.
    """
    rows = []
    for i, name in enumerate(outcome_names):
        y, p = labels[:, i], probs[:, i]
        auroc, auprc = _safe_auc_pair(y, p)
        best_f1, f1_05, minrp = _f1_triplet(y, p)
        rows.append({
            "outcome": name,
            "n_pos": int(np.nansum(y)),
            "prevalence": float(np.nanmean(y)),
            "auroc": auroc, "auprc": auprc,
            "best_f1": best_f1, "f1_0_5": f1_05, "minrp": minrp,
        })
    return pd.DataFrame(rows)


def aggregate(per_outcome: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Collapse the per-outcome table into the two headline rows.
    Method:  `macro` is an unweighted mean across outcomes; `weighted` weights
             each outcome by its positive support, so rare outcomes contribute
             proportionally rather than dominating the average through noise.

    Args:
        per_outcome (pd.DataFrame): Output of `per_outcome_metrics`.

    Returns:
        pd.DataFrame: Two rows (`macro`, `weighted`) x METRIC_NAMES.
    """
    # Only METRIC_NAMES are averaged here; timing is aggregated separately in
    # `score_run` because it is weighted by positive count, not by support.
    weights = per_outcome["n_pos"].to_numpy(dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights),
                                                                       1.0 / max(len(weights), 1))
    rows = {"macro": {}, "weighted": {}}
    for metric in METRIC_NAMES:
        vals = per_outcome[metric].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        rows["macro"][metric] = float(np.nanmean(vals)) if finite.any() else np.nan
        denom = weights[finite].sum()
        rows["weighted"][metric] = (float((vals[finite] * weights[finite]).sum() / denom)
                                    if denom > 0 else np.nan)
    return pd.DataFrame(rows).T.rename_axis("average").reset_index()


def time_to_event_mae(time_true: np.ndarray, time_pred: np.ndarray,
                      labels: np.ndarray, outcome_names: list) -> pd.DataFrame:
    """
    Purpose: Onset-timing error, per outcome, in hours.
    Method:  MAE over POSITIVE patients only -- "when will it happen" is
             meaningless for a complication that never happens, so negatives are
             masked out rather than scored against a placeholder. Rows whose
             ground truth or prediction is non-finite are dropped too.

    Args:
        time_true     (np.ndarray): [N, K] observed onset hours (NaN if never).
        time_pred     (np.ndarray): [N, K] predicted onset hours.
        labels        (np.ndarray): [N, K] binary occurrence.
        outcome_names (list[str]):  Column order.

    Returns:
        pd.DataFrame: outcome, time_mae_h, n_time (contributing patients).
    """
    rows = []
    for i, name in enumerate(outcome_names):
        mask = (labels[:, i] > 0) & np.isfinite(time_true[:, i]) & np.isfinite(time_pred[:, i])
        mae = (float(np.abs(time_pred[mask, i] - time_true[mask, i]).mean())
               if mask.any() else np.nan)
        rows.append({"outcome": name, "time_mae_h": mae, "n_time": int(mask.sum())})
    return pd.DataFrame(rows)


def length_of_stay_mae(los_true: np.ndarray, los_pred: np.ndarray) -> tuple:
    """
    Purpose: Length-of-stay regression error, in hours.
    Method:  MAE over patients with a finite ground-truth LoS -- i.e. those who
             were actually discharged. Patients who died or have no terminus
             carry NaN and are excluded rather than imputed.

    Args:
        los_true (np.ndarray): Ground-truth hours to RELEASE (NaN where absent).
        los_pred (np.ndarray): Predicted hours.

    Returns:
        tuple[float, int]: (MAE in hours, number of contributing patients).
    """
    mask = np.isfinite(los_true) & np.isfinite(los_pred)
    if not mask.any():
        return float("nan"), 0
    return float(np.abs(los_pred[mask] - los_true[mask]).mean()), int(mask.sum())


# ---------------------------------------------------------------------------
# Patient-level bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metrics(labels, probs, outcome_names, los_true=None, los_pred=None,
                      time_true=None, time_pred=None,
                      n_resamples: int = 2000, seed: int = 42) -> dict:
    """
    Purpose: 95 % CIs for every headline number from a single training run.
    Method:  Resample PATIENTS with replacement `n_resamples` times, recompute
             every metric on each resample, and report the mean plus the
             [2.5 %, 97.5 %] quantiles. Macro and support-weighted aggregates
             are formed inside each iteration (using that resample's own
             supports) so the CI reflects support noise too. Matches the
             protocol used by INTERVenE-Enc and the med-transformers baseline.

    Args:
        labels        (np.ndarray): [N, K] binary ground truth.
        probs         (np.ndarray): [N, K] predicted probabilities.
        outcome_names (list[str]):  Column order.
        los_true      (np.ndarray|None): Ground-truth LoS hours.
        los_pred      (np.ndarray|None): Predicted LoS hours.
        time_true     (np.ndarray|None): [N, K] observed onset hours.
        time_pred     (np.ndarray|None): [N, K] predicted onset hours.
        n_resamples   (int):        Bootstrap iterations; 0 skips the block.
        seed          (int):        RNG seed.

    Returns:
        dict: {'per_outcome': {metric: {mean/lo/hi arrays}},
               'overall': {metric: {macro/weighted: {mean/lo/hi}}},
               'los': {mean/lo/hi},
               'per_outcome_onset': {mean/lo/hi arrays},
               'onset': {macro/weighted: {mean/lo/hi}},
               'n_resamples': int}

    Note on the onset interval: it is computed inside the SAME patient
    resample as the risk metrics, so the numbers stay mutually consistent. But
    onset is conditional on occurrence -- only positive patients contribute --
    so for a rare outcome the interval reflects which of a handful of positives
    happened to be drawn as much as it reflects model uncertainty. Read it
    alongside `n_time` in the per-outcome table, and treat a wide interval on a
    single-digit support as the small sample it is.
    """
    if n_resamples <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n_patients, n_outcomes = labels.shape
    samples = {m: np.full((n_resamples, n_outcomes), np.nan) for m in METRIC_NAMES}
    n_pos_samples = np.zeros((n_resamples, n_outcomes), dtype=float)
    los_samples = np.full(n_resamples, np.nan)
    has_los = los_true is not None and los_pred is not None
    onset_samples = np.full((n_resamples, n_outcomes), np.nan)
    n_time_samples = np.zeros((n_resamples, n_outcomes), dtype=float)
    has_time = time_true is not None and time_pred is not None
    if has_time:
        time_true = np.asarray(time_true, dtype=float)
        time_pred = np.asarray(time_pred, dtype=float)

    for b in range(n_resamples):
        idx = rng.integers(0, n_patients, size=n_patients)
        for i in range(n_outcomes):
            y, p = labels[idx, i], probs[idx, i]
            samples["auroc"][b, i], samples["auprc"][b, i] = _safe_auc_pair(y, p)
            best_f1, f1_05, minrp = _f1_triplet(y, p)
            samples["best_f1"][b, i] = best_f1
            samples["f1_0_5"][b, i] = f1_05
            samples["minrp"][b, i] = minrp
            n_pos_samples[b, i] = np.nansum(y)
            if has_time:
                # Positives only: a complication that did not occur has no
                # onset, so it contributes nothing to the error.
                m = ((labels[idx, i] > 0)
                     & np.isfinite(time_true[idx, i])
                     & np.isfinite(time_pred[idx, i]))
                n_time_samples[b, i] = m.sum()
                if m.any():
                    onset_samples[b, i] = float(
                        np.abs(time_pred[idx, i][m] - time_true[idx, i][m]).mean())
        if has_los:
            lt, lp = los_true[idx], los_pred[idx]
            mask = np.isfinite(lt) & np.isfinite(lp)
            if mask.any():
                los_samples[b] = float(np.abs(lp[mask] - lt[mask]).mean())

    def _ci(arr, axis=None):
        # An outcome with no positives is all-NaN across every resample; numpy
        # warns on that rather than erroring, and NaN is the right answer, so
        # silence the noise instead of letting it flood the training logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {
                "mean": np.nanmean(arr, axis=axis),
                "lo": np.nanpercentile(arr, 2.5, axis=axis),
                "hi": np.nanpercentile(arr, 97.5, axis=axis),
            }

    per_outcome = {m: _ci(samples[m], axis=0) for m in METRIC_NAMES}

    # Support weights per resample; rows with zero total support fall back to a
    # uniform mean rather than being discarded.
    weights = n_pos_samples / np.maximum(n_pos_samples.sum(axis=1, keepdims=True), 1)
    weights = np.where(weights.sum(axis=1, keepdims=True) > 0, weights, 1.0 / n_outcomes)

    overall = {}
    for m in METRIC_NAMES:
        vals = samples[m]
        valid = ~np.isnan(vals)
        weighted = (np.nansum(vals * weights, axis=1)
                    / np.nansum(weights * valid, axis=1).clip(min=1e-12))
        macro = np.nanmean(vals, axis=1)
        overall[m] = {
            "weighted": {k: float(v) for k, v in _ci(weighted).items()},
            "macro": {k: float(v) for k, v in _ci(macro).items()},
        }

    out = {"per_outcome": per_outcome, "overall": overall, "n_resamples": n_resamples}
    if has_los and np.isfinite(los_samples).any():
        out["los"] = {k: float(v) for k, v in _ci(los_samples).items()}

    if has_time and np.isfinite(onset_samples).any():
        out["per_outcome_onset"] = _ci(onset_samples, axis=0)
        # Weighted by how many positives each outcome contributed IN THAT
        # resample, so an outcome that happened to draw no positives does not
        # silently carry the average.
        weights_t = n_time_samples / np.maximum(
            n_time_samples.sum(axis=1, keepdims=True), 1)
        weights_t = np.where(weights_t.sum(axis=1, keepdims=True) > 0,
                             weights_t, 1.0 / n_outcomes)
        valid_t = ~np.isnan(onset_samples)
        weighted_onset = (np.nansum(onset_samples * weights_t, axis=1)
                          / np.nansum(weights_t * valid_t, axis=1).clip(min=1e-12))
        macro_onset = np.nanmean(onset_samples, axis=1)
        out["onset"] = {
            "weighted": {k: float(v) for k, v in _ci(weighted_onset).items()},
            "macro": {k: float(v) for k, v in _ci(macro_onset).items()},
        }
    return out


# ---------------------------------------------------------------------------
# Run artefacts
# ---------------------------------------------------------------------------

def _is_scratch(dirpath: str, root: str) -> bool:
    """
    Purpose: Exclude scratch runs from every reported table.
    Method:  Any path component under the output root starting with "_" is
             scratch -- the smoke test writes to `outputs/_smoke/`. It cleans up
             after itself, but a crashed or interrupted smoke run would
             otherwise leave 1-epoch toy-width results sitting in the headline
             table, which is exactly the kind of number that gets believed.

    Args:
        dirpath (str): Candidate run directory.
        root    (str): Output root being scanned.

    Returns:
        bool: True when the directory should be skipped.
    """
    rel = os.path.relpath(dirpath, root)
    return any(part.startswith("_") for part in rel.split(os.sep))


def write_predictions(output_dir: str, patient_ids, labels, probs, outcome_names,
                      los_true=None, los_pred=None,
                      time_true=None, time_pred=None) -> str:
    """
    Purpose: Emit the unified per-patient prediction file every model shares.
    Method:  One row per test patient; `label_<outcome>` / `prob_<outcome>`
             column pairs plus the two length-of-stay columns.

    Args:
        output_dir    (str):        Run directory.
        patient_ids   (array-like): Test PatientIds, aligned with the matrices.
        labels        (np.ndarray): [N, K] ground truth.
        probs         (np.ndarray): [N, K] predicted probabilities.
        outcome_names (list[str]):  Column order.
        los_true      (np.ndarray|None): Ground-truth LoS hours.
        los_pred      (np.ndarray|None): Predicted LoS hours.
        time_true     (np.ndarray|None): [N, K] observed onset hours.
        time_pred     (np.ndarray|None): [N, K] predicted onset hours.

    Returns:
        str: Path to the written csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({"PatientId": np.asarray(patient_ids)})
    for i, name in enumerate(outcome_names):
        df[f"label_{name}"] = labels[:, i]
        df[f"prob_{name}"] = probs[:, i]
        if time_true is not None:
            df[f"time_true_{name}"] = np.asarray(time_true, dtype=float)[:, i]
        if time_pred is not None:
            df[f"time_pred_{name}"] = np.asarray(time_pred, dtype=float)[:, i]
    df["los_true_hours"] = (np.full(len(df), np.nan) if los_true is None
                            else np.asarray(los_true, dtype=float))
    df["los_pred_hours"] = (np.full(len(df), np.nan) if los_pred is None
                            else np.asarray(los_pred, dtype=float))
    path = os.path.join(output_dir, "test_predictions.csv")
    df.to_csv(path, index=False)
    return path


def score_run(output_dir: str, n_resamples: int = 2000, seed: int = 42,
              verbose: bool = True) -> dict:
    """
    Purpose: Score one finished run from its artefacts on disk.
    Method:  Read `test_predictions.csv` + `run_meta.json`, compute the
             per-outcome table, the macro/weighted aggregates, the LoS MAE and
             the bootstrap CIs, then write `test_per_outcome_metrics.csv`,
             `test_overall_metrics.csv` and `test_bootstrap.json` beside them.

    Args:
        output_dir  (str):  Run directory.
        n_resamples (int):  Bootstrap iterations (0 to skip).
        seed        (int):  Bootstrap RNG seed.
        verbose     (bool): Print the headline table.

    Returns:
        dict: {'per_outcome': DataFrame, 'overall': DataFrame,
               'los_mae': float, 'los_n': int, 'bootstrap': dict, 'meta': dict}
    """
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    df = pd.read_csv(pred_path)
    meta_path = os.path.join(output_dir, "run_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    outcome_names = meta.get("outcome_names") or [
        c[len("label_"):] for c in df.columns if c.startswith("label_")
    ]
    labels = df[[f"label_{n}" for n in outcome_names]].to_numpy(dtype=float)
    probs = df[[f"prob_{n}" for n in outcome_names]].to_numpy(dtype=float)
    los_true = df["los_true_hours"].to_numpy(dtype=float)
    los_pred = df["los_pred_hours"].to_numpy(dtype=float)

    per_outcome = per_outcome_metrics(labels, probs, outcome_names)

    # Onset timing, where the run emitted it. Models without a time head simply
    # have no `time_pred_*` columns and the whole block is skipped.
    time_cols = [f"time_pred_{n}" for n in outcome_names]
    truth_cols = [f"time_true_{n}" for n in outcome_names]
    time_pred = time_true = None
    if all(c in df.columns for c in time_cols):
        time_pred = df[time_cols].to_numpy(dtype=float)
        time_true = (df[truth_cols].to_numpy(dtype=float)
                     if all(c in df.columns for c in truth_cols)
                     else np.full_like(time_pred, np.nan))
        timing = time_to_event_mae(time_true, time_pred, labels, outcome_names)
        per_outcome = per_outcome.merge(timing, on="outcome", how="left")

    overall = aggregate(per_outcome)
    los_mae, los_n = length_of_stay_mae(los_true, los_pred)
    boot = bootstrap_metrics(labels, probs, outcome_names, los_true, los_pred,
                             time_true=time_true, time_pred=time_pred,
                             n_resamples=n_resamples, seed=seed)

    # Every per-outcome number gets its interval attached, so the CSV is
    # self-contained -- a point estimate on a 3-positive outcome is not a
    # finding, and the columns should make that visible without a second file.
    if boot:
        for metric in METRIC_NAMES:
            block = boot["per_outcome"][metric]
            per_outcome[f"{metric}_lo"] = np.asarray(block["lo"])
            per_outcome[f"{metric}_hi"] = np.asarray(block["hi"])
        if "per_outcome_onset" in boot:
            onset_block = boot["per_outcome_onset"]
            per_outcome["time_mae_h_lo"] = np.asarray(onset_block["lo"])
            per_outcome["time_mae_h_hi"] = np.asarray(onset_block["hi"])
        per_outcome["ci_reliable"] = per_outcome["n_pos"] >= MIN_POSITIVES_FOR_CI

    per_outcome.to_csv(os.path.join(output_dir, "test_per_outcome_metrics.csv"), index=False)
    overall_out = overall.copy()
    overall_out["los_mae_hours"] = los_mae
    overall_out["los_n"] = los_n
    if "time_mae_h" in per_outcome.columns:
        finite = per_outcome[per_outcome["time_mae_h"].notna()]
        if len(finite):
            weights = finite["n_time"].to_numpy(dtype=float)
            values = finite["time_mae_h"].to_numpy(dtype=float)
            overall_out.loc[overall_out["average"] == "macro", "time_mae_h"] = values.mean()
            overall_out.loc[overall_out["average"] == "weighted", "time_mae_h"] = (
                float((values * weights).sum() / weights.sum()) if weights.sum() > 0
                else values.mean())
            overall_out["n_time"] = int(finite["n_time"].sum())
    if boot and "onset" in boot:
        for average in ("macro", "weighted"):
            row = overall_out["average"] == average
            overall_out.loc[row, "time_mae_h_lo"] = boot["onset"][average]["lo"]
            overall_out.loc[row, "time_mae_h_hi"] = boot["onset"][average]["hi"]
    overall_out.to_csv(os.path.join(output_dir, "test_overall_metrics.csv"), index=False)
    if boot:
        # numpy arrays are not json-serialisable; per-outcome blocks go to lists.
        serialisable = {
            "overall": boot["overall"],
            "n_resamples": boot["n_resamples"],
            "per_outcome": {m: {k: np.asarray(v).tolist() for k, v in blk.items()}
                            for m, blk in boot["per_outcome"].items()},
            "outcome_names": outcome_names,
        }
        if "los" in boot:
            serialisable["los"] = boot["los"]
        if "onset" in boot:
            serialisable["onset"] = boot["onset"]
        if "per_outcome_onset" in boot:
            serialisable["per_outcome_onset"] = {
                k: np.asarray(v).tolist() for k, v in boot["per_outcome_onset"].items()}
        with open(os.path.join(output_dir, "test_bootstrap.json"), "w") as f:
            json.dump(serialisable, f, indent=2)

    if verbose:
        tag = (f"{meta.get('model', '?')} | K={meta.get('context_days', '?')} | "
               f"QA={meta.get('use_qa', '?')}")
        print(f"\n=== {tag} === ({output_dir})")
        print(per_outcome.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        thin = per_outcome[per_outcome["n_pos"] < MIN_POSITIVES_FOR_CI]
        if len(thin):
            print(f"  note: {len(thin)} outcome(s) have fewer than "
                  f"{MIN_POSITIVES_FOR_CI} positives on test. Their intervals "
                  f"reflect which few patients were drawn, not model uncertainty "
                  f"(ci_reliable=False): "
                  + ", ".join(f"{r.outcome}(n={r.n_pos})" for r in thin.itertuples()))
        print(overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"LoS MAE: {los_mae:.2f} h (n={los_n})")
        if "time_mae_h" in overall_out.columns:
            weighted = overall_out.loc[overall_out["average"] == "weighted", "time_mae_h"]
            if len(weighted) and np.isfinite(weighted.iloc[0]):
                ci = ""
                if boot and "onset" in boot:
                    block = boot["onset"]["weighted"]
                    ci = f" [{block['lo']:.2f}, {block['hi']:.2f}]"
                print(f"Onset MAE (support-weighted, positives only): "
                      f"{weighted.iloc[0]:.2f} h{ci}")

    return {"per_outcome": per_outcome, "overall": overall, "los_mae": los_mae,
            "los_n": los_n, "bootstrap": boot, "meta": meta}


def collect_runs(output_root: str = None) -> pd.DataFrame:
    """
    Purpose: Build the master comparison table across every finished run.
    Method:  Walk `outputs/` for directories holding both `run_meta.json` and
             `test_overall_metrics.csv`, and stack their headline rows tagged by
             model / K / QA arm.

    Args:
        output_root (str|None): Root to scan. Defaults to `outputs/`.

    Returns:
        pd.DataFrame: run, model, context_days, use_qa, average, metrics...,
                      los_mae_hours, n_test.
    """
    from kineret.config import paths as _paths
    output_root = output_root or _paths.OUTPUT_DIR

    rows = []
    for dirpath, _dirnames, filenames in os.walk(output_root):
        if "run_meta.json" not in filenames or "test_overall_metrics.csv" not in filenames:
            continue
        if _is_scratch(dirpath, output_root):
            continue
        meta = json.load(open(os.path.join(dirpath, "run_meta.json")))
        overall = pd.read_csv(os.path.join(dirpath, "test_overall_metrics.csv"))
        for _, row in overall.iterrows():
            rows.append({
                "run": os.path.relpath(dirpath, output_root),
                "model": meta.get("model"),
                "context_days": meta.get("context_days"),
                "use_qa": meta.get("use_qa"),
                "n_test": meta.get("n_test"),
                **row.to_dict(),
            })
    if not rows:
        return pd.DataFrame(columns=["run", "model", "context_days", "use_qa", "average"])
    return pd.DataFrame(rows).sort_values(
        ["model", "context_days", "use_qa", "average"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Post-hoc analyses added for the AIIM paper.
#
# Every function below reads only `test_predictions.csv` (plus optional
# per-patient covariates for subgroup work) -- no re-training required.
# These support the AIIM open-items in `papers-drafts/AIIM2027/tasks.tex`.
# ---------------------------------------------------------------------------


def load_predictions(run_dir: str) -> pd.DataFrame:
    """
    Purpose: Load a single arm's `test_predictions.csv` with the columns the
             downstream helpers expect (patient id, per-outcome labels /
             probabilities / times, and LoS true/pred).
    Method:  Read + light validation.

    Args:
        run_dir (str): Path to an arm's output directory (contains
                       `test_predictions.csv` and `run_meta.json`).

    Returns:
        pd.DataFrame: Row per test patient. Columns include `PatientId`,
                      `label_<outcome>`, `prob_<outcome>`, `time_true_<outcome>`,
                      `time_pred_<outcome>`, `los_true_hours`, `los_pred_hours`.
    """
    pred_path = os.path.join(run_dir, "test_predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)
    return pd.read_csv(pred_path)


def outcome_names_from(preds: pd.DataFrame) -> list:
    """Purpose: Recover the ordered outcome list from a prediction frame."""
    return [c[len("prob_"):] for c in preds.columns if c.startswith("prob_")]


# ---------------------------------------------------------------------------
# Subgroup / fairness metrics
# ---------------------------------------------------------------------------

def subgroup_metrics(
    run_dir: str,
    subgroup_frame: pd.DataFrame,
    subgroup_col: str,
    id_col: str = "PatientId",
    metrics=("auroc", "auprc", "best_f1"),
    min_positives: int = 5,
) -> pd.DataFrame:
    """
    Purpose: Split an arm's held-out predictions by a subgroup column
             (e.g., age band, sex, admitting-department) and compute
             per-outcome metrics inside each subgroup, so fairness /
             heterogeneity questions can be read off one table.
    Method:  Left-join predictions onto the caller-supplied subgroup frame,
             then loop over subgroup values applying `per_outcome_metrics`
             independently. Rows with fewer than `min_positives` are marked
             `ci_reliable=False`; the estimate is kept but not to be quoted
             alone.

    Args:
        run_dir           (str):  Arm's output dir (has `test_predictions.csv`).
        subgroup_frame    (DataFrame): One row per patient, indexed by
                          `id_col`; must carry `subgroup_col`.
        subgroup_col      (str):  Column to split on.
        id_col            (str):  Patient key. Default `PatientId`, matching
                          the prediction schema.
        metrics           (tuple): Metric names to include. Any subset of
                          `METRIC_NAMES`.
        min_positives     (int):  Cell-level positive-count floor for
                          `ci_reliable`. Matches `MIN_POSITIVES_FOR_CI`.

    Returns:
        pd.DataFrame: Long-form table with columns
                      [`subgroup`, `outcome`, `n`, `n_pos`, `ci_reliable`,
                       *metrics].
    """
    preds = load_predictions(run_dir)
    outcomes = outcome_names_from(preds)
    if id_col not in subgroup_frame.columns:
        raise KeyError(f"{id_col!r} not in subgroup_frame")
    if subgroup_col not in subgroup_frame.columns:
        raise KeyError(f"{subgroup_col!r} not in subgroup_frame")

    merged = preds.merge(
        subgroup_frame[[id_col, subgroup_col]].drop_duplicates(id_col),
        on=id_col, how="left",
    )
    missing = merged[subgroup_col].isna().sum()
    if missing:
        warnings.warn(
            f"{missing} patients missing subgroup value; dropped from analysis"
        )
        merged = merged.dropna(subset=[subgroup_col])

    rows = []
    for value, chunk in merged.groupby(subgroup_col, sort=True):
        labels = chunk[[f"label_{o}" for o in outcomes]].to_numpy(dtype=float)
        probs  = chunk[[f"prob_{o}"  for o in outcomes]].to_numpy(dtype=float)
        table = per_outcome_metrics(labels, probs, outcomes)
        for _, r in table.iterrows():
            n_pos = int(r["n_pos"])
            rows.append({
                "subgroup": value,
                "outcome": r["outcome"],
                "n": int(len(chunk)),
                "n_pos": n_pos,
                "ci_reliable": n_pos >= min_positives,
                **{m: r[m] for m in metrics if m in r},
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired-bootstrap Δ for cross-arm significance
# ---------------------------------------------------------------------------

def paired_bootstrap_delta(
    run_a: str,
    run_b: str,
    metric: str = "auprc",
    n_resamples: int = 2000,
    seed: int = 42,
    average: str = "weighted",
) -> dict:
    """
    Purpose: Test whether arm A beats arm B on a given metric under a
             paired patient-level bootstrap -- so cross-arm comparisons
             report a Δ CI and a p-value rather than only overlap-of-CIs.
    Method:  Both arms are scored on the same test patients. Resample the
             patient set with replacement `n_resamples` times, compute the
             metric on each arm on the same resample, and take the
             difference. Report the Δ mean, 95% CI, and a two-sided p-value
             derived from the sign of Δ across resamples (fraction crossing
             zero, doubled).

    Args:
        run_a, run_b   (str):  Two arm output directories.
        metric         (str):  One of METRIC_NAMES.
        n_resamples    (int):  Bootstrap replicates.
        seed           (int):  RNG seed.
        average        (str):  "weighted" (support-weighted) or "macro".

    Returns:
        dict: {'metric_a': float, 'metric_b': float,
               'delta_mean': float, 'delta_lo': float, 'delta_hi': float,
               'p_value_two_sided': float, 'n_test': int}.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}")
    a = load_predictions(run_a)
    b = load_predictions(run_b)
    if len(a) != len(b) or not a["PatientId"].equals(b["PatientId"]):
        raise ValueError("run_a and run_b were not scored on the same "
                         "patients in the same order; wiring check failed")

    outcomes = outcome_names_from(a)
    if outcome_names_from(b) != outcomes:
        raise ValueError("run_a and run_b have different outcome sets")

    lab = a[[f"label_{o}" for o in outcomes]].to_numpy(dtype=float)
    pa  = a[[f"prob_{o}"  for o in outcomes]].to_numpy(dtype=float)
    pb  = b[[f"prob_{o}"  for o in outcomes]].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n = lab.shape[0]
    deltas = np.empty(n_resamples, dtype=float)

    def _aggr(labels_sub, probs_sub):
        per = per_outcome_metrics(labels_sub, probs_sub, outcomes)
        agg = aggregate(per)
        row = agg[agg["average"] == average].iloc[0]
        return float(row[metric])

    metric_a = _aggr(lab, pa)
    metric_b = _aggr(lab, pb)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _aggr(lab[idx], pa[idx]) - _aggr(lab[idx], pb[idx])

    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return dict(metric_a=metric_a, metric_b=metric_b,
                    delta_mean=np.nan, delta_lo=np.nan, delta_hi=np.nan,
                    p_value_two_sided=np.nan, n_test=n)

    # Two-sided empirical p from the resample distribution.
    p_pos = np.mean(deltas > 0.0)
    p_val = 2.0 * min(p_pos, 1.0 - p_pos)

    return dict(
        metric_a=metric_a, metric_b=metric_b,
        delta_mean=float(deltas.mean()),
        delta_lo=float(np.quantile(deltas, 0.025)),
        delta_hi=float(np.quantile(deltas, 0.975)),
        p_value_two_sided=float(p_val),
        n_test=int(n),
    )


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """
    Purpose: Adjust a set of p-values with the Benjamini--Hochberg FDR
             procedure, so per-outcome comparisons across the arm ladder
             can be controlled at q < alpha.
    Method:  Standard BH: sort p, threshold at (i/m)*alpha, retain all
             p_(i) up to the largest one that passes; report adjusted
             p-values as the step-up transform.

    Args:
        pvalues (array-like of float): Raw p-values.
        alpha   (float):               FDR level.

    Returns:
        dict: {'reject': np.ndarray[bool], 'p_adjusted': np.ndarray[float]}.
              Order matches the input.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        ranked[::-1] * m / np.arange(m, 0, -1)
    )[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    p_adj = np.empty_like(p)
    p_adj[order] = adjusted
    reject = p_adj < alpha
    return {"reject": reject, "p_adjusted": p_adj}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def reliability_data(
    y_true, y_prob,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Purpose: Bin predicted probabilities against observed positive rates,
             so a reliability diagram can be plotted or a summary ECE
             computed.
    Method:  `quantile` bins put roughly equal test patients in each bin
             (good for skewed prevalence); `uniform` bins split [0, 1]
             into `n_bins` equal ranges (matches the standard ECE
             definition). Empty bins are dropped.

    Args:
        y_true      (array-like of {0, 1}): True labels.
        y_prob      (array-like of float [0, 1]): Predicted probabilities.
        n_bins      (int): Number of bins.
        strategy    (str): 'quantile' or 'uniform'.

    Returns:
        pd.DataFrame: Row per bin with `bin_low`, `bin_high`, `n`,
                      `mean_pred`, `mean_obs`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_prob) == 0:
        return pd.DataFrame(columns=["bin_low", "bin_high", "n",
                                      "mean_pred", "mean_obs"])

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # collapse ties -- may yield fewer bins
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin_low":   float(edges[b]),
            "bin_high":  float(edges[b + 1]),
            "n":         int(m.sum()),
            "mean_pred": float(y_prob[m].mean()),
            "mean_obs":  float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true, y_prob,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """
    Purpose: Summarise reliability with the expected calibration error --
             the average |predicted - observed| gap across bins, weighted
             by bin population.
    Method:  Uses `reliability_data`; defaults to uniform bins to match
             the ECE convention in the calibration literature.

    Args:
        y_true, y_prob   (array-like): Labels and predicted probabilities.
        n_bins           (int): Number of bins.
        strategy         (str): 'uniform' (standard ECE) or 'quantile'.

    Returns:
        float: ECE in probability units.
    """
    rel = reliability_data(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if rel.empty:
        return float("nan")
    weights = rel["n"] / rel["n"].sum()
    return float(np.sum(weights * np.abs(rel["mean_pred"] - rel["mean_obs"])))


def fit_temperature(y_true, y_logit_or_prob, is_prob: bool = True,
                     tol: float = 1e-4, max_iter: int = 100) -> float:
    """
    Purpose: Fit a single scalar temperature T that minimises NLL on a
             held-out set (typically validation), so per-outcome
             temperature scaling can be applied unchanged to test.
    Method:  Simple bisection over log(T) on the sigmoid-BCE loss. Robust
             enough for the modest sizes (~4.7k patients) here without
             pulling in scipy.

    Args:
        y_true              (array-like): {0, 1}.
        y_logit_or_prob     (array-like): Either raw logits or sigmoid
                            probabilities (see `is_prob`).
        is_prob             (bool): True if `y_logit_or_prob` is already
                            in [0, 1]; internally converted back to
                            logits via logit(p).
        tol, max_iter       Bisection stopping criteria.

    Returns:
        float: Optimal temperature. Values > 1 downweight confidence;
               values < 1 sharpen it.
    """
    y_true = np.asarray(y_true, dtype=float)
    z = np.asarray(y_logit_or_prob, dtype=float)
    mask = np.isfinite(z) & np.isfinite(y_true)
    y_true, z = y_true[mask], z[mask]
    if is_prob:
        # Clip to avoid inf logits at 0 or 1.
        z = np.clip(z, 1e-6, 1.0 - 1e-6)
        z = np.log(z / (1.0 - z))

    def _nll(T):
        s = z / max(T, 1e-6)
        # log(1 + exp(-|s|)) + max(-s, 0) -- numerically stable BCE.
        return float(np.mean(np.logaddexp(0.0, -s * (2.0 * y_true - 1.0))))

    lo, hi = 0.05, 20.0
    for _ in range(max_iter):
        mid1 = lo + (hi - lo) / 3.0
        mid2 = hi - (hi - lo) / 3.0
        if _nll(mid1) < _nll(mid2):
            hi = mid2
        else:
            lo = mid1
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def apply_temperature(y_prob, T: float):
    """Purpose: Rescale sigmoid probabilities by temperature T."""
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    z = np.log(p / (1.0 - p)) / max(T, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))

