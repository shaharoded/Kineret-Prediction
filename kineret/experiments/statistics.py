"""Person-clustered, paired evaluation on immutable prediction tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from kineret.evaluation import (outcome_names_from, expected_calibration_error,
                              reliability_data, fit_temperature, apply_temperature,
                              benjamini_hochberg)
from .cohorts import people


def attach_people(preds, cohort):
    if preds.PatientId.duplicated().any():
        raise ValueError("Expected one prediction per admission at a fixed K.")
    out = preds.copy()
    out["PersonId"] = people(cohort, out.PatientId)
    return out.sort_values("PatientId").reset_index(drop=True)


def cluster_indices(ids, n_resamples=2000, seed=2023):
    ids = np.asarray(ids)
    if not len(ids) or pd.isna(ids).any():
        raise ValueError("Nonempty, complete person IDs are required.")
    if n_resamples < 1:
        raise ValueError("At least one bootstrap resample is required for experiment statistics.")
    codes, unique = pd.factorize(ids, sort=True)
    groups = [np.flatnonzero(codes == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])


def _metric(y, p, metric):
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        return np.nan
    if metric == "brier":
        return float(np.mean((p-y)**2))
    if len(np.unique(y)) < 2:
        return np.nan
    return float((average_precision_score if metric == "auprc" else roc_auc_score)(y, p))


def metric_vector(frame, outcomes, metric):
    values = np.array([_metric(frame[f"label_{o}"].to_numpy(float),
                               frame[f"prob_{o}"].to_numpy(float), metric) for o in outcomes])
    support = np.array([frame[f"label_{o}"].sum() for o in outcomes], float)
    # Do not silently change the target set in degenerate bootstrap draws.
    if np.isfinite(values).all():
        macro = values.mean()
        weighted = np.average(values, weights=support) if support.sum() else np.nan
    else:
        macro = weighted = np.nan
    return np.r_[values, macro, weighted]


def _interval(draws):
    finite = np.asarray(draws)[np.isfinite(draws)]
    return (*np.quantile(finite, [.025, .975]), len(finite)) if len(finite) else (np.nan, np.nan, 0)


def score_predictions(frame, n_resamples=2000, seed=2023):
    outcomes = outcome_names_from(frame)
    names = outcomes + ["macro", "weighted"]
    rows = []
    for metric in ("auprc", "auroc", "brier"):
        point = metric_vector(frame, outcomes, metric)
        draws = np.array([metric_vector(frame.iloc[idx], outcomes, metric)
                          for idx in cluster_indices(frame.PersonId, n_resamples, seed)])
        for j, name in enumerate(names):
            lo, hi, valid = _interval(draws[:, j])
            npos = int(frame[f"label_{name}"].sum()) if name in outcomes else np.nan
            rows.append(dict(outcome=name, metric=metric, estimate=point[j], lo=lo, hi=hi,
                             n=len(frame), n_people=frame.PersonId.nunique(), n_pos=npos,
                             prevalence=npos/len(frame), valid_resamples=valid,
                             ci_reliable=valid >= .95*n_resamples and
                             (name not in outcomes or (npos >= 5 and len(frame)-npos >= 5))))
    # Regression heads retain their original definitions and get clustered CIs too.
    pairs = [(o, f"time_true_{o}", f"time_pred_{o}") for o in outcomes]
    pairs += [("length_of_stay", "los_true_hours", "los_pred_hours")]
    for name, truth, pred in pairs:
        if truth not in frame or pred not in frame:
            continue
        err = np.abs(frame[truth].to_numpy(float) - frame[pred].to_numpy(float))
        if name in outcomes:
            err[frame[f"label_{name}"].to_numpy() != 1] = np.nan
        err[~np.isfinite(err)] = np.nan
        valid = np.isfinite(err)
        if not valid.any():
            continue
        draws = [np.nanmean(err[idx]) if np.isfinite(err[idx]).any() else np.nan
                 for idx in cluster_indices(frame.PersonId, n_resamples, seed)]
        lo, hi, count = _interval(draws)
        rows.append(dict(outcome=name, metric="mae_hours", estimate=np.nanmean(err), lo=lo, hi=hi,
                         n=len(frame), n_people=frame.PersonId.nunique(), n_pos=int(valid.sum()),
                         valid_resamples=count, ci_reliable=valid.sum() >= 5 and count >= .95*n_resamples))
    return pd.DataFrame(rows)


def paired_contrasts(predictions, contrasts, n_resamples=2000, seed=2023):
    rows = []
    for title, arm_a, arm_b in contrasts:
        a, b = [predictions[k].sort_values("PatientId").reset_index(drop=True) for k in (arm_a, arm_b)]
        outcomes = outcome_names_from(a)
        cols = ["PatientId", "PersonId"] + [f"label_{o}" for o in outcomes]
        if outcomes != outcome_names_from(b) or not a[cols].equals(b[cols]):
            raise ValueError(f"{title}: paired predictions must have identical IDs, people and labels.")
        for metric in ("auprc", "auroc"):
            point = metric_vector(a, outcomes, metric)-metric_vector(b, outcomes, metric)
            draws = np.array([metric_vector(a.iloc[idx], outcomes, metric)-metric_vector(b.iloc[idx], outcomes, metric)
                              for idx in cluster_indices(a.PersonId, n_resamples, seed)])
            for j, outcome in enumerate(outcomes + ["macro", "weighted"]):
                lo, hi, nvalid = _interval(draws[:, j])
                d = draws[:, j]; d = d[np.isfinite(d)]
                # Center the bootstrap error distribution under H0; finite-sample correction.
                p = ((1+np.sum(np.abs(d-point[j]) >= abs(point[j])))/(len(d)+1)
                     if len(d) and np.isfinite(point[j]) else np.nan)
                npos = int(a[f"label_{outcome}"].sum()) if outcome in outcomes else np.nan
                reliable = nvalid >= .95*n_resamples and (outcome not in outcomes or
                                                          (npos >= 5 and len(a)-npos >= 5))
                rows.append(dict(contrast=title, outcome=outcome, metric=metric,
                                 delta=point[j], lo=lo, hi=hi, p_value=p, valid_resamples=nvalid,
                                 n=len(a), n_people=a.PersonId.nunique(), n_pos=npos,
                                 ci_reliable=reliable))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["family"] = np.where(result.outcome.isin(["macro", "weighted"]),
        "headline:" + result.metric + ":" + result.outcome,
        "per_outcome:" + result.contrast + ":" + result.metric)
    result["q_value"] = np.nan
    for _, g in result.groupby("family"):
        valid = g.p_value.notna() & g.ci_reliable
        ix = g.index[valid]
        if len(ix):
            result.loc[ix, "q_value"] = benjamini_hochberg(result.loc[ix, "p_value"].to_numpy())["p_adjusted"]
    return result


def subgroups(frame, demographics, n_resamples=2000, seed=2023):
    joined = frame.merge(demographics, on="PatientId", validate="one_to_one")
    if "age" in joined:
        joined["age_band"] = pd.cut(pd.to_numeric(joined.age, errors="coerce"),
            [-np.inf, 55, 65, 75, 85, np.inf], right=False,
            labels=["<55", "55–64", "65–74", "75–84", "85+"])
    tables = []
    for column in ("age_band", "sex", "hospital"):
        if column not in joined:
            continue
        for group, chunk in joined.groupby(column, observed=True, dropna=(column == "hospital")):
            table = score_predictions(chunk, n_resamples, seed)
            table["subgroup_type"], table["subgroup"] = column, str(group)
            tables.append(table)
    if not tables:
        raise ValueError("No age, sex or hospital columns mapped for subgroup analysis.")
    return pd.concat(tables, ignore_index=True)


def calibration(test, validation=None):
    if validation is not None and set(test.PersonId) & set(validation.PersonId):
        raise ValueError("Calibration validation and test people overlap.")
    rows, curves = [], []
    for outcome in outcome_names_from(test):
        y, p = (test[f"{prefix}_{outcome}"].to_numpy(float) for prefix in ("label", "prob"))
        variants = [("raw", p, 1.)]
        if validation is not None:
            vy, vp = (validation[f"{prefix}_{outcome}"].to_numpy(float) for prefix in ("label", "prob"))
            if len(np.unique(vy[np.isfinite(vy)])) == 2:
                temperature = fit_temperature(vy, vp)
                variants.append(("validation_temperature", apply_temperature(p, temperature), temperature))
        for name, prob, temperature in variants:
            rows.append(dict(outcome=outcome, variant=name, temperature=temperature,
                             n=len(y), n_pos=int(y.sum()), ece_15=expected_calibration_error(y, prob, n_bins=15),
                             brier=_metric(y, prob, "brier")))
            curve = reliability_data(y, prob, n_bins=10, strategy="quantile")
            curve["outcome"], curve["variant"] = outcome, name
            curves.append(curve)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)
