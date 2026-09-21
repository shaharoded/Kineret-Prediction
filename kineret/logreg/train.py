"""
Logistic-regression baseline -- the non-interval, non-neural reference point.

Implemented in PyTorch rather than scikit-learn for two reasons: it runs on the
same GPU as the other two models (the multi-label problem is one dense matmul,
so a K-outcome fit is a single batched op), and it shares their training
contract exactly -- multi-label BCE with positive-class weighting, early
stopping on val AUROC+AUPRC, one held-out test pass at the end. The model is
still a plain linear-in-features logistic regression: one weight vector and one
bias per outcome, L2 via weight decay, no hidden layer.

A length-of-stay head rides along as a second linear map on the same features,
so the LoS MAE column is populated for this model too.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kineret.cohort import Cohort
from kineret.config import data_config as C
from kineret.config import paths
from kineret.evaluation import score_run, write_predictions
from kineret.io_utils import load_table, normalise_temporal
from kineret.logreg.features import assemble_design_matrix, build_feature_frame

LOGREG_SETTINGS = {
    "lr": 1e-2,
    "weight_decay": 1e-3,     # the L2 penalty; the only regulariser in play
    "max_epochs": 500,
    "patience": 30,
    "batch_size": 0,          # 0 = full batch; the design matrix fits in VRAM
    "los_loss_weight": 1.0,
    # Matches INTERVenE's `phase3_time_lambda`, so onset timing is weighted
    # against risk identically in both.
    "time_loss_weight": 0.1,
    "use_pos_weight": True,
}


class MultiLabelLogisticRegression(nn.Module):
    """
    Multi-label logistic regression with an auxiliary length-of-stay head.

    Three linear maps off the same feature vector, so this baseline answers the
    same three questions as every other model in the benchmark: will the
    complication happen, when, and how long is the stay.

    Attributes:
        risk (nn.Linear): in_features -> K outcome logits.
        time (nn.Linear): in_features -> K z-scored onset times.
        los  (nn.Linear): in_features -> 1 z-scored length-of-stay prediction.
    """

    def __init__(self, in_features: int, n_outcomes: int):
        """
        Args:
            in_features (int): Design-matrix width.
            n_outcomes  (int): Number of prediction targets.
        """
        super().__init__()
        self.risk = nn.Linear(in_features, n_outcomes)
        self.time = nn.Linear(in_features, n_outcomes)
        self.los = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Purpose: Score a batch of patients.

        Args:
            x (Tensor): [B, F] standardised features.

        Returns:
            tuple[Tensor, Tensor, Tensor]: ([B, K] risk logits,
                                            [B, K] z-scored onset,
                                            [B] z-scored LoS).
        """
        return self.risk(x), self.time(x), self.los(x).squeeze(-1)


def _standardise(train_X, *others):
    """
    Purpose: Z-score the design matrix on TRAIN statistics only.
    Method:  Zero-variance columns keep a unit scale so they map to a constant
             0 rather than producing inf.

    Args:
        train_X (np.ndarray): TRAIN design matrix.
        others  (np.ndarray): Further matrices to transform with the same stats.

    Returns:
        list[np.ndarray]: Standardised matrices, TRAIN first.
    """
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return [((m - mean) / std).astype(np.float32) for m in (train_X, *others)]


def _val_metric(y_true, probs):
    """
    Purpose: The early-stopping selector.
    Method:  Macro AUROC + macro AUPRC, the same criterion the STraTS driver
             uses, so model selection is not a confound in the comparison.

    Args:
        y_true (np.ndarray): [N, K] labels.
        probs  (np.ndarray): [N, K] probabilities.

    Returns:
        float: Selection score (higher is better).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    aurocs, auprcs = [], []
    for i in range(y_true.shape[1]):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            continue
        aurocs.append(roc_auc_score(y, probs[:, i]))
        auprcs.append(average_precision_score(y, probs[:, i]))
    if not aurocs:
        return -np.inf
    return float(np.mean(aurocs) + np.mean(auprcs))


def run(context_days=None, use_qa=False, cohort: Cohort = None, raw: pd.DataFrame = None,
        output_root=None, device=None, seed=C.SEED, overrides=None,
        bootstrap_resamples=2000, **_ignored):
    """
    Purpose: Fit and score the logistic-regression baseline for one sweep cell.
    Method:  Build one feature row per augmented sample -- a (patient, context
             window) pair -- from the raw table, standardise on TRAIN, fit
             multi-label BCE + masked onset MSE + masked LoS MSE with AdamW,
             early-stop on validation AUROC+AUPRC at the single evaluation
             window, then score the held-out test split with the shared scorer.

    Args:
        context_days        (int|None): Accepted for call-site symmetry; the
                            windows are a study-level decision and come from the
                            cohort's sample table.
        use_qa              (bool): QA ablation arm.
        cohort              (Cohort|None): Shared cohort; loaded when None.
        raw                 (pd.DataFrame|None): Pre-loaded raw table with an
                            `hours` column; loaded and anchored when None.
        output_root         (str|None): Defaults to `outputs/`.
        device              (str|None): Torch device string.
        seed                (int):  RNG seed.
        overrides           (dict|None): Hyperparameter overrides.
        bootstrap_resamples (int):  Resamples for the final CIs.

    Returns:
        dict: {'output_dir': str, 'scores': dict}
    """
    settings = dict(LOGREG_SETTINGS)
    settings.update(overrides or {})
    torch.manual_seed(seed)
    np.random.seed(seed)

    cohort = cohort or Cohort.load()
    if raw is None:
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        admission = cohort.patients.set_index("PatientId")["admission_time"]
        raw = raw[raw["PatientId"].isin(set(cohort.patients["PatientId"]))].copy()
        raw["hours"] = ((raw["StartDateTime"] - raw["PatientId"].map(admission))
                        .dt.total_seconds() / 3600.0)

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    output_root = output_root or paths.OUTPUT_DIR
    run_dir = os.path.join(output_root, "logreg", "qa" if use_qa else "noqa")
    os.makedirs(run_dir, exist_ok=True)

    samples = cohort.samples()
    train_s = samples[samples["split"] == "train"].reset_index(drop=True)
    val_s = samples[samples["split"] == "val"].reset_index(drop=True)
    test_s = samples[samples["split"] == "test"].reset_index(drop=True)
    train_ids = train_s["sample_id"].to_numpy()
    val_ids = val_s["sample_id"].to_numpy()
    test_ids = test_s["sample_id"].to_numpy()

    outcomes = list(cohort.outcome_names)
    blocked = cohort.leakage_blocklist()
    # Same canonical event support as ss-STraTS and INTERVenE. No-op when the
    # caller already handed over a harmonised frame (`benchmark.raw_with_hours`).
    if not raw.attrs.get("kineret_events_harmonised"):
        raw = cohort.harmonise_events(raw, patient_ids=cohort.patients["PatientId"])
        admission = cohort.patients.set_index("PatientId")["admission_time"]
        raw["hours"] = ((raw["StartDateTime"] - raw["PatientId"].map(admission))
                        .dt.total_seconds() / 3600.0)

    labels = cohort.labels_for_samples(samples).set_index("sample_id")

    # Variable vocabulary and imputation statistics are fitted on TRAIN only.
    feat_train, keep_vars = build_feature_frame(raw, train_s, blocked)
    feat_val, _ = build_feature_frame(raw, val_s, blocked, keep_vars)
    feat_test, _ = build_feature_frame(raw, test_s, blocked, keep_vars)

    context = cohort.context_for_samples(samples, use_qa)
    X_train, medians = assemble_design_matrix(feat_train, context)
    X_val, _ = assemble_design_matrix(feat_val, context, medians)
    X_test, _ = assemble_design_matrix(feat_test, context, medians)
    # Val/test can lack a column TRAIN produced (and vice versa); pin the layout.
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    print(f"[logreg] qa={use_qa}: {X_train.shape[1]} features, "
          f"{len(train_ids)} train samples (augmented over "
          f"K={list(C.TRAIN_CONTEXT_DAYS)}) / {len(val_ids)} val / "
          f"{len(test_ids)} test @ K={C.EVAL_CONTEXT_DAYS}.")

    Xtr, Xva, Xte = _standardise(X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy())
    ytr = labels.loc[train_ids, outcomes].to_numpy(dtype=np.float32)
    yva = labels.loc[val_ids, outcomes].to_numpy(dtype=np.float32)
    yte = labels.loc[test_ids, outcomes].to_numpy(dtype=np.float32)

    # Length-of-stay is z-scored on TRAIN; patients without a RELEASE are
    # masked out of the loss rather than imputed.
    los_tr = labels.loc[train_ids, "length_of_stay_hours"].to_numpy(dtype=np.float64)
    los_mean = float(np.nanmean(los_tr)) if np.isfinite(los_tr).any() else 0.0
    los_std = float(np.nanstd(los_tr)) if np.isfinite(los_tr).any() else 1.0
    los_std = los_std if np.isfinite(los_std) and los_std > 0 else 1.0

    def _los_tensors(ids):
        raw_los = labels.loc[ids, "length_of_stay_hours"].to_numpy(dtype=np.float64)
        mask = np.isfinite(raw_los)
        target = np.where(mask, (raw_los - los_mean) / los_std, 0.0)
        return (torch.tensor(target, dtype=torch.float32, device=device),
                torch.tensor(mask.astype(np.float32), device=device), raw_los)

    # Onset-time targets, z-scored per outcome on TRAIN POSITIVES only -- a
    # complication that never occurs has no onset to learn from.
    first_hour_cols = [f"{o}__first_hour" for o in outcomes]
    train_hours = labels.loc[train_ids, first_hour_cols].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        time_mean = np.array([
            np.nanmean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
            for col in train_hours.T])
        time_std = np.array([
            np.nanstd(col[np.isfinite(col)]) if np.isfinite(col).sum() > 1 else 1.0
            for col in train_hours.T])
    time_std = np.where(np.isfinite(time_std) & (time_std > 0), time_std, 1.0)

    def _time_tensors(ids):
        raw_hours = labels.loc[ids, first_hour_cols].to_numpy(dtype=np.float64)
        mask = np.isfinite(raw_hours)
        target = np.where(mask, (raw_hours - time_mean[None, :]) / time_std[None, :], 0.0)
        return (torch.tensor(target, dtype=torch.float32, device=device),
                torch.tensor(mask.astype(np.float32), device=device),
                np.where(mask, raw_hours, np.nan))

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Xtr_t, Xva_t, Xte_t = t(Xtr), t(Xva), t(Xte)
    ytr_t, yva_t = t(ytr), t(yva)
    los_tr_t, los_tr_mask, _ = _los_tensors(train_ids)
    _, _, los_te_raw = _los_tensors(test_ids)
    time_tr_t, time_tr_mask, _ = _time_tensors(train_ids)
    _, _, time_te_raw = _time_tensors(test_ids)

    # Positive-class weighting: rare complications would otherwise be predicted
    # away entirely by an unweighted BCE.
    if settings["use_pos_weight"]:
        n_pos = ytr.sum(axis=0)
        pos_weight = t((len(ytr) - n_pos) / np.maximum(n_pos, 1.0))
    else:
        pos_weight = None

    model = MultiLabelLogisticRegression(Xtr.shape[1], len(outcomes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])

    best_metric, best_state, wait = -np.inf, None, settings["patience"]
    batch_size = settings["batch_size"] or len(Xtr_t)

    for epoch in range(settings["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            risk_logits, time_pred, los_pred = model(Xtr_t[idx])
            loss = F.binary_cross_entropy_with_logits(
                risk_logits, ytr_t[idx], pos_weight=pos_weight)
            t_mask = time_tr_mask[idx]
            if t_mask.sum() > 0:
                sq_err = (time_pred - time_tr_t[idx]) ** 2 * t_mask
                loss = loss + settings["time_loss_weight"] * sq_err.sum() / t_mask.sum().clamp(min=1.0)
            mask = los_tr_mask[idx]
            if mask.sum() > 0:
                sq_err = (los_pred - los_tr_t[idx]) ** 2 * mask
                loss = loss + settings["los_loss_weight"] * sq_err.sum() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xva_t)[0]).cpu().numpy()
        metric = _val_metric(yva, val_probs)
        if metric > best_metric:
            best_metric, wait = metric, settings["patience"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait -= 1
            if wait == 0:
                print(f"[logreg] Early stop at epoch {epoch} "
                      f"(best val AUROC+AUPRC = {best_metric:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits, test_time, test_los = model(Xte_t)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        los_pred_hours = test_los.cpu().numpy() * los_std + los_mean
        time_pred_hours = (test_time.cpu().numpy() * time_std[None, :]
                           + time_mean[None, :])

    # Keyed by PATIENT: test samples are one per patient at the evaluation
    # window, and every arm must write the same index so the cross-arm label
    # check can compare them.
    write_predictions(run_dir, test_s["PatientId"].to_numpy(),
                      yte, test_probs, outcomes,
                      los_true=los_te_raw, los_pred=los_pred_hours,
                      time_true=time_te_raw, time_pred=time_pred_hours)
    meta = {
        "model": "logreg",
        "context_days": C.EVAL_CONTEXT_DAYS,
        "train_context_days": list(C.TRAIN_CONTEXT_DAYS),
        "use_qa": use_qa,
        "seed": seed,
        "outcome_names": outcomes,
        "label_window_hours": [C.EVAL_CONTEXT_DAYS * 24.0, C.HORIZON_END_DAYS * 24.0],
        "n_train_samples": int(len(train_ids)), "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "n_features": int(Xtr.shape[1]),
        "best_val_metric": float(best_metric),
        "hyperparameters": settings,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Coefficients are the reason to run a linear model at all -- keep them.
    coef = pd.DataFrame(model.risk.weight.detach().cpu().numpy().T,
                        index=X_train.columns, columns=outcomes)
    coef.to_csv(os.path.join(run_dir, "coefficients.csv"))

    scores = score_run(run_dir, n_resamples=bootstrap_resamples, seed=seed)
    return {"output_dir": run_dir, "scores": scores}
