from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import os
import pandas as pd


def _drop_nonfinite(y_true, y_pred):
    """Return only the rows where y_pred is finite (some freshly-initialised
    models emit NaN on edge-case inputs — patients with all-masked
    observation triplets, etc. Filter those out so sklearn doesn't choke)."""
    mask = np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _safe_auc_pair(y_true, y_pred):
    """AUROC/AUPRC; returns (nan, nan) when only one class is present
    or when no finite predictions remain after filtering."""
    y_true, y_pred = _drop_nonfinite(y_true, y_pred)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    return (
        roc_auc_score(y_true, y_pred),
        average_precision_score(y_true, y_pred),
    )


def _max_f1_pair(y_true, y_pred):
    """max-F1 from sweeping the PR curve plus F1 at the fixed 0.5 threshold."""
    y_true, y_pred = _drop_nonfinite(y_true, y_pred)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    f1_curve = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    return (
        np.nanmax(f1_curve),
        float(f1_score(y_true, y_pred >= 0.5)),
        np.minimum(precision, recall).max(),
    )


def _bootstrap_test_metrics(true, pred, outcome_names,
                            los_pred_hours, los_true_hours, los_mask,
                            n_resamples, rng_seed):
    """
    Purpose: 2,000-resample patient-level bootstrap on the held-out test set
             so per-outcome AUROC / AUPRC / max-F1 / F1@0.5 / minRP and the
             length-of-stay MAE (denormalised, RELEASE-discharged subset)
             each carry a 95 % CI from a single training run — matching the
             protocol used by INTERVenE-Ar / INTERVenE-Enc.
    Method:  Sample patient indices with replacement; recompute each metric
             on every resample; report mean and the [2.5 %, 97.5 %] quantiles.

    Returns:
        dict with per-outcome arrays (point estimates + CI bounds) and an
        overall section keyed by 'support_weighted' / 'macro' / 'los'.
    """
    rng = np.random.default_rng(rng_seed)
    n_patients = true.shape[0]
    n_outcomes = len(outcome_names)
    metric_names = ('auroc', 'auprc', 'best_f1', 'f1_0_5', 'minrp')

    # Per-outcome accumulators (one row per resample).
    samples = {m: np.full((n_resamples, n_outcomes), np.nan) for m in metric_names}
    # Per-outcome positive support per resample (needed for the support-
    # weighted overall aggregate inside each bootstrap iteration).
    n_pos_samples = np.zeros((n_resamples, n_outcomes), dtype=float)
    # Length-of-stay MAE per resample (RELEASE-discharged subset only).
    los_samples = np.full(n_resamples, np.nan)

    for b in range(n_resamples):
        idx = rng.integers(0, n_patients, size=n_patients)
        for i in range(n_outcomes):
            y_true = true[idx, i]
            y_pred = pred[idx, i]
            samples['auroc'][b, i], samples['auprc'][b, i] = _safe_auc_pair(y_true, y_pred)
            max_f1, f1_05, minrp = _max_f1_pair(y_true, y_pred)
            samples['best_f1'][b, i] = max_f1
            samples['f1_0_5'][b, i]  = f1_05
            samples['minrp'][b, i]   = minrp
            n_pos_samples[b, i] = y_true.sum()
        # LoS MAE on the RELEASE-discharged subset of the resample.
        mask_b = los_mask[idx]
        if mask_b.any():
            err = np.abs(los_pred_hours[idx][mask_b] - los_true_hours[idx][mask_b])
            err = err[np.isfinite(err)]
            if err.size:
                los_samples[b] = float(err.mean())

    def _ci_block(arr_2d):
        """Return point estimate (mean across resamples) + 95 % CI per outcome."""
        return {
            'mean':  np.nanmean(arr_2d, axis=0),
            'lo':    np.nanpercentile(arr_2d, 2.5,  axis=0),
            'hi':    np.nanpercentile(arr_2d, 97.5, axis=0),
        }

    per_outcome = {m: _ci_block(samples[m]) for m in metric_names}

    # Support-weighted and macro aggregates per bootstrap iteration, then CI.
    overall = {}
    for m in metric_names:
        vals = samples[m]  # [n_resamples, n_outcomes]
        # Support weights per resample, normalised; rows with all-zero support
        # (no positives at all) fall back to a uniform mean so the row is not
        # discarded.
        weights = n_pos_samples / np.maximum(n_pos_samples.sum(axis=1, keepdims=True), 1)
        weights = np.where(weights.sum(axis=1, keepdims=True) > 0, weights, 1.0 / n_outcomes)
        weighted = np.nansum(vals * weights, axis=1) / np.nansum(weights * ~np.isnan(vals), axis=1).clip(min=1e-12)
        macro = np.nanmean(vals, axis=1)
        overall[m] = {
            'support_weighted': {
                'mean': float(np.nanmean(weighted)),
                'lo':   float(np.nanpercentile(weighted, 2.5)),
                'hi':   float(np.nanpercentile(weighted, 97.5)),
            },
            'macro': {
                'mean': float(np.nanmean(macro)),
                'lo':   float(np.nanpercentile(macro, 2.5)),
                'hi':   float(np.nanpercentile(macro, 97.5)),
            },
        }

    los_summary = {
        'mean': float(np.nanmean(los_samples)),
        'lo':   float(np.nanpercentile(los_samples, 2.5)),
        'hi':   float(np.nanpercentile(los_samples, 97.5)),
    }

    return {
        'per_outcome': per_outcome,
        'overall':     overall,
        'los':         los_summary,
        'n_resamples': n_resamples,
    }


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.bootstrap_resamples = int(getattr(args, 'bootstrap_resamples', 2000))
        self.bootstrap_seed      = int(getattr(args, 'bootstrap_seed', 42))

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0,num_samples,self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            # In eval mode we do not need supervision tensors; drop the
            # binary labels and the LoS regression target/mask before
            # moving inputs to the GPU.
            for k in ('labels', 'los_target_norm', 'los_mask',
                      'time_target_norm', 'time_mask'):
                batch.pop(k, None)
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())
        true, pred = torch.cat(true).numpy(), torch.cat(pred).numpy()
        if true.ndim==1:
            true = true.reshape(-1, 1)
            pred = pred.reshape(-1, 1)
        # The head emits [K risk probs | K normalised onset times | 1 normalised
        # LoS]. Split it, denormalise both regressions back to hours, and leave
        # `pred` as the K probabilities so it lines up with `true` again.
        n_outcomes = len(self.args.outcome_names)
        los_pred_norm = pred[:, -1]
        time_pred_norm = pred[:, n_outcomes:2 * n_outcomes]
        pred = pred[:, :n_outcomes]

        time_mean = np.array([getattr(self.args, 'time_mean', {}).get(o, 0.0)
                              for o in self.args.outcome_names])
        time_std = np.array([getattr(self.args, 'time_std', {}).get(o, 1.0) or 1.0
                             for o in self.args.outcome_names])
        time_pred_hours = time_pred_norm * time_std[None, :] + time_mean[None, :]
        los_mean = getattr(self.args, 'los_mean', 0.0)
        los_std  = getattr(self.args, 'los_std',  1.0) or 1.0
        los_pred_hours = los_pred_norm * los_std + los_mean
        los_true_hours = dataset.los_hours_raw[eval_ind]
        los_mask = dataset.los_mask[eval_ind].astype(bool)
        # A freshly-initialised model can emit NaN LoS predictions on edge-case
        # inputs; combine the RELEASE-discharged mask with a finite-prediction
        # mask so the MAE never propagates NaN through the run.
        los_eval_mask = los_mask & np.isfinite(los_pred_hours) & np.isfinite(los_true_hours)
        if los_eval_mask.any():
            los_mae_hours = float(np.abs(los_pred_hours[los_eval_mask] - los_true_hours[los_eval_mask]).mean())
        else:
            los_mae_hours = float('nan')
        rows = []
        for i, outcome in enumerate(self.args.outcome_names):
            y_true_full, y_pred_full = true[:, i], pred[:, i]
            n_pos_total = int(y_true_full.sum())
            n_neg_total = int((1 - y_true_full).sum())
            # Drop rows with non-finite preds (a freshly-initialised model can
            # emit NaN on patients with all-masked observation triplets — we
            # exclude those rows rather than crash sklearn).
            y_true, y_pred = _drop_nonfinite(y_true_full, y_pred_full)
            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                rows.append({'outcome':outcome, 'auroc':np.nan, 'auprc':np.nan,
                             'minrp':np.nan, 'f1_0_5':np.nan, 'best_f1':np.nan,
                             'n_pos':n_pos_total, 'n_neg':n_neg_total})
                continue
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            f1_curve = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
            rows.append({'outcome':outcome,
                         'auroc':roc_auc_score(y_true, y_pred),
                         'auprc':average_precision_score(y_true, y_pred),
                         'minrp':np.minimum(precision, recall).max(),
                         'f1_0_5':f1_score(y_true, y_pred >= 0.5),
                         'best_f1':np.nanmax(f1_curve),
                         'n_pos':n_pos_total, 'n_neg':n_neg_total})
        table = pd.DataFrame(rows).set_index('outcome')
        result = {'auroc':float(table['auroc'].mean(skipna=True)),
                  'auprc':float(table['auprc'].mean(skipna=True)),
                  'minrp':float(table['minrp'].mean(skipna=True)),
                  'f1_0_5':float(table['f1_0_5'].mean(skipna=True)),
                  'best_f1':float(table['best_f1'].mean(skipna=True)),
                  'los_mae_hours':los_mae_hours,
                  'los_n_valid':int(los_mask.sum())}

        # Headline confidence intervals come from a 2,000-resample patient-
        # level bootstrap on the held-out test set, matching the INTERVenE
        # protocol so the cross-method comparison stays apples-to-apples.
        # Val / eval_train stay as single point estimates (used only for the
        # early-stop selector, which is invariant to CIs).
        if split == 'test' and self.bootstrap_resamples > 0:
            boot = _bootstrap_test_metrics(
                true, pred, self.args.outcome_names,
                los_pred_hours, los_true_hours, los_mask,
                n_resamples=self.bootstrap_resamples,
                rng_seed=self.bootstrap_seed,
            )
            # Add per-outcome CI columns to the printed table.
            for metric in ('auroc', 'auprc', 'best_f1', 'f1_0_5', 'minrp'):
                table[metric + '_lo'] = boot['per_outcome'][metric]['lo']
                table[metric + '_hi'] = boot['per_outcome'][metric]['hi']
            # Surface the support-weighted and macro CI bands in the summary.
            for metric in ('auroc', 'auprc', 'best_f1', 'f1_0_5', 'minrp'):
                result[metric + '_w_mean'] = boot['overall'][metric]['support_weighted']['mean']
                result[metric + '_w_lo']   = boot['overall'][metric]['support_weighted']['lo']
                result[metric + '_w_hi']   = boot['overall'][metric]['support_weighted']['hi']
                result[metric + '_macro_lo'] = boot['overall'][metric]['macro']['lo']
                result[metric + '_macro_hi'] = boot['overall'][metric]['macro']['hi']
            result['los_mae_hours_lo'] = boot['los']['lo']
            result['los_mae_hours_hi'] = boot['los']['hi']
            result['bootstrap_resamples'] = boot['n_resamples']
        if split=='test':
            self.write_outputs(dataset, eval_ind, true, pred, table,
                               los_pred_hours=los_pred_hours,
                               los_true_hours=los_true_hours,
                               los_mask=los_mask,
                               time_pred_hours=time_pred_hours)
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                              +str(train_step)+': '+str(result))
            self.args.logger.write('\nPer-outcome '+split+' results:\n'+str(table))
        return result

    def write_outputs(self, dataset, eval_ind, true, pred, table,
                      los_pred_hours=None, los_true_hours=None, los_mask=None,
                      time_pred_hours=None):
        """
        Purpose: Write the held-out test artefacts for this run.
        Method:  Kineret tweak -- the prediction file now uses the repo-wide
                 schema (`label_<outcome>` / `prob_<outcome>` /
                 `los_true_hours` / `los_pred_hours`) so
                 `kineret.evaluation.score_run` can score ss-STraTS,
                 INTERVenE-Enc and LogReg with the same code path. The
                 STraTS-specific extras (`first_hour_*`, `has_release`) ride
                 along as additional columns; the shared scorer ignores them.
        """
        table.to_csv(os.path.join(self.args.output_dir, 'test_per_outcome_metrics.csv'))
        # Key by PATIENT, not by sample: a test sample is one per patient at
        # the evaluation window, and every arm must write the same index for
        # the cross-arm label check to be able to compare them.
        sample_ids = [dataset.ind_to_ts_id[i] for i in eval_ind]
        sample_patient = getattr(self.args, 'sample_patient', {}) or {}
        patient_ids = [sample_patient.get(s, s) for s in sample_ids]
        pred_df = pd.DataFrame({'PatientId':patient_ids})
        for i, outcome in enumerate(self.args.outcome_names):
            pred_df['label_'+outcome] = true[:, i]
            pred_df['prob_'+outcome]  = pred[:, i]
            if time_pred_hours is not None:
                pred_df['time_pred_'+outcome] = time_pred_hours[:, i]
            if dataset.first_hours is not None:
                # inf marks "never occurred"; the scorer masks those out of the
                # onset MAE rather than scoring against a placeholder.
                truth = dataset.first_hours[eval_ind, i]
                pred_df['time_true_'+outcome] = np.where(np.isfinite(truth), truth, np.nan)
                pred_df['first_hour_'+outcome] = truth
        if los_pred_hours is not None:
            # LoS ground truth is NaN for patients with no RELEASE (died / no
            # terminus). Blank those out rather than scoring against a
            # placeholder, matching the shared scorer's finite-only masking.
            pred_df['los_pred_hours'] = los_pred_hours
            pred_df['los_true_hours'] = np.where(los_mask, los_true_hours, np.nan)
            pred_df['has_release']    = los_mask.astype(int)
        else:
            pred_df['los_pred_hours'] = np.nan
            pred_df['los_true_hours'] = np.nan
        pred_df.to_csv(os.path.join(self.args.output_dir, 'test_predictions.csv'), index=False)
