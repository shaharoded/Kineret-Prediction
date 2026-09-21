"""
Build the ss-STraTS input pickle for one QA arm.

Adapted from `med-transformers-baseline/scripts/preprocess_mimic_iv.py`. Three
things changed, all deliberate:

1. **Labels are no longer derived here.** The original script re-implemented the
   Mediator's glucose / creatinine rules so it could synthesise events from
   MIMIC. Here the Mediator has already run, so labels come from
   `kineret.cohort` -- identical to what INTERVenE-Enc and LogReg are trained
   against. This file builds *inputs* only.

2. **Training is augmented over context windows.** The unit of the table is no
   longer a patient but a **sample**: a (patient, K) pair. A train patient
   contributes one sample per K in `TRAIN_CONTEXT_DAYS`, each clipped to its own
   window with the label window re-derived to match. Validation and test carry
   one sample per patient at `EVAL_CONTEXT_DAYS`, so model selection and every
   reported number describe a single stated window.

3. **No forecasting stage.** This is ss-STraTS, the supervised-only variant, so
   there is one pickle and no pretrain table.

Output shape is otherwise unchanged, so `dataset.py` reads it as before:

    [data, labels, train_ids, valid_ids, test_ids, metadata]

`data` is a long table -- ts_id / minute / variable / value -- where `ts_id` is
a sample id, with that sample's static context replayed at minute 0.0.
"""

import os
import pickle

import numpy as np
import pandas as pd

from kineret.cohort import Cohort
from kineret.config import data_config as C
from kineret.config import paths
from kineret.io_utils import load_table, normalise_temporal, to_numeric_value
from kineret.strats import config as SC


def _build_input_table(raw: pd.DataFrame, samples: pd.DataFrame,
                       static_long: pd.DataFrame, blocked: set,
                       support_threshold: float):
    """
    Purpose: Turn raw temporal rows into the long (ts_id, minute, variable, value)
             table, one block per augmented sample.
    Method:  For each context window present in `samples`, take the rows of that
             window's patients inside [0, K*24] h, relabel `PatientId` to the
             sample id, and stack. A patient therefore appears once per training
             window, each time with strictly less history than the next.

             Values are coerced to float; rows whose Value is categorical become
             indicator variables named `<Concept>_<Value>` with value 1.0 -- the
             trick the official STraTS uses to keep categorical events in scope.
             The static context is appended at minute 0 and duplicate
             (sample, minute, variable) triples are averaged.

    Args:
        raw               (pd.DataFrame): Raw rows with `PatientId` and `minute`.
        samples           (pd.DataFrame): sample_id / PatientId / k / split.
        static_long       (pd.DataFrame): Melted static context, per sample.
        blocked           (set):          Leakage blocklist.
        support_threshold (float):        Drop concepts below this sample support.

    Returns:
        tuple[pd.DataFrame, list, list]: (data, kept_concepts, dropped_concepts).
    """
    cols = ["ts_id", "minute", "variable", "value"]
    usable = raw[(raw["minute"] >= 0) & (~raw["ConceptName"].isin(blocked))]

    blocks = []
    for k, group in samples.groupby("k", sort=True):
        limit = float(k) * 24.0 * 60.0
        sample_of = dict(zip(group["PatientId"], group["sample_id"]))
        block = usable[(usable["minute"] <= limit)
                       & (usable["PatientId"].isin(sample_of))].copy()
        block["ts_id"] = block["PatientId"].map(sample_of)
        blocks.append(block)
    ti = pd.concat(blocks, ignore_index=True)

    if support_threshold > 0:
        n_samples = max(samples["sample_id"].nunique(), 1)
        support = ti.groupby("ConceptName")["ts_id"].nunique() / n_samples
        kept = sorted(support[support >= support_threshold].index)
        dropped = sorted(support[support < support_threshold].index)
        ti = ti[ti["ConceptName"].isin(kept)].copy()
    else:
        kept, dropped = sorted(ti["ConceptName"].unique()), []

    ti["value"] = to_numeric_value(ti["Value"])
    categorical = ti["value"].isna()
    ti.loc[categorical, "ConceptName"] = (
        ti.loc[categorical, "ConceptName"].astype(str) + "_"
        + ti.loc[categorical, "Value"].astype(str)
    )
    ti.loc[categorical, "value"] = 1.0
    ti = ti.dropna(subset=["value"])
    ti = ti.rename(columns={"ConceptName": "variable"})[cols]

    data = pd.concat([ti, static_long[cols]], ignore_index=True)
    data = data.groupby(["ts_id", "minute", "variable"], as_index=False)["value"].mean()
    return data, kept, dropped


def build_pickle(cohort: Cohort, use_qa: bool, raw: pd.DataFrame = None,
                 verbose: bool = True) -> str:
    """
    Purpose: Write the ss-STraTS input pickle for one QA arm.
    Method:  Enumerate the cohort's augmented samples, harmonise the raw table
             onto the canonical event support, clip each sample to its own
             window, attach that sample's static context (which carries the
             aggregated QA block in the QA arm) and the cohort's labels for that
             sample's K.

    Args:
        cohort  (Cohort):            Shared cohort artefact.
        use_qa  (bool):              QA ablation arm.
        raw     (pd.DataFrame|None): Pre-loaded raw table; read from disk when
                                     None. Pass it in so the file is read once.
        verbose (bool):              Print a summary.

    Returns:
        str: Path to the written pickle.
    """
    say = print if verbose else (lambda *a, **kw: None)
    os.makedirs(SC.PROCESSED_DIR, exist_ok=True)

    if raw is None:
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))

    samples = cohort.samples()
    train_ids = cohort.sample_ids("train")
    valid_ids = cohort.sample_ids("val")
    test_ids = cohort.sample_ids("test")

    keep_patients = set(cohort.patients["PatientId"])
    raw = raw[raw["PatientId"].isin(keep_patients)].copy()
    # Canonical event support, shared with every other model. No-op when the
    # caller already handed over a harmonised frame.
    raw = cohort.harmonise_events(raw, patient_ids=cohort.patients["PatientId"],
                                  verbose=verbose)
    admission = cohort.patients.set_index("PatientId")["admission_time"]
    raw["minute"] = ((raw["StartDateTime"] - raw["PatientId"].map(admission))
                     .dt.total_seconds() / 60.0)

    # Static context, per SAMPLE: the QA block is aggregated over that sample's
    # own window, so a K=2 sample never sees four days of compliance.
    context = cohort.context_for_samples(samples, use_qa)
    static_varis = list(context.columns)
    static_long = (context.rename_axis("ts_id").reset_index()
                   .melt(id_vars="ts_id", value_vars=static_varis,
                         var_name="variable", value_name="value"))
    static_long["minute"] = 0.0

    blocked = cohort.leakage_blocklist()
    say(f"[strats/preprocess] qa={use_qa}: blocking {len(blocked)} terminus "
        f"concept(s); {len(samples):,} samples "
        f"({len(train_ids):,} train / {len(valid_ids):,} val / {len(test_ids):,} test).")

    data, kept, dropped = _build_input_table(
        raw, samples, static_long, blocked, SC.CONCEPT_SUPPORT_THRESHOLD)

    labels = cohort.labels_for_samples(samples).rename(columns={"sample_id": "ts_id"})
    outcomes = list(cohort.outcome_names)

    # Length-of-stay: z-scored on the TRAIN samples. The evaluator denormalises
    # the head's output back to hours before reporting MAE.
    train_mask = labels["ts_id"].isin(set(train_ids.tolist()))
    train_los = labels.loc[train_mask, "length_of_stay_hours"].dropna()
    los_mean = float(train_los.mean()) if len(train_los) else 0.0
    los_std = float(train_los.std()) if len(train_los) > 1 else 1.0
    if not np.isfinite(los_std) or los_std <= 0:
        los_std = 1.0

    # Per-outcome onset normalisation, on TRAIN POSITIVES only. The time head
    # answers "given that this happens, when?", so negatives (first_hour = inf)
    # carry no target and are masked out of the statistics and of the loss.
    time_mean, time_std = {}, {}
    for outcome in outcomes:
        hours = labels.loc[train_mask, f"{outcome}__first_hour"]
        hours = hours[np.isfinite(hours)]
        time_mean[outcome] = float(hours.mean()) if len(hours) else 0.0
        std = float(hours.std()) if len(hours) > 1 else 1.0
        time_std[outcome] = std if (np.isfinite(std) and std > 0) else 1.0

    metadata = {
        "static_varis": static_varis,
        "outcome_names": outcomes,
        "dropped_outcomes": cohort.dropped_outcomes,
        "train_context_days": list(C.TRAIN_CONTEXT_DAYS),
        "eval_context_days": C.EVAL_CONTEXT_DAYS,
        "use_qa": use_qa,
        "input_hours": C.EVAL_CONTEXT_DAYS * 24.0,
        "horizon_end_hours": C.HORIZON_END_DAYS * 24.0,
        "los_mean": los_mean, "los_std": los_std,
        "time_mean": time_mean, "time_std": time_std,
        "kept_input_concepts": kept,
        "dropped_input_concepts": dropped,
        "blocked_concepts": sorted(blocked),
        "concept_support_threshold": SC.CONCEPT_SUPPORT_THRESHOLD,
        # sample -> patient, so the evaluator can key its predictions by
        # patient like every other arm does.
        "sample_patient": dict(zip(samples["sample_id"], samples["PatientId"])),
    }

    path = SC.dataset_pkl(use_qa)
    with open(path, "wb") as f:
        pickle.dump([data, labels, train_ids, valid_ids, test_ids, metadata], f,
                    protocol=4)

    say(f"[strats/preprocess] wrote {path}")
    say(f"    rows={len(data):,}  samples={data['ts_id'].nunique():,}  "
        f"variables={data['variable'].nunique():,}")
    test_labels = labels[labels["ts_id"].isin(set(test_ids.tolist()))]
    say(f"    test prevalence @ K={C.EVAL_CONTEXT_DAYS}: "
        + ", ".join(f"{o}={test_labels[o].mean():.1%}" for o in outcomes))
    return path
