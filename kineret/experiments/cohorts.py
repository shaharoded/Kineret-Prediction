"""Evaluation subsets and genuinely held-out site splits, without cache swaps."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from kineret.config import data_config as C

START = "2022-08-01"
END = "2025-12-31"


@contextmanager
def evaluation_window(k=4):
    old = C.EVAL_CONTEXT_DAYS
    C.EVAL_CONTEXT_DAYS = int(k)
    try:
        yield
    finally:
        C.EVAL_CONTEXT_DAYS = old


def people(cohort, ids):
    mapping = cohort.person_of_patient
    result = pd.Series(list(ids)).map(mapping)
    if result.isna().any():
        raise ValueError("Every admission needs a person ID for clustered CIs and splits.")
    return result.to_numpy()


def subset(cohort, ids):
    # Copy the small manifest, not the multi-GB temporal source tables.
    import copy
    new = copy.copy(cohort)
    ids = set(ids)
    new.patients = cohort.patients[cohort.patients.PatientId.isin(ids)].copy()
    new.events = cohort.events[cohort.events.PatientId.isin(ids)].copy()
    new.context = cohort.context.loc[cohort.context.index.isin(ids)].copy()
    new.qa_by_k = {k: f.loc[f.index.isin(ids)].copy() if f is not None else None
                   for k, f in cohort.qa_by_k.items()}
    new.person_of_patient = {i: p for i, p in cohort.person_of_patient.items() if i in ids}
    new.meta = deepcopy(cohort.meta)
    return new


def date_subset(cohort, start=START, end=END):
    p = cohort.patients
    dates = pd.to_datetime(p.admission_time, utc=True).dt.tz_localize(None)
    last = pd.Timestamp(end) + pd.Timedelta(days=1)
    keep = (dates >= pd.Timestamp(start)) & (dates + pd.Timedelta(days=C.HORIZON_END_DAYS) < last)
    new = subset(cohort, p.loc[keep, "PatientId"])
    new.meta["evaluation_period"] = [start, end]
    new.meta["checkpoint_training_period_unchanged"] = True
    return new


def hospital_series(cohort, visits_path):
    from kineret.loo import load_visits_master
    master = load_visits_master(visits_path)
    conflicts = master.attrs.get("conflicting_visit_ids", [])
    visits = master.set_index("VisitId").hospital
    eligible = cohort.patients.loc[~cohort.patients.PatientId.isin(conflicts), "PatientId"]
    result = visits.reindex(eligible)
    result.attrs["conflicting_visit_ids"] = conflicts
    if result.isna().any():
        raise ValueError(f"Hospital missing for {result.isna().sum()} admissions; resolve before LOO.")
    return result


def _new_split(cohort, test_ids, eligible_train_ids, seed=2023, val_fraction=.15):
    test_ids, eligible = set(test_ids), set(eligible_train_ids)
    test_people = set(people(cohort, sorted(test_ids)))
    train_pool = [i for i in sorted(eligible - test_ids)
                  if cohort.person_of_patient[i] not in test_people]
    person_ids = np.unique(people(cohort, train_pool))
    if len(person_ids) < 2 or not test_ids:
        raise ValueError("Insufficient people for a disjoint train/validation/test split.")
    rng = np.random.default_rng(seed)
    rng.shuffle(person_ids)
    nval = min(len(person_ids)-1, max(1, int(round(len(person_ids)*val_fraction))))
    val_people = set(person_ids[:nval])
    new = subset(cohort, test_ids | set(train_pool))
    new.patients["split"] = ["test" if i in test_ids else
                             "val" if new.person_of_patient[i] in val_people else "train"
                             for i in new.patients.PatientId]
    new.meta["excluded_cross_split_admissions"] = len(eligible - test_ids) - len(train_pool)
    validate_split(new)
    return new


def site_split(cohort, hospital, visits_path, seed=2023):
    sites = hospital_series(cohort, visits_path)
    new = _new_split(cohort, sites.index[sites == hospital], sites.index[sites != hospital], seed)
    new.meta["held_out_hospital"] = str(hospital)
    return new


def validate_split(cohort):
    groups = {s: set(people(cohort, cohort.patients.loc[cohort.patients.split == s, "PatientId"]))
              for s in ("train", "val", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Person overlap across splits.")
    if any(not g for g in groups.values()):
        raise ValueError("Empty split.")


def fingerprint(cohort, extra=None):
    p = cohort.patients.sort_values("PatientId")
    columns = [c for c in ("PatientId", "split", "admission_time", "trajectory_hours") if c in p]
    payload = p[columns].to_csv(index=False) + json.dumps(
        {"outcomes": cohort.outcome_names, "extra": extra,
         "person_ids": people(cohort, p.PatientId).tolist(),
         "horizon": C.HORIZON_END_DAYS}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode())
    frames = [cohort.events.sort_values(["PatientId", "outcome", "hours"]), cohort.context.sort_index()]
    frames += [f.sort_index() for _, f in sorted(cohort.qa_by_k.items()) if f is not None]
    for frame in frames:
        digest.update(str(list(frame.columns)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def export_split(cohort, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = cohort.patients.copy()
    frame["PersonId"] = people(cohort, frame.PatientId)
    frame.to_csv(directory / "split_manifest.csv", index=False)
    (directory / "cohort_provenance.json").write_text(json.dumps(cohort.meta, indent=2, default=str))


def support_table(cohort, k=4):
    """Fixed-window support by split, counting each admission once."""
    p = cohort.patients[cohort.patients.trajectory_hours > k*24]
    labels = cohort.labels_for_k(k).set_index("PatientId")
    rows = []
    for split in ("train", "val", "test"):
        group = p[p.split == split]
        for outcome in cohort.outcome_names:
            n_pos = int(labels.loc[group.PatientId, outcome].sum())
            rows.append(dict(split=split, outcome=outcome, k=k, n_admissions=len(group),
                             n_people=len(set(people(cohort, group.PatientId))), n_pos=n_pos,
                             prevalence=n_pos/len(group) if len(group) else np.nan,
                             low_support=n_pos < 5 or len(group)-n_pos < 5))
    return pd.DataFrame(rows)
