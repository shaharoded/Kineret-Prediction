"""The main notebook's small, resumable orchestration API."""
from contextlib import contextmanager
from pathlib import Path
import gc
import json
import hashlib

import numpy as np
import pandas as pd

from kineret.config import paths, data_config as C
from kineret.cohort import Cohort
from kineret.benchmark import ARM_SPEC, ARM_ORDER, run_dir_for
from .cohorts import (date_subset, evaluation_window, fingerprint, export_split,
                      site_split, hospital_series, support_table)
from .inference import checkpoint_for, predict_arm
from .statistics import attach_people, score_predictions, paired_contrasts, subgroups, calibration

CONTRASTS = [("KB vs sigma", "intervene_kb", "intervene_std"),
             ("KB vs SAX", "intervene_kb", "intervene_sax"),
             ("SAX vs sigma", "intervene_sax", "intervene_std"),
             ("KB vs STraTS", "intervene_kb", "strats"),
             ("QA in KB", "intervene_kb_qa", "intervene_kb"),
             ("QA in STraTS", "strats_qa", "strats"),
             ("QA in LogReg", "logreg_qa", "logreg")]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def load_study(cohort_path=None, start="2022-08-01", end="2025-12-31"):
    # Never ask ensure_prepared to invalidate the old cohort/checkpoint artifacts.
    source = Path(cohort_path or paths.COHORT_PKL)
    original = Cohort.load(str(source))
    cohort = date_subset(original, start, end)
    cohort.meta["source_cohort"] = str(source.resolve())
    cohort.meta["source_cohort_meta"] = original.meta
    return cohort


def inventory(output_root=None, checkpoint_root=None):
    return pd.DataFrame([dict(arm=arm, checkpoint=str(checkpoint_for(arm, output_root, checkpoint_root)),
        checkpoint_available=checkpoint_for(arm, output_root, checkpoint_root).exists(),
        test_predictions=(Path(run_dir_for(arm, output_root))/"test_predictions.csv").exists())
        for arm in ARM_ORDER])


def _stamp(path):
    path = Path(path)
    if not path.exists():
        return None
    s = path.stat()
    return dict(path=str(path.resolve()), bytes=s.st_size, modified_ns=s.st_mtime_ns)


def evaluate(cohort, destination, arms=None, k=4, source_outputs=None, checkpoint_root=None,
             device=None, n_resamples=2000, schema_paths=None, std_table=None, std_repo=None,
             arm_sources=None, include_validation=True):
    """Reuse matching exported predictions or frozen checkpoints; never train.

    Missing legacy LR artifacts are recorded per arm, not replaced with new weights.
    New statistics and predictions always go under destination, not source_outputs.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    export_split(cohort, destination)
    support_table(cohort, k).to_csv(destination/"support.csv", index=False)
    status, predictions, summaries = [], {}, []
    cohort_id = fingerprint(cohort)
    for arm in (arms or ARM_ORDER):
        print(f"[evaluate] {arm} @ K={k}")
        model = ARM_SPEC[arm][0]
        arm_source = (arm_sources or {}).get(arm, {})
        arm_outputs = arm_source.get("outputs", source_outputs)
        arm_checkpoints = arm_source.get("checkpoints", checkpoint_root)
        arm_std_table = arm_source.get("std_table", std_table)
        arm_std_repo = arm_source.get("std_repo", std_repo)
        ckpt = checkpoint_for(arm, arm_outputs, arm_checkpoints)
        source = Path(run_dir_for(arm, arm_outputs))
        target = destination / arm
        target.mkdir(exist_ok=True)
        if target.resolve() == source.resolve():
            raise ValueError("Evaluation outputs must be separate from original model outputs.")
        for split in (("test", "val") if include_validation else ("test",)):
            source_pred = source / f"{split}_predictions.csv"
            cache = target / f"{split}_predictions.csv"
            manifest = target / f"{split}_manifest.json"
            signature = dict(cohort=cohort_id, k=int(k), split=split,
                             checkpoint=_stamp(ckpt), source_predictions=_stamp(source_pred),
                             schema=_stamp((schema_paths or {}).get(arm, source/"inference_schema.pkl")),
                             std_table=_stamp(arm_std_table or Path(paths.PROCESSED_DIR)/"std_temporal_data.csv")
                             if arm == "intervene_std" else None,
                             inputs={key: _stamp(getattr(paths, key)) for key in
                                     ("RAW_TEMPORAL_FILE", "ABSTRACT_FILE", "QA_FILE", "CONTEXT_FILE", "TAK_REPO_PATH")})
            if arm == "intervene_sax":
                sax_root = ckpt.parent.parent
                signature["sax_preprocessing"] = {name: _stamp(sax_root/name) for name in
                    ("sax_design.json", "sax_training_statistics.csv", "tak_repo_sax.json")}
            try:
                if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == signature:
                    frame = pd.read_csv(cache)
                    origin = "cached_evaluation"
                elif k == 4 and source_pred.exists():
                    frame = pd.read_csv(source_pred)
                    with evaluation_window(k):
                        expected = cohort.samples(split)
                    frame = frame[frame.PatientId.isin(expected.PatientId)].copy()
                    if set(frame.PatientId) != set(expected.PatientId):
                        raise ValueError("Existing predictions do not cover the requested split; use a compatible checkpoint.")
                    # Reuse only when labels still match the frozen task exactly.
                    truth = cohort.labels_for_k(k).set_index("PatientId")
                    for outcome in cohort.outcome_names:
                        if not np.array_equal(frame[f"label_{outcome}"], truth.loc[frame.PatientId, outcome]):
                            raise ValueError("Existing predictions have different labels.")
                    frame = attach_people(frame, cohort)
                    origin = "filtered_existing_predictions"
                else:
                    kwargs = dict(device=device)
                    if arm == "intervene_std":
                        kwargs.update(std_table=arm_std_table, std_repo=arm_std_repo)
                    if model == "strats":
                        kwargs["schema_path"] = (schema_paths or {}).get(arm)
                    frame = predict_arm(arm, cohort, ckpt, k=k, split=split, **kwargs)
                    origin = "frozen_checkpoint"
                frame.to_csv(cache, index=False)
                write_json(manifest, signature)
                status.append(dict(arm=arm, split=split, status="ok", source=origin, n=len(frame)))
                if split == "test":
                    predictions[arm] = frame
                    score_path = target/"metrics.csv"
                    stat_stamp = target/"statistics_manifest.json"
                    stat_spec = dict(signature=signature, n_resamples=n_resamples, version=1)
                    if score_path.exists() and stat_stamp.exists() and json.loads(stat_stamp.read_text()) == stat_spec:
                        table = pd.read_csv(score_path)
                    else:
                        table = score_predictions(frame, n_resamples)
                        table.to_csv(score_path, index=False)
                        write_json(stat_stamp, stat_spec)
                    summaries.append(table.assign(arm=arm, k=k))
            except (FileNotFoundError, ValueError) as exc:
                status.append(dict(arm=arm, split=split, status="unavailable", detail=str(exc)))
                print(f"[{arm}/{split}] {exc}")
        gc.collect()
    status = pd.DataFrame(status)
    status.to_csv(destination/"status.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(destination/"metrics.csv", index=False)
    return predictions, summary, status


def require_predictions(predictions, expected_arms):
    missing = sorted(set(expected_arms)-set(predictions))
    if missing:
        raise RuntimeError(f"Incomplete experiment: missing test predictions for {missing}. "
                           "See status.csv for the artifacts to supply or recover.")


def analyze(predictions, directory, demographics=None, subgroup_arms=("intervene_kb", "intervene_kb_qa"),
            n_resamples=2000, do_paired=True, do_calibration=True):
    directory = Path(directory)
    contrasts = [c for c in CONTRASTS if c[1] in predictions and c[2] in predictions]
    paired = paired_contrasts(predictions, contrasts, n_resamples) if do_paired else pd.DataFrame()
    if do_paired:
        paired.to_csv(directory/"paired_contrasts.csv", index=False)
    calibration_tables, curves, subgroup_tables = [], [], []
    for arm, frame in predictions.items():
        val_path = directory/arm/"val_predictions.csv"
        # Only consume a validation export whose manifest belongs to this test evaluation.
        val = None
        if val_path.exists():
            vm = json.loads((directory/arm/"val_manifest.json").read_text())
            tm = json.loads((directory/arm/"test_manifest.json").read_text())
            if all(vm.get(key) == tm.get(key) for key in
                   ("cohort", "k", "checkpoint", "schema", "std_table", "inputs")):
                val = pd.read_csv(val_path)
        if do_calibration:
            table, curve = calibration(frame, val)
            calibration_tables.append(table.assign(arm=arm)); curves.append(curve.assign(arm=arm))
        if demographics is not None and arm in subgroup_arms:
            subgroup_tables.append(subgroups(frame, demographics, n_resamples).assign(arm=arm))
    for name, tables in (("calibration", calibration_tables), ("reliability", curves), ("subgroups", subgroup_tables)):
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(directory/f"{name}.csv", index=False)
    return paired


@contextmanager
def isolated_processing(directory):
    """Only new experiment caches change roots; original cache/checkpoint files stay put."""
    from kineret.strats import config as SC
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    old, old_sc = paths.PROCESSED_DIR, SC.PROCESSED_DIR
    paths.PROCESSED_DIR = SC.PROCESSED_DIR = str(directory)
    try:
        yield
    finally:
        paths.PROCESSED_DIR, SC.PROCESSED_DIR = old, old_sc


def prepare_sigma(cohort, directory):
    """New holdout experiments fit sigma cutoffs on training admissions only."""
    from kineret.abstraction import OUT_COLS, collapse_intervals, build_std_tak_repo
    from kineret.io_utils import normalise_temporal, load_table
    from .io import load_temporal
    raw = load_temporal(paths.RAW_TEMPORAL_FILE, cohort.patients.PatientId)
    rename = {s: canonical for canonical, spellings in cohort.structural_aliases.items() for s in spellings}
    raw.ConceptName = raw.ConceptName.replace(rename)
    blocked = set(cohort.kb_event_names) | set(cohort.replaced_spellings)
    blocked |= {s for aliases in cohort.outcome_aliases.values() for s in aliases}
    raw = raw[~raw.ConceptName.isin(blocked)].copy()
    raw["numeric_value"] = pd.to_numeric(raw.Value, errors="coerce")
    train = set(cohort.patients.loc[cohort.patients.split == "train", "PatientId"])
    stats = raw[raw.PatientId.isin(train)].groupby("ConceptName").numeric_value.agg(["mean", "std", "count"])
    stats = stats[stats["count"] > 0]
    stats["std"] = stats["std"].fillna(1.).where(stats["std"] > 0, 1.)
    numeric = raw[raw.numeric_value.notna()].merge(stats, left_on="ConceptName", right_index=True, how="inner")
    bins = pd.cut((numeric.numeric_value-numeric["mean"])/numeric["std"], C.STD_BIN_EDGES,
                  labels=C.STD_BIN_LABELS, right=False).astype(str)
    numeric.ConceptName = numeric.ConceptName.astype(str)+"_STD_"+bins
    numeric.Value = bins
    binned = pd.concat([numeric[OUT_COLS], raw.loc[raw.numeric_value.isna(), OUT_COLS]], ignore_index=True)
    table = collapse_intervals(binned)
    directory = Path(directory)
    repo = build_std_tak_repo(table.ConceptName.unique(), out_path=str(directory/"tak_repo_std.json"))
    known = set(json.loads(Path(repo).read_text())["taks"])
    table = table[table.ConceptName.isin(known)]
    table.to_csv(directory/"std_temporal_data.csv", index=False)
    stats.to_csv(directory/"sigma_training_statistics.csv")


def run_holdout(cohort, directory, arms, device=None, n_resamples=2000):
    """Train missing fold models only. Matching completed checkpoints go straight to inference."""
    from kineret.benchmark import run_one
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    outputs, checkpoints, processed = directory/"models", directory/"checkpoints", directory/"processed"
    spec = dict(cohort=fingerprint(cohort), arms=list(arms), k=4,
                task="risk in (96,336] hours", sigma_fit="training_admissions_only",
                source_inputs={key: _stamp(getattr(paths, key)) for key in
                               ("RAW_TEMPORAL_FILE", "ABSTRACT_FILE", "CONTEXT_FILE", "QA_FILE", "TAK_REPO_PATH")},
                train_windows=list(C.TRAIN_CONTEXT_DAYS), seed=C.SEED)
    manifest = directory/"training_design.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Fold design changed. Choose a new experiment directory; existing checkpoints are preserved.")
    write_json(manifest, spec); export_split(cohort, directory)
    support = support_table(cohort, 4)
    support.to_csv(directory/"support_before_training.csv", index=False)
    print(support.to_string(index=False))
    if (support.n_admissions == 0).any():
        raise ValueError("A fold has no eligible admissions at the 96-hour window; inspect support_before_training.csv.")
    with evaluation_window(4), isolated_processing(processed):
        for arm in arms:
            ckpt = checkpoint_for(arm, str(outputs), str(checkpoints))
            if ckpt.exists():
                print(f"[reuse] {arm}: {ckpt}")
                continue
            if arm == "intervene_std":
                prepare_sigma(cohort, processed)
            if ARM_SPEC[arm][0] == "strats":
                from kineret.strats.preprocess import build_pickle
                from kineret.strats.train import run
                data_path = build_pickle(cohort, ARM_SPEC[arm][1])
                run(use_qa=ARM_SPEC[arm][1], output_root=str(outputs), device=device,
                    dataset_path=data_path, bootstrap_resamples=0)
            else:
                run_one(arm, cohort=cohort, output_root=str(outputs), checkpoint_root=str(checkpoints),
                        bootstrap_resamples=0, device=device)
            gc.collect()
        result = evaluate(cohort, directory/"evaluation", arms, source_outputs=str(outputs),
                          checkpoint_root=str(checkpoints), device=device, n_resamples=n_resamples,
                          std_table=str(processed/"std_temporal_data.csv"), std_repo=str(processed/"tak_repo_std.json"))
        require_predictions(result[0], arms)
        analyze(result[0], directory/"evaluation", n_resamples=n_resamples)
        from .figures import export_figures
        export_figures(directory/"evaluation")
    return result


def hospital_loo(cohort, directory, visits_path, arms=("strats", "intervene_kb", "intervene_kb_qa"),
                 hospitals=None, **kwargs):
    sites = hospital_series(cohort, visits_path)
    Path(directory).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PatientId": sites.attrs.get("conflicting_visit_ids", [])}).to_csv(
        Path(directory)/"excluded_hospital_conflicts.csv", index=False)
    tables = []
    for site in (hospitals if hospitals is not None else sorted(sites.unique())):
        folded = site_split(cohort, site, visits_path)
        tag = hashlib.sha256(str(site).encode()).hexdigest()[:12]
        _, table, _ = run_holdout(folded, Path(directory)/tag, arms, **kwargs)
        tables.append(table.assign(hospital=site))
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(Path(directory)/"site_metrics.csv", index=False)
    return result


def train_missing_main(cohort, directory, arms=None, source_outputs=None, checkpoint_root=None, **kwargs):
    """Optional activation: fit only arms with neither checkpoint nor existing test predictions.

    New models have separate per-arm directories; old checkpoints are never moved.
    Return source overrides so every subsequent analysis uses the selected model.
    """
    sources = {}
    for arm in (arms or ARM_ORDER):
        if checkpoint_for(arm, source_outputs, checkpoint_root).exists():
            continue
        if (Path(run_dir_for(arm, source_outputs))/"test_predictions.csv").exists():
            print(f"[preserve] {arm}: existing predictions; missing inference artifact does not trigger training.")
            continue
        root = Path(directory)/arm
        source = dict(outputs=str(root/"models"), checkpoints=str(root/"checkpoints"),
                      std_table=str(root/"processed/std_temporal_data.csv"),
                      std_repo=str(root/"processed/tak_repo_std.json"))
        local_checkpoint = checkpoint_for(arm, source["outputs"], source["checkpoints"])
        if local_checkpoint.exists():
            # Primary evaluations may change dates while keeping the trained
            # split and weights. Fold experiments retain their stricter guards.
            print(f"[reuse] {arm}: {local_checkpoint}")
        else:
            run_holdout(cohort, root, [arm], **kwargs)
        sources[arm] = source
    return sources
