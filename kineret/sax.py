"""Fixed SAX-style abstraction for irregular EMR: global train scaling, 4h means, 7 symbols.

Numeric segment values become available at the segment END. Prefix views discard
unfinished segments before merging intervals, so a future reading cannot affect
an earlier prediction window. No imputation, QA tokens, or clinical thresholds.
"""
from pathlib import Path
from statistics import NormalDist
import json

import numpy as np
import pandas as pd

from kineret.config import paths, data_config as C
from kineret.abstraction import OUT_COLS
from kineret.experiments.io import load_temporal
from kineret.experiments.cohorts import fingerprint

RECIPE = dict(method="SAX-style time-bin means", segment_hours=4, alphabet_size=7,
              normalization="training-population per-concept mean/std of numeric observations",
              missing="leave empty", merge="contiguous identical symbols after prefix clipping",
              boundaries="(start,end], with admission-time readings in the first segment",
              availability="segment end", version=1)
SYMBOLS = [f"S{i}" for i in range(1, 8)]
BREAKPOINTS = [NormalDist().inv_cdf(i/7) for i in range(1, 7)]
RECIPE["gaussian_breakpoints"] = BREAKPOINTS


def raw_view(cohort, raw=None):
    """Same knowledge-free event access as sigma; retain raw categories and framing."""
    df = raw.copy() if raw is not None else load_temporal(paths.RAW_TEMPORAL_FILE, cohort.patients.PatientId)
    df = df[df.PatientId.isin(cohort.patients.PatientId)].copy()
    rename = {s: canonical for canonical, aliases in cohort.structural_aliases.items() for s in aliases}
    df.ConceptName = df.ConceptName.replace(rename)
    df.loc[df.ConceptName.isin(cohort.structural_aliases), "Value"] = "True"
    blocked = set(cohort.outcome_names) | set(cohort.kb_event_names) | set(cohort.replaced_spellings)
    blocked |= {s for aliases in cohort.outcome_aliases.values() for s in aliases}
    # Treatment patterns are KB products, never supplied to this arm.
    df = df[~df.ConceptName.isin(blocked) & ~df.ConceptName.str.contains("PATTERN", case=False, na=False)].copy()
    origins = cohort.patients.set_index("PatientId").admission_time
    df["hours"] = (pd.to_datetime(df.StartDateTime)-df.PatientId.map(origins)).dt.total_seconds()/3600
    return df[df.hours.between(0, C.HORIZON_END_DAYS*24)].copy()


def fit_statistics(raw, train_ids):
    train = raw[raw.PatientId.isin(set(train_ids))].copy()
    train["number"] = pd.to_numeric(train.Value, errors="coerce")
    train = train[np.isfinite(train.number)]
    stats = train.groupby("ConceptName").number.agg(["count", "mean", "std"])
    stats["std"] = stats["std"].where(np.isfinite(stats["std"]) & (stats["std"] > 0), 1.)
    if stats.empty:
        raise ValueError("SAX requires numeric observations in training admissions.")
    return stats


def transform(raw, cohort, stats):
    df = raw.copy()
    df["number"] = pd.to_numeric(df.Value, errors="coerce")
    numeric = df[np.isfinite(df.number) & df.ConceptName.isin(stats.index)].copy()
    categorical = df[df.number.isna()].copy()
    categorical["SAXAvailableHour"] = np.nan
    # A reading at the cutoff belongs to the segment ending at that cutoff.
    numeric["segment"] = np.maximum(0, np.ceil(numeric.hours/RECIPE["segment_hours"]).astype(int)-1)
    means = numeric.groupby(["PatientId", "ConceptName", "segment"], as_index=False).number.mean()
    means = means.merge(stats[["mean", "std"]], left_on="ConceptName", right_index=True, how="inner")
    z = (means.number-means["mean"])/means["std"]
    codes = np.searchsorted(BREAKPOINTS, z.to_numpy(), side="right")
    symbols = np.asarray(SYMBOLS)[codes]
    origins = cohort.patients.set_index("PatientId").admission_time
    means["StartDateTime"] = means.PatientId.map(origins)+pd.to_timedelta(means.segment*4, unit="h")
    means["SAXAvailableHour"] = (means.segment+1)*4
    means["EndDateTime"] = means.PatientId.map(origins)+pd.to_timedelta(means.SAXAvailableHour, unit="h")
    means["ConceptName"] = means.ConceptName.astype(str)+"_SAX_"+symbols
    means["Value"] = symbols
    columns = OUT_COLS+["SAXAvailableHour"]
    result = pd.concat([means[columns], categorical[columns]], ignore_index=True)
    return result.sort_values(["PatientId", "StartDateTime", "ConceptName"]).reset_index(drop=True)


def prefix_view(table, k=None):
    """Filter atomic segments by availability, THEN merge adjacent identical symbols."""
    if "SAXAvailableHour" not in table:
        raise ValueError("SAX table is missing segment-availability metadata.")
    numeric = table[table.SAXAvailableHour.notna()].copy()
    other = table[table.SAXAvailableHour.isna()].copy()
    if k is not None:
        numeric = numeric[numeric.SAXAvailableHour <= float(k)*24].copy()
    numeric = numeric.sort_values(["PatientId", "ConceptName", "StartDateTime"])
    if not numeric.empty:
        grouped = numeric.groupby(["PatientId", "ConceptName"], sort=False)
        previous_end = grouped.EndDateTime.shift()
        numeric["run"] = (previous_end.isna() | (numeric.StartDateTime != previous_end)).cumsum()
        numeric = numeric.groupby(["PatientId", "ConceptName", "Value", "run"], as_index=False).agg(
            StartDateTime=("StartDateTime", "min"), EndDateTime=("EndDateTime", "max"),
            SAXAvailableHour=("SAXAvailableHour", "max"))
    return pd.concat([numeric[OUT_COLS+["SAXAvailableHour"]], other[OUT_COLS+["SAXAvailableHour"]]],
                     ignore_index=True).sort_values(["PatientId", "StartDateTime", "ConceptName"]).reset_index(drop=True)


def build_repository(stats, raw, directory):
    base = json.loads(Path(paths.TAK_REPO_PATH).read_text(encoding="utf-8"))
    taks = base["taks"]
    added_roots = []
    observed = set(raw.ConceptName)
    for concept in stats.index:
        if concept not in taks:
            if concept not in observed:
                raise ValueError(f"SAX training concept {concept!r} is absent from raw input.")
            # Raw measurements need no clinical definition to be symbolized.
            # Register a plain root, without inventing KB ancestry or thresholds.
            taks[concept] = dict(name=concept, family="raw-concept", derived_from=None,
                                 attributes=[], description="Raw input measurement for SAX")
            added_roots.append(concept)
        for symbol in SYMBOLS:
            name = f"{concept}_SAX_{symbol}"
            taks[name] = dict(name=name, family="state", derived_from=concept, attributes=[],
                              description=f"Population-relative SAX symbol {symbol} of {concept}")
    pd.DataFrame({"ConceptName": added_roots}).to_csv(
        Path(directory)/"sax_added_raw_roots.csv", index=False)
    if added_roots:
        print(f"[SAX] Registered {len(added_roots)} raw measurement roots missing from the KB hierarchy.")
    target = Path(directory)/"tak_repo_sax.json"
    target.write_text(json.dumps(base), encoding="utf-8")
    return str(target), set(taks)


def ensure_sax_temporal(cohort, directory, raw=None):
    """Fit once in this model/fold directory and retain the exact preprocessing artifacts."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    design = dict(recipe=RECIPE, cohort=fingerprint(cohort),
                  source=str(Path(paths.RAW_TEMPORAL_FILE).resolve()),
                  hierarchy=str(Path(paths.TAK_REPO_PATH).resolve()))
    for key, path in (("source", paths.RAW_TEMPORAL_FILE), ("hierarchy", paths.TAK_REPO_PATH)):
        source_path = Path(path)
        if source_path.exists():
            stat = source_path.stat()
            design[key+"_stamp"] = [stat.st_size, stat.st_mtime_ns]
    manifest = directory/"sax_design.json"
    table_path, repo_path = directory/"sax_temporal_data.csv", directory/"tak_repo_sax.json"
    stats_path = directory/"sax_training_statistics.csv"
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError("SAX preprocessing design changed; use a new model directory.")
    if all(p.exists() for p in (manifest, table_path, repo_path, stats_path)):
        return str(table_path), str(repo_path)
    source = raw_view(cohort, raw)
    train_ids = cohort.patients.loc[cohort.patients.split == "train", "PatientId"]
    stats = fit_statistics(source, train_ids)
    repo, known = build_repository(stats, source, directory)
    table = transform(source, cohort, stats)
    unknown = table.loc[~table.ConceptName.isin(known)].groupby("ConceptName").size()
    unknown.rename("n_rows").to_csv(directory/"sax_unresolved_concepts.csv")
    table = table[table.ConceptName.isin(known)]
    table.to_csv(table_path, index=False)
    stats.to_csv(stats_path)
    manifest.write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(f"[SAX] {len(stats)} training concepts; {len(table):,} rows; recipe: 4h/7 symbols/no imputation")
    return str(table_path), repo


def frozen_view(cohort, checkpoint_directory, k=4):
    directory = Path(checkpoint_directory)
    design = json.loads((directory/"sax_design.json").read_text())
    if design["recipe"] != RECIPE:
        raise ValueError("Saved SAX recipe is incompatible with this implementation.")
    stats = pd.read_csv(directory/"sax_training_statistics.csv", index_col="ConceptName")
    raw = raw_view(cohort)
    table = prefix_view(transform(raw, cohort, stats), k)
    repo_path = directory/"tak_repo_sax.json"
    known = set(json.loads(repo_path.read_text())["taks"])
    return table[table.ConceptName.isin(known)], str(repo_path)
