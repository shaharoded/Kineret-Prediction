"""
Knowledge-free temporal abstraction -- the bottom rung of the ablation ladder.

INTERVenE-Enc is an interval model: it consumes symbolic intervals, not raw
measurements. So "does the knowledge base earn its keep?" cannot be answered by
feeding INTERVenE raw numbers -- it has to be fed *intervals derived without
knowledge*, and compared against the Mediator's.

This module builds that arm. Ported from
`autoresearch/autoresearch-encoder/ablation/preprocess_std_bins.py`, which is
the ablation this study replicates:

  * numeric measurements are z-scored per concept and cut into 7 fixed bins at
    +-0.5 sigma, +-1 sigma, +-2 sigma -- purely distributional, no clinical
    thresholds;
  * the concept becomes `<CONCEPT>_STD_<BIN>` and the value the bin label;
  * consecutive same-concept observations within 24 h collapse into one
    interval, which is what gives the encoder START/END pairs to model;
  * categorical rows and structural framing pass through, canonically renamed.

Statistics are computed globally rather than on train only. That matches the
Mediator, which abstracts over the whole dataset, and keeps the arms comparable
-- an arm fitted on train-only cutoffs would be handicapped against a KB that
saw everything.

The resulting table plugs into the same `DataProcessor` the KB arm uses, so the
two arms differ in exactly one thing: where the intervals came from.
"""

import json
import os

import numpy as np
import pandas as pd

from kineret.config import data_config as C
from kineret.config import paths
from kineret.io_utils import load_table, normalise_temporal

OUT_COLS = ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def std_bin(temporal: pd.DataFrame):
    """
    Purpose: Replace numeric measurements with distribution-derived symbols.
    Method:  Per ConceptName, compute (mean, std) over every observation, z-score
             each reading and cut it at the configured sigma edges. The concept
             name absorbs the bin (`GLUCOSE_MEASURE_STD_HIGH`) and the value
             becomes the bin label, matching the Mediator's
             `<concept>` + symbolic-`Value` shape.

             Rows whose Value does not parse as a number -- booleans, meal
             categories, renamed `*_EVENT` markers -- pass through untouched;
             they are already indicator-style tokens.

    Args:
        temporal (pd.DataFrame): Canonical temporal rows.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (binned rows, per-concept stats).
    """
    df = temporal.copy()
    df["numeric_value"] = pd.to_numeric(df["Value"], errors="coerce")
    numeric = df[df["numeric_value"].notna()].copy()
    other = df[df["numeric_value"].isna()].copy()

    if len(numeric) == 0:
        return other[OUT_COLS].copy(), pd.DataFrame(columns=["ConceptName", "count", "mean", "std"])

    stats = (numeric.groupby("ConceptName")["numeric_value"]
                    .agg(["count", "mean", "std"]).reset_index())
    # A concept measured once, or with a constant value, has std 0 or NaN. Unit
    # scale sends every reading to NORMAL, which is the correct behaviour: a
    # channel with no variance carries no state information.
    stats["std"] = stats["std"].fillna(1.0)
    stats.loc[stats["std"] <= 0, "std"] = 1.0
    stats = stats[stats["count"] > 0]

    numeric = numeric.merge(stats[["ConceptName", "mean", "std"]],
                            on="ConceptName", how="inner")
    z = (numeric["numeric_value"] - numeric["mean"]) / numeric["std"]
    # right=False puts each upper edge in the next bin, so z = 0.5 lands in
    # SLIGHTLY_HIGH rather than NORMAL.
    bins = pd.cut(z, bins=C.STD_BIN_EDGES, labels=C.STD_BIN_LABELS, right=False)
    numeric["ConceptName"] = (numeric["ConceptName"].astype(str)
                              + "_STD_" + bins.astype(str))
    numeric["Value"] = bins.astype(str)

    return (pd.concat([numeric[OUT_COLS], other[OUT_COLS]], ignore_index=True),
            stats)


def collapse_intervals(rows: pd.DataFrame, max_gap_hours=None) -> pd.DataFrame:
    """
    Purpose: Turn repeated point observations into the intervals INTERVenE models.
    Method:  Per (patient, concept), consecutive observations no more than
             `max_gap_hours` apart join one run; the run becomes a single row
             spanning first-to-last observation time. The gap is checked
             pairwise against the previous observation, so a chain of readings
             12 h apart merges into one long interval while a 30 h gap breaks it.

             `EndDateTime` is the last observation's START, not its end: the raw
             table's own EndDateTime is unreliable for point events.

    Args:
        rows          (pd.DataFrame): Binned rows.
        max_gap_hours (float|None):   Merge threshold. Defaults to config.

    Returns:
        pd.DataFrame: Collapsed intervals in canonical column order.
    """
    max_gap_hours = C.STD_COLLAPSE_HOURS if max_gap_hours is None else max_gap_hours
    df = rows.copy()
    df["StartDateTime"] = pd.to_datetime(df["StartDateTime"])
    df = df.sort_values(["PatientId", "ConceptName", "StartDateTime"]).reset_index(drop=True)

    prev = df.groupby(["PatientId", "ConceptName"])["StartDateTime"].shift()
    gap_hours = (df["StartDateTime"] - prev).dt.total_seconds() / 3600.0
    df["group_id"] = (gap_hours.isna() | (gap_hours > max_gap_hours)).cumsum()

    agg = (df.groupby(["PatientId", "ConceptName", "group_id"], sort=False)
             .agg(StartDateTime=("StartDateTime", "first"),
                  EndDateTime=("StartDateTime", "last"),
                  Value=("Value", "first"))
             .reset_index().drop(columns=["group_id"]))
    return agg[OUT_COLS]


# ---------------------------------------------------------------------------
# TAK repository for the std arm
# ---------------------------------------------------------------------------

def build_std_tak_repo(concept_names, source_repo_path=None, out_path=None) -> str:
    """
    Purpose: Let the std arm resolve its concepts through the same machinery.
    Method:  `DataProcessor` walks every concept back to its raw parents through
             the TAK repository and raises on anything it does not recognise --
             and `GLUCOSE_MEASURE_STD_HIGH` is not a TAK. Rather than weaken that
             check, emit an augmented repository in which each `<X>_STD_<BIN>`
             is registered as derived from `<X>`.

             This is deliberate, not a workaround: the embedder uses the raw
             parent as a hierarchy level, so registering the parentage keeps the
             std arm on the same architecture as the KB arm. The ablation then
             isolates abstraction QUALITY rather than accidentally testing what
             happens when the hierarchy breaks.

    Args:
        concept_names    (Iterable[str]): Concepts appearing in the std table.
        source_repo_path (str|None):      Base TAK repo. Defaults to config.
        out_path         (str|None):      Destination. Defaults to
                                          `data/processed/tak_repo_std.json`.

    Returns:
        str: Path to the augmented repository.
    """
    source_repo_path = source_repo_path or paths.TAK_REPO_PATH
    out_path = out_path or os.path.join(paths.PROCESSED_DIR, "tak_repo_std.json")

    with open(source_repo_path, encoding="utf-8") as f:
        repo_json = json.load(f)
    taks = repo_json["taks"]

    added, unknown = 0, []
    for name in sorted({str(c) for c in concept_names}):
        if name in taks:
            continue
        if "_STD_" in name:
            base = name.rsplit("_STD_", 1)[0]
            if base not in taks:
                unknown.append(name)
                continue
            taks[name] = {
                "name": name,
                "family": "state",
                "derived_from": base,
                "attributes": [],
                "description": f"Distribution-derived state bin of {base} "
                               f"(knowledge-free ablation arm).",
            }
            added += 1
        else:
            unknown.append(name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(repo_json, f)

    print(f"[abstraction] Augmented TAK repo -> {out_path} "
          f"({added} std-bin concepts registered).")
    if unknown:
        print(f"[abstraction] {len(unknown)} concepts have no TAK entry and no "
              f"resolvable parent; they are dropped from the std arm: "
              f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}")
    return out_path


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_std_temporal(cohort, raw: pd.DataFrame = None, include_kb_events=None,
                       verbose=True):
    """
    Purpose: Produce the knowledge-free temporal table INTERVenE's std arm reads.
    Method:  Canonically rename structural and outcome concepts (so the encoder's
             ADMISSION / terminal logic works identically in both arms), std-bin
             the numeric channels, collapse them into intervals, then write an
             augmented TAK repository covering the new concept names.

             `include_kb_events` decides whether the Mediator's derived
             complication events are injected as input tokens. Default False:
             without a knowledge base you do not get `HYPERGLYCEMIA_EVENT`
             either, and pretending otherwise would hand the "no knowledge" arm
             the knowledge base's output. Set True to isolate the representation
             difference alone, holding event availability constant.

    Args:
        cohort            (Cohort):          Shared cohort artefact.
        raw               (pd.DataFrame|None): Raw table with `hours`; loaded when None.
        include_kb_events (bool|None):       Override the config default.
        verbose           (bool):            Print progress.

    Returns:
        tuple[pd.DataFrame, str]: (std temporal table, augmented TAK repo path).
    """
    say = print if verbose else (lambda *a, **kw: None)
    include_kb_events = (C.STD_INCLUDE_KB_EVENTS if include_kb_events is None
                         else include_kb_events)

    if raw is None:
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
    df = raw[raw["PatientId"].isin(set(cohort.patients["PatientId"]))].copy()
    df = df[OUT_COLS].copy()
    say(f"[abstraction] std arm from {len(df):,} raw rows.")

    # --- canonical renaming ---------------------------------------------
    # Structural framing only. Outcome rows are handled by the harmonisation
    # step below, which is also what puts them on the canonical support.
    rename = {}
    for canonical, spellings in cohort.structural_aliases.items():
        rename.update({s: canonical for s in spellings})
    n_renamed = int(df["ConceptName"].isin(rename).sum())
    df["ConceptName"] = df["ConceptName"].map(lambda c: rename.get(c, c))
    say(f"[abstraction] Renamed {n_renamed:,} structural rows to canonical form.")

    # --- event harmonisation ---------------------------------------------
    # Same canonical event set as every other arm. This is what keeps the
    # std-vs-KB comparison about the ABSTRACTION layer: both arms carry the
    # identical complication events, and differ only in how the measurement
    # channels between them are represented.
    if include_kb_events:
        df = cohort.harmonise_events(df, patient_ids=df["PatientId"].unique(),
                                     verbose=verbose)
        say("[abstraction] Canonical events injected (STD_INCLUDE_KB_EVENTS is on).")
    else:
        # Knowledge-free arm: the events themselves are a knowledge-base
        # product, so strip them entirely rather than hand them over.
        all_spellings = {sp for ss in cohort.outcome_aliases.values() for sp in ss}
        n_events = int(df["ConceptName"].isin(all_spellings).sum())
        df = df[~df["ConceptName"].isin(all_spellings)].copy()
        say(f"[abstraction] Removed {n_events:,} outcome-event rows "
            f"(STD_INCLUDE_KB_EVENTS is off -- no knowledge base, no derived events).")

    # --- bin, collapse ----------------------------------------------------
    binned, stats = std_bin(df)
    say(f"[abstraction] Binned {len(stats)} numeric concepts into "
        f"{len(C.STD_BIN_LABELS)} sigma bins.")
    n_before = len(binned)
    std_df = collapse_intervals(binned)
    std_df = std_df.sort_values(["PatientId", "StartDateTime", "ConceptName"]) \
                   .reset_index(drop=True)
    say(f"[abstraction] Collapsed {n_before:,} -> {len(std_df):,} rows "
        f"({1 - len(std_df) / max(n_before, 1):.1%} reduction) at a "
        f"{C.STD_COLLAPSE_HOURS:.0f} h merge gap.")

    # --- TAK repo, then drop anything it still cannot resolve -------------
    tak_path = build_std_tak_repo(std_df["ConceptName"].unique())
    with open(tak_path, encoding="utf-8") as f:
        resolvable = set(json.load(f)["taks"].keys())
    unresolvable = set(std_df["ConceptName"].unique()) - resolvable
    if unresolvable:
        n_dropped = int(std_df["ConceptName"].isin(unresolvable).sum())
        std_df = std_df[~std_df["ConceptName"].isin(unresolvable)].copy()
        say(f"[abstraction] Dropped {n_dropped:,} rows on {len(unresolvable)} "
            f"unresolvable concepts.")

    say(f"[abstraction] std table: {len(std_df):,} rows, "
        f"{std_df['ConceptName'].nunique():,} concepts, "
        f"{std_df['PatientId'].nunique():,} patients.")
    return std_df, tak_path


def std_table_path() -> str:
    """Purpose: Canonical on-disk location of the cached std temporal table."""
    return os.path.join(paths.PROCESSED_DIR, "std_temporal_data.csv")


def ensure_std_temporal(cohort, raw=None, rebuild=False, verbose=True):
    """
    Purpose: Build the std arm's inputs once and reuse them across the K sweep.
    Method:  The table is K-independent (truncation happens later, inside
             `DataProcessor`), so it is cached on disk and rebuilt only on
             request.

    Args:
        cohort  (Cohort):            Shared cohort artefact.
        raw     (pd.DataFrame|None): Raw table with `hours`.
        rebuild (bool):              Force a rebuild.
        verbose (bool):              Print progress.

    Returns:
        tuple[str, str]: (std table path, augmented TAK repo path).
    """
    table_path = std_table_path()
    tak_path = os.path.join(paths.PROCESSED_DIR, "tak_repo_std.json")
    if os.path.exists(table_path) and os.path.exists(tak_path) and not rebuild:
        if verbose:
            print(f"[abstraction] Reusing {table_path}")
        return table_path, tak_path

    std_df, tak_path = build_std_temporal(cohort, raw=raw, verbose=verbose)
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    std_df.to_csv(table_path, index=False)
    if verbose:
        print(f"[abstraction] Wrote {table_path}")
    return table_path, tak_path
