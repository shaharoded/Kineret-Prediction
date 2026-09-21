"""
Per-hospital leave-one-out (LOO) support.

The random-patient split evaluates within-cohort generalization: the model
has seen every hospital during training. LOO removes one hospital at a time
from training and evaluates on that hospital, so the number reported is
"performance on a site the model has never seen." This is the transferability
question a multi-center reviewer will ask.

Where the hospital column comes from
------------------------------------

The canonical Kineret ETL does not fold the hospital identifier into
`context_data.csv`. The information lives in `visits_master.csv`, keyed on
`visit_id` (= our `PatientId` / admission id). See the sister notebook
`Mediator/run_mediator.ipynb` for how the JAMIA analysis performs the same
join. `visits_master.csv` also carries `person_id` and the admission
start/end datetimes; we only need `hospital` here.

Drop `visits_master.csv` next to the other four source files (or point at
it with `KINERET_VISITS_MASTER=/path/to/visits_master.csv`) and this
module does the rest. If the file is missing at LOO time, the error names
exactly where to put it.

Public surface
--------------

  * `load_visits_master()`                                -- read + minimal cleanup.
  * `attach_hospital_to_cohort(cohort)`                   -- add a `hospital`
    column to `cohort.context`; returns the modified cohort (in place).
  * `hospital_column_of(cohort)`                          -- return the column
    name (auto-attaches when missing).
  * `list_hospitals(cohort)`                              -- sorted list.
  * `filter_cohort_by_hospital(cohort, held_out_hospital)`-- shallow-copy the
    cohort with one hospital's admissions removed from every artifact.
  * `swap_default_cohort(new_cohort)`                     -- context manager
    that installs the filtered cohort as the on-disk default so each arm's
    `run()` picks it up transparently via `Cohort.load()`, restoring the
    canonical cohort on exit.
  * `loo_output_root(base, hospital)`                     -- path helper.

The 6-run driver lives in `notebooks/per_hospital_loo.ipynb`.
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
from typing import List, Optional

import pandas as pd

from kineret.config import paths


# ---------------------------------------------------------------------------
# Loading visits_master.csv
# ---------------------------------------------------------------------------

# Column aliases we accept in visits_master. Kept narrow -- the source column
# name in the Kineret ETL is `visit_id` (lowercase) and `hospital` (lowercase);
# aliases are here purely to catch obvious variants a curator might use.
_VISIT_ID_ALIASES = ("visit_id", "VisitId", "PatientId", "patient_id")
_HOSPITAL_ALIASES = ("hospital", "Hospital", "HOSPITAL",
                     "site", "Site", "care_site", "CareSite")


def _resolve(df: pd.DataFrame, aliases) -> str:
    """Purpose: Return the first alias that exists in df.columns."""
    for name in aliases:
        if name in df.columns:
            return name
    return None


def load_visits_master(path: Optional[str] = None) -> pd.DataFrame:
    """
    Purpose: Read `visits_master.csv` from `data/source/` (or the path given)
             and return a two-column DataFrame keyed on the admission id,
             carrying just the hospital label. Everything else is dropped --
             the Cohort's context table already supplies age, sex,
             comorbidities, and person_id.

    Method:  Locate the file via `paths.VISITS_MASTER_FILE` unless overridden.
             Standardize the visit-id column to `VisitId`, keep only that and
             `hospital`, and reduce to one row per VisitId (last by insertion
             order, matching the mediator notebook's dedup).

    Args:
        path (str|None): Override for the source path.

    Returns:
        pd.DataFrame: Columns `VisitId`, `hospital`. Deduplicated on VisitId.
    """
    src = path or paths.VISITS_MASTER_FILE
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"visits_master.csv not found at {src!r}.\n\n"
            "Per-hospital LOO needs the hospital identifier for every "
            "admission. The canonical Kineret ETL keeps this in "
            "`visits_master.csv` under the `hospital` column, keyed on "
            "`visit_id`. Copy it into `data/source/` alongside the other "
            "four source tables, or set the env var "
            "`KINERET_VISITS_MASTER=/absolute/path/to/visits_master.csv`.\n\n"
            "The mediator notebook (`Mediator/run_mediator.ipynb`) shows the "
            "same join pattern used by the JAMIA analysis."
        )

    df = pd.read_csv(src, low_memory=False)

    id_col = _resolve(df, _VISIT_ID_ALIASES)
    hosp_col = _resolve(df, _HOSPITAL_ALIASES)
    if id_col is None:
        raise KeyError(
            f"{os.path.basename(src)}: no visit-id column found; expected one "
            f"of {list(_VISIT_ID_ALIASES)!r} but got {list(df.columns)!r}."
        )
    if hosp_col is None:
        raise KeyError(
            f"{os.path.basename(src)}: no hospital column found; expected one "
            f"of {list(_HOSPITAL_ALIASES)!r} but got {list(df.columns)!r}."
        )

    out = (
        df[[id_col, hosp_col]]
        .rename(columns={id_col: "VisitId", hosp_col: "hospital"})
        .dropna(subset=["VisitId", "hospital"])
        .drop_duplicates(subset=["VisitId"], keep="last")
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# Cohort integration
# ---------------------------------------------------------------------------

def attach_hospital_to_cohort(cohort, path: Optional[str] = None,
                               strict: bool = False):
    """
    Purpose: Add a `hospital` column to `cohort.context` by joining
             `visits_master.csv` on the admission id, so the LOO helpers
             below can filter by it. The Cohort was pickled without this
             column because canonical training does not need it; the join
             is deferred to LOO time.
    Method:  Read visits_master (via `load_visits_master`), left-join on the
             cohort's context index (which is `PatientId` = admission id).
             `strict=True` raises if any patient is missing a hospital;
             otherwise those rows are dropped with a warning.

    Args:
        cohort         (Cohort): Shared cohort from `Cohort.load()`.
        path           (str|None): Override for visits_master.csv.
        strict         (bool): Fail on missing patients rather than warn.

    Returns:
        Cohort: The same cohort object with `context['hospital']` populated.
                Returned for chaining; the modification is in-place.
    """
    if "hospital" in cohort.context.columns:
        return cohort

    vm = load_visits_master(path=path).set_index("VisitId")

    ctx = cohort.context
    # `cohort.context` is indexed on PatientId (= admission id / visit id).
    joined = ctx.join(vm[["hospital"]], how="left")

    missing = joined["hospital"].isna()
    if missing.any():
        message = (
            f"attach_hospital_to_cohort: {int(missing.sum()):,} of "
            f"{len(joined):,} patients have no hospital in visits_master. "
            "Dropped." if not strict else
            f"attach_hospital_to_cohort: {int(missing.sum()):,} of "
            f"{len(joined):,} patients have no hospital in visits_master."
        )
        if strict:
            raise ValueError(message)
        import warnings
        warnings.warn(message)
        # Preserve everyone with a hospital; the LOO filter will only look at
        # patients whose hospital is known. Rows without a hospital are still
        # eligible for the train/val split via the canonical cohort, they
        # just cannot be a LOO test target.

    cohort.context = joined
    return cohort


def hospital_column_of(cohort, path: Optional[str] = None) -> str:
    """
    Purpose: Return the name of the hospital column on `cohort.context`,
             attaching it from visits_master.csv if it is not already there.
    Method:  Call `attach_hospital_to_cohort` when needed, then return
             `'hospital'` -- which is the canonical name post-attach.

    Args:
        cohort         (Cohort): Shared cohort.
        path           (str|None): Override for visits_master.csv.

    Returns:
        str: Always `'hospital'` on success.
    """
    if "hospital" not in cohort.context.columns:
        attach_hospital_to_cohort(cohort, path=path)
    return "hospital"


def list_hospitals(cohort, path: Optional[str] = None) -> List:
    """Purpose: Sorted list of the hospital identifiers seen in the cohort."""
    col = hospital_column_of(cohort, path=path)
    return sorted(cohort.context[col].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Filtering + on-disk swap
# ---------------------------------------------------------------------------

def filter_cohort_by_hospital(cohort, held_out_hospital,
                               hospital_col: Optional[str] = None,
                               visits_master_path: Optional[str] = None):
    """
    Purpose: Return a new Cohort with one hospital removed from every
             artifact, so the caller can save it as the default cohort and
             train each arm on the remaining sites.
    Method:  Ensure the hospital column is attached; identify the patients
             at the held-out hospital; shallow-copy the cohort and rebuild
             every downstream artifact (patients / events / context /
             qa_by_k) with those patients filtered out. The original is
             untouched.

    Args:
        cohort                 The shared cohort (`Cohort` instance).
        held_out_hospital      Value in the hospital column to exclude.
        hospital_col           Column name in `cohort.context`. Auto-detected
                               when None.
        visits_master_path     Override for visits_master.csv.

    Returns:
        Cohort: Shallow-copied cohort with the hospital removed. Its
                `meta` is stamped with `loo_hospital` and
                `loo_kept_hospitals` so cohort-drift checks treat it as
                distinct.
    """
    col = hospital_col or hospital_column_of(cohort, path=visits_master_path)

    context = cohort.context
    if col not in context.columns:
        # Defensive -- hospital_column_of already attaches, but if the caller
        # passed a column name that isn't there, fail legibly.
        raise KeyError(
            f"Column {col!r} not on cohort.context. Call "
            "`attach_hospital_to_cohort(cohort)` first, or leave "
            "`hospital_col` unset."
        )

    keep_mask = context[col] != held_out_hospital
    kept_pids = set(context.index[keep_mask].tolist())

    if not kept_pids:
        raise ValueError(
            f"Held-out hospital {held_out_hospital!r} covers the entire "
            "cohort -- nothing left to train on."
        )
    dropped_n = len(context) - keep_mask.sum()
    if dropped_n == 0:
        raise ValueError(
            f"Held-out hospital {held_out_hospital!r} matched zero rows. "
            f"Available hospitals: {list_hospitals(cohort)!r}"
        )

    new = copy.copy(cohort)                     # shallow -- replace filtered fields only
    new.patients = cohort.patients[
        cohort.patients["PatientId"].isin(kept_pids)
    ].reset_index(drop=True)
    new.events = cohort.events[
        cohort.events["PatientId"].isin(kept_pids)
    ].reset_index(drop=True)
    new.context = cohort.context[keep_mask].copy()
    new.qa_by_k = {
        k: df[df.index.isin(kept_pids)].copy() if df is not None else None
        for k, df in cohort.qa_by_k.items()
    }
    new.meta = dict(cohort.meta)
    new.meta["loo_hospital"] = held_out_hospital
    new.meta["loo_kept_hospitals"] = [
        h for h in cohort.context[col].dropna().unique()
        if h != held_out_hospital
    ]
    return new


@contextlib.contextmanager
def swap_default_cohort(new_cohort, backup_suffix: str = ".loo_backup"):
    """
    Purpose: Temporarily install `new_cohort` as the on-disk default cohort
             so every arm's `run()` -- which calls `Cohort.load()` --
             transparently picks it up. Restore the canonical cohort on
             exit, including on error.
    Method:  Rename the canonical `cohort.pkl` aside, save the filtered
             cohort in its place, and swap back in the `finally` clause.

    Args:
        new_cohort      (Cohort): The filtered cohort to install.
        backup_suffix   (str):    Suffix for the canonical pickle while the
                                  filtered cohort is active.

    Yields:
        str: Path to the (now-backed-up) canonical cohort pickle.
    """
    default_path = getattr(paths, "COHORT_PKL",
                           os.path.join(paths.PROCESSED_DIR, "cohort.pkl"))
    backup_path = default_path + backup_suffix

    if os.path.exists(default_path):
        shutil.move(default_path, backup_path)
    try:
        new_cohort.save(default_path)
        yield default_path
    finally:
        if os.path.exists(default_path):
            os.remove(default_path)
        if os.path.exists(backup_path):
            shutil.move(backup_path, default_path)


def loo_output_root(base_root: str, held_out_hospital) -> str:
    """Purpose: `outputs/loo/<sanitized_hospital>/` for one LOO iteration."""
    tag = str(held_out_hospital).replace(os.sep, "_").replace(" ", "_")
    return os.path.join(base_root, "loo", tag)
