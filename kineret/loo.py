"""
Per-hospital leave-one-out (LOO) support.

The random-patient split evaluates within-cohort generalization: the model
has seen every hospital during training. LOO removes one hospital at a time
from training and evaluates on that hospital, so the number reported is
"performance on a site the model has never seen." This is the transferability
question a multi-center reviewer will ask.

This module provides the plumbing to run LOO without changing cohort
assembly:

  * `hospital_column_of(cohort)` -- locate the hospital identifier in the
    shared context table, or raise a legible error if the ETL export did
    not carry it forward.
  * `filter_cohort_by_hospital(cohort, held_out_hospital)` -- return a NEW
    Cohort with the held-out hospital removed from every artifact
    (patients, events, context, qa_by_k). The original is unmodified.
  * `swap_default_cohort(new_cohort)` -- context manager that saves the
    filtered cohort to the default cohort path so each arm's `run()`
    picks it up transparently via `Cohort.load()`, restoring the
    canonical cohort on exit.

The 6-run driver lives in `notebooks/per_hospital_loo.ipynb`.
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
from typing import Iterable, List, Optional

import pandas as pd

from kineret.config import paths


# Column names we look for in `cohort.context` to identify the hospital.
# The Kineret ETL export is expected to carry one of these; the LOO
# driver raises with a legible error if none is present.
_HOSPITAL_ALIASES = (
    "hospital", "Hospital", "HOSPITAL",
    "site", "Site", "SITE",
    "care_site", "CareSite", "CARE_SITE",
    "hospital_id", "site_id",
)


def hospital_column_of(cohort) -> str:
    """
    Purpose: Locate the hospital-identifier column in a shared cohort's
             context table, or raise a legible error naming what the ETL
             export needs to carry for LOO to work.
    Method:  Walk a small alias list; return the first match.

    Args:
        cohort (Cohort): The shared cohort from `Cohort.load()`.

    Returns:
        str: The name of the hospital column in `cohort.context`.

    Raises:
        KeyError: If no known hospital column is present. The message
                  spells out how to add it via the ETL step.
    """
    ctx = cohort.context
    for name in _HOSPITAL_ALIASES:
        if name in ctx.columns:
            return name
    raise KeyError(
        "The shared cohort's `context` table does not carry a hospital "
        "identifier. Per-hospital LOO needs one of "
        f"{list(_HOSPITAL_ALIASES)!r} to be present.\n\n"
        "The Kineret raw extract knows which hospital each admission "
        "came from (it is stored per source table in the OMOP export). "
        "Add the mapping into `context_data.csv` as a `hospital` column "
        "in the ETL step (`Kineret-ETL/db_load.ipynb`) and re-run the "
        "cohort build. The column values are opaque -- any consistent "
        "identifier per hospital works, integer or string."
    )


def list_hospitals(cohort) -> List:
    """Purpose: Sorted list of the hospital identifiers seen in the cohort."""
    col = hospital_column_of(cohort)
    return sorted(cohort.context[col].dropna().unique().tolist())


def filter_cohort_by_hospital(cohort, held_out_hospital,
                              hospital_col: Optional[str] = None):
    """
    Purpose: Return a new Cohort with one hospital removed from every
             artifact, so the caller can save it as the default cohort and
             train each arm on the remaining sites.
    Method:  Left-join the hospital column onto the patients frame so each
             admission carries its hospital; select the complement of
             `held_out_hospital`; filter every downstream table by
             PatientId. `qa_by_k` is a dict of DataFrames -- each is
             filtered by index.

    Args:
        cohort              The shared cohort (`Cohort` instance).
        held_out_hospital   Value in the hospital column to exclude.
        hospital_col        Column name in `cohort.context`. Auto-detected
                            when None.

    Returns:
        Cohort: A shallow-copied cohort with the hospital removed from
                every relevant artifact. The original is untouched.
    """
    col = hospital_col or hospital_column_of(cohort)

    context = cohort.context
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
            "Check the value against `list_hospitals(cohort)`."
        )

    new = copy.copy(cohort)   # shallow -- we replace only the filtered fields
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
    # Keep meta but stamp the LOO decision so downstream `cohort_drift`
    # comparisons treat the filtered cohort as distinct.
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
    Purpose: Temporarily install `new_cohort` as the on-disk default
             cohort so every arm's `run()` -- which calls
             `Cohort.load()` -- transparently picks it up. Restore the
             canonical cohort on exit.
    Method:  Move the canonical `cohort.pkl` aside, save the filtered
             cohort in its place, and swap back in the `finally` clause.

    Args:
        new_cohort      (Cohort): The filtered cohort to install.
        backup_suffix   (str):    Suffix appended to the canonical pickle
                                  while the filtered cohort is active.

    Yields:
        str: Path to the canonical (now backed-up) cohort pickle. Useful
             for logging.
    """
    default_path = paths.COHORT_CACHE_PATH if hasattr(paths, "COHORT_CACHE_PATH") \
        else os.path.join(paths.PROCESSED_DIR, "cohort.pkl")
    backup_path = default_path + backup_suffix

    if os.path.exists(default_path):
        shutil.move(default_path, backup_path)
    try:
        new_cohort.save(default_path)
        yield default_path
    finally:
        # Always restore, even on error.
        if os.path.exists(default_path):
            os.remove(default_path)
        if os.path.exists(backup_path):
            shutil.move(backup_path, default_path)


def loo_output_root(base_root: str, held_out_hospital) -> str:
    """Purpose: `outputs/loo/<sanitized_hospital>/` for one LOO iteration."""
    tag = str(held_out_hospital).replace(os.sep, "_").replace(" ", "_")
    return os.path.join(base_root, "loo", tag)
