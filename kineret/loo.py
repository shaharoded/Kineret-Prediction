"""Hospital metadata and the person-disjoint LOO split used by the main notebook.

Full orchestration lives in kineret.experiments.workflow.hospital_loo. The
held-out hospital remains in test; it is never deleted or added to model context.
"""
import os
import pandas as pd
from typing import Optional
from kineret.config import paths

_VISIT_ID_ALIASES = ("visit_id", "VisitId", "PatientId", "patient_id")
_HOSPITAL_ALIASES = ("hospital", "Hospital", "HOSPITAL", "site", "Site", "care_site", "CareSite")


def _resolve(df, aliases):
    return next((name for name in aliases if name in df), None)


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

    assignments = df[[id_col, hosp_col]].dropna().drop_duplicates()
    conflicts = assignments.groupby(id_col)[hosp_col].nunique()
    if (conflicts > 1).any():
        raise ValueError("Admissions have conflicting hospital assignments in visits_master.")
    out = (
        df[[id_col, hosp_col]]
        .rename(columns={id_col: "VisitId", hosp_col: "hospital"})
        .dropna(subset=["VisitId", "hospital"])
        .drop_duplicates(subset=["VisitId"], keep="last")
        .reset_index(drop=True)
    )
    return out



def filter_cohort_by_hospital(cohort, held_out_hospital, hospital_col=None, visits_master_path=None):
    """Compatibility name: retain held-out admissions as test with disjoint people."""
    from kineret.experiments.cohorts import site_split
    import copy
    clean = copy.copy(cohort)
    clean.context = cohort.context.drop(columns=[hospital_col or "hospital"], errors="ignore").copy()
    return site_split(clean, held_out_hospital, visits_master_path or paths.VISITS_MASTER_FILE)
