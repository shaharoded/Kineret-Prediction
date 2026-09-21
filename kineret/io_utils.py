"""
Source-table loading helpers.

The Kineret drop is a handful of excels; the reference pipelines were written
against csvs. Everything funnels through `load_table` so neither format leaks
into downstream code.
"""

import os
import re
import pandas as pd


def load_table(path: str, **read_kwargs) -> pd.DataFrame:
    """
    Purpose: Read one source table regardless of whether it is an excel or a csv.
    Method:  Dispatch on extension; excels go through openpyxl, csvs through
             `read_csv(low_memory=False)` so mixed-dtype Value columns do not
             trigger chunked type inference.

    Args:
        path        (str): Absolute path to the table.
        read_kwargs      : Forwarded to the pandas reader.

    Returns:
        pd.DataFrame: The loaded table.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[io_utils] Source table not found: {path}\n"
            f"Drop it into data/source/ or point the matching KINERET_* env var at it."
        )
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, **read_kwargs)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, low_memory=False, **read_kwargs)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"[io_utils] Unsupported source table extension: {ext} ({path})")


def normalise_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Coerce any temporal table to the canonical
             ['PatientId', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value'] shape.
    Method:  Rename the known column aliases (Mediator emits StartTime/EndTime,
             the ETL emits StartDateTime/EndDateTime), parse both datetime
             columns to naive UTC, and fill a missing EndDateTime with the
             start (an instantaneous event).

    Args:
        df (pd.DataFrame): Raw temporal or abstraction table.

    Returns:
        pd.DataFrame: Canonicalised copy.
    """
    df = df.rename(columns={
        "StartTime": "StartDateTime", "EndTime": "EndDateTime",
        "start_time": "StartDateTime", "end_time": "EndDateTime",
        "concept_name": "ConceptName", "value": "Value",
    }).copy()

    # --- one row, two identities ----------------------------------------
    # `mediator_input.csv` carries BOTH: `PatientId` is the person and
    # `VisitId` is the admission. `mediator_output.csv` carries only the
    # admission, under the name `PatientId`, because the Mediator is allowed
    # exactly one id column.
    #
    # The study's unit is the ADMISSION, so VisitId wins wherever it exists and
    # the person is preserved under `PersonId`. Without this the two files are
    # keyed on different things and share almost no ids -- which is what
    # collapsed the cohort to 84 patients.
    lower = {str(c).lower(): c for c in df.columns}
    visit_column = lower.get("visitid") or lower.get("visit_id")
    if visit_column:
        if "PatientId" in df.columns:
            df = df.rename(columns={"PatientId": "PersonId"})
        elif lower.get("patient_id"):
            df = df.rename(columns={lower["patient_id"]: "PersonId"})
        df = df.rename(columns={visit_column: "PatientId"})
    elif "PatientId" not in df.columns and lower.get("patient_id"):
        df = df.rename(columns={lower["patient_id"]: "PatientId"})

    missing = {"PatientId", "ConceptName", "StartDateTime"} - set(df.columns)
    if missing:
        raise ValueError(f"[io_utils] Temporal table missing required columns: {sorted(missing)}")

    if "Value" not in df.columns:
        df["Value"] = "True"
    if "EndDateTime" not in df.columns:
        df["EndDateTime"] = df["StartDateTime"]

    for col in ("StartDateTime", "EndDateTime"):
        parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
        df[col] = parsed.dt.tz_convert(None)
    df["EndDateTime"] = df["EndDateTime"].fillna(df["StartDateTime"])

    n_bad = int(df["StartDateTime"].isna().sum())
    if n_bad:
        print(f"[io_utils] Dropping {n_bad} rows with an unparseable StartDateTime.")
        df = df[df["StartDateTime"].notna()].copy()

    df["ConceptName"] = df["ConceptName"].astype(str).str.strip()

    # The admission window the export already knows, parsed the same way as the
    # event stamps so it can anchor the clock and give a true length of stay.
    for col in ("AdmissionStart", "AdmissionEnd"):
        source = lower.get(col.lower())
        if source and source in df.columns:
            parsed = pd.to_datetime(df[source], utc=True, errors="coerce")
            df[col] = parsed.dt.tz_convert(None)
    return df


def compile_alias(pattern: str) -> re.Pattern:
    """Purpose: Case-insensitive fullmatch matcher for one outcome/structural alias."""
    return re.compile(rf"^(?:{pattern})$", flags=re.IGNORECASE)


def match_concepts(concepts, pattern: str):
    """
    Purpose: List every ConceptName in the data that a canonical alias regex covers.
    Method:  Fullmatch, case-insensitive, sorted for determinism.

    Args:
        concepts (Iterable[str]): Observed ConceptName values.
        pattern  (str):           Regex body from OUTCOME_ALIAS_REGEX.

    Returns:
        list[str]: Matching concept names.
    """
    rx = compile_alias(pattern)
    return sorted({c for c in concepts if rx.match(str(c))})


def to_numeric_value(series: pd.Series) -> pd.Series:
    """
    Purpose: Turn a mixed Value column into floats where possible.
    Method:  Numeric coercion first; anything left over that reads as a boolean
             string becomes 1.0/0.0. Genuinely categorical values stay NaN so the
             caller can decide to expand them into indicator variables.

    Args:
        series (pd.Series): Raw Value column.

    Returns:
        pd.Series: Float series with NaN at categorical positions.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    boolish = series.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    return numeric.fillna(boolish)
