"""
Filesystem contract for the Kineret benchmark.

Every path the pipeline touches is resolved here so a VM deployment only has to
drop files into ``data/source/`` and, if the filenames differ, override the four
``*_FILE`` constants (or set the matching ``KINERET_*`` environment variable).
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
SOURCE_DIR     = os.path.join(DATA_DIR, "source")       # user-supplied excels / csvs
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")    # cohort.pkl + per-K strats pickles
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs")
# Every figure the notebook draws is also written here as a PNG, so plots can be
# exported without re-running the notebook.
FIGURE_DIR     = os.path.join(OUTPUT_DIR, "figures")

# The concept hierarchy INTERVenE-Enc resolves every abstraction through. Ships
# inside the package, not in `data/source/` -- it is knowledge, not cohort data.
TAK_REPO_PATH = os.path.join(PROJECT_ROOT, "kineret", "config", "tak_repo_portable.json")


def _src(env_key: str, stem: str) -> str:
    """
    Purpose: Resolve one source table to a concrete path.
    Method:  Environment override first (``KINERET_<KEY>``), then `<stem>.csv`
             under ``data/source/``, then `<stem>.xlsx` for the same stem. If
             neither exists the canonical `.csv` path is returned, so the
             not-found error names the file the pipeline actually expects rather
             than a generic message.

             Exactly one stem per table, deliberately: accepting a family of
             aliases means a mis-named or stale file can be picked up silently,
             which is the opposite of frictionless.

    Args:
        env_key (str): Suffix of the ``KINERET_<KEY>`` environment override.
        stem    (str): Canonical basename without extension.

    Returns:
        str: Absolute path to the source table.
    """
    override = os.environ.get(f"KINERET_{env_key}")
    if override:
        return os.path.abspath(override)
    for extension in (".csv", ".xlsx"):
        candidate = os.path.join(SOURCE_DIR, stem + extension)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(SOURCE_DIR, stem + ".csv")


# --- The four source tables -------------------------------------------------
# Canonical filenames, dropped into `data/source/`. `.csv` is the expected
# form; the same stem with `.xlsx` is accepted for each.
#
#   mediator_input   Mediator *input* -- one row per raw measurement,
#                    administration or structural event. Consumed by LogReg,
#                    ss-STraTS and the sigma-bin arm.
#   mediator_output  Mediator *output* -- temporal abstractions plus the derived
#                    `*_EVENT` rows. Consumed by INTERVenE-Enc's KB arms, and it
#                    is the source of truth for outcome labels across ALL arms.
#   context_data     One row per admission, static numeric features. The QA
#                    block is concatenated onto this at run time, per sample.
#   qa_scores        Mediator QA output -- temporal-shaped compliance scores,
#                    aggregated into the context vector for the QA arms.
RAW_TEMPORAL_FILE = _src("RAW_TEMPORAL", "mediator_input")
ABSTRACT_FILE     = _src("ABSTRACT", "mediator_output")
CONTEXT_FILE      = _src("CONTEXT", "context_data")
QA_FILE           = _src("QA", "qa_scores")

# Optional -- not part of the training pipeline, only consumed by
# `kineret/loo.py` for the per-hospital leave-one-out notebook. Carries the
# `hospital` and `person_id` columns keyed by `visit_id` (= our PatientId /
# admission id). Absence is fine for every canonical run; LOO fails with a
# legible message if the file is missing.
VISITS_MASTER_FILE = _src("VISITS_MASTER", "visits_master")

# Canonical stems, for error messages and the packaging check.
SOURCE_STEMS = {
    "RAW_TEMPORAL": "mediator_input",
    "ABSTRACT": "mediator_output",
    "CONTEXT": "context_data",
    "QA": "qa_scores",
    # Optional -- LOO-only, canonical runs never open this.
    "VISITS_MASTER": "visits_master",
}

COHORT_PKL = os.path.join(PROCESSED_DIR, "cohort.pkl")


def describe_sources() -> str:
    """
    Purpose: Report where each source table resolved, and whether it is there.
    Method:  One line per table with a present/missing marker, so a run that is
             about to fail on a missing file says which one before it starts.

    Returns:
        str: Multi-line report.
    """
    rows = []
    for key, path in (("mediator_input", RAW_TEMPORAL_FILE),
                      ("mediator_output", ABSTRACT_FILE),
                      ("context_data", CONTEXT_FILE),
                      ("qa_scores", QA_FILE)):
        mark = "ok     " if os.path.exists(path) else "MISSING"
        rows.append(f"  [{mark}] {key:<16} {path}")
    rows.append(f"  [{'ok     ' if os.path.exists(TAK_REPO_PATH) else 'MISSING'}] "
                f"{'tak_repo':<16} {TAK_REPO_PATH}")
    return "\n".join(rows)


def ensure_dirs():
    """Purpose: Create every runtime directory the pipeline writes into."""
    for d in (DATA_DIR, SOURCE_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
              FIGURE_DIR):
        os.makedirs(d, exist_ok=True)
