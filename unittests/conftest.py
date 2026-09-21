"""
Shared fixtures: a small synthetic cohort, built once per test session.

The tests never touch the real Kineret data. They generate a synthetic drop
with the same shape (including the raw-vs-Mediator naming split), point the
path resolver at it via the `KINERET_*` environment overrides, and build the
cohort from that.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    """
    Purpose: Write a synthetic source drop and point the path resolver at it.
    Method:  Generate the four tables, write them as csvs into a temp dir, set
             the `KINERET_*` environment overrides, then reload
             `kineret.config.paths` so the module-level constants pick them up.

    Returns:
        str: The directory holding the synthetic tables.
    """
    import importlib

    from scripts.make_synthetic_data import generate

    out = tmp_path_factory.mktemp("kineret_synth")
    raw, abstract, context, qa = generate(n_patients=80, seed=11,
                                          min_days=8.0, max_days=16.0)
    files = {"RAW_TEMPORAL": ("mediator_input.csv", raw),
             "ABSTRACT": ("mediator_output.csv", abstract),
             "CONTEXT": ("context_data.csv", context),
             "QA": ("qa_scores.csv", qa)}
    for key, (name, df) in files.items():
        path = out / name
        df.to_csv(path, index=False)
        os.environ[f"KINERET_{key}"] = str(path)

    from kineret.config import paths
    importlib.reload(paths)
    return str(out)


@pytest.fixture(scope="session")
def cohort(synth_dir, tmp_path_factory):
    """
    Purpose: A built `Cohort` over the synthetic drop.
    Method:  Redirect the processed/checkpoint/output directories into a temp
             tree so a test run never writes into the working repo.

    Returns:
        Cohort: The built artefact.
    """
    from kineret.cohort import build_cohort
    from kineret.config import data_config
    from kineret.config import paths

    # The synthetic generator stamps admissions across a recent 12-month span,
    # which will not sit inside the real study window. Widen it for the tests so
    # they exercise the range machinery without depending on today's date.
    data_config.DATE_RANGE_START = "2000-01-01"
    data_config.DATE_RANGE_END = "2099-12-31"

    work = tmp_path_factory.mktemp("kineret_work")
    paths.PROCESSED_DIR = str(work / "processed")
    paths.CHECKPOINT_DIR = str(work / "checkpoints")
    paths.OUTPUT_DIR = str(work / "outputs")
    paths.COHORT_PKL = os.path.join(paths.PROCESSED_DIR, "cohort.pkl")
    paths.ensure_dirs()
    return build_cohort(verbose=False)
