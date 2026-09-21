"""
Notebook-facing benchmark API.

`notebooks/benchmark.ipynb` is the entry point for the whole study, and it
should read like the study: smoke-test the wiring, decide the context window,
run the grid, read the table. This module holds the machinery so those cells
stay one line each.

    from kineret.benchmark import smoke_test, run_grid, comparison_table

    smoke_test()                                  # wiring check, ~minutes
    run_grid(models=["logreg", "strats", "intervene_enc"], k_values=[4])
    comparison_table()                            # the clean 3-model x QA table
"""

import gc
import json
import os
import shutil
import time
import traceback

import numpy as np
import pandas as pd

from kineret.cohort import Cohort
from kineret.config import data_config as C
from kineret.config import paths
from kineret.evaluation import METRIC_NAMES, _is_scratch, collect_runs
from kineret.io_utils import load_table, normalise_temporal

# The study's arms, in ladder order. This is an explicit list, not a cartesian
# product: there is no std+QA arm, because the knowledge-free abstraction has no
# pattern tokens to pair the QA block with, and pairing them would confound the
# two things the ladder is trying to separate.
#
# Each entry is (arm_key, model_key, use_qa, label).
ARMS = [
    ("logreg",            "logreg",            False, "LogReg (raw)"),
    ("logreg_qa",         "logreg",            True,  "LogReg (raw + QA)"),
    ("strats",            "strats",            False, "ss-STraTS (raw)"),
    ("strats_qa",         "strats",            True,  "ss-STraTS (raw + QA)"),
    ("intervene_std",     "intervene_enc_std", False, "INTERVenE-Enc (σ-bins)"),
    ("intervene_kb",      "intervene_enc",     False, "INTERVenE-Enc (KB)"),
    ("intervene_kb_qa",   "intervene_enc",     True,  "INTERVenE-Enc (KB + QA)"),
]

# arm_key -> (model_key, use_qa) and arm_key -> display label.
ARM_SPEC = {key: (model, qa) for key, model, qa, _label in ARMS}
ARM_LABELS = {key: label for key, _model, _qa, label in ARMS}
ARM_ORDER = [key for key, *_ in ARMS]

# Which interval source each INTERVenE model key feeds the encoder. The other
# models read raw measurements and have no abstraction layer at all.
MODEL_ABSTRACTION = {"intervene_enc": "kb", "intervene_enc_std": "std"}

# Kept for the run artefacts, which are keyed by the underlying model.
MODEL_LABELS = {
    "logreg": "LogReg",
    "strats": "ss-STraTS",
    "intervene_enc_std": "INTERVenE-Enc (σ-bins)",
    "intervene_enc": "INTERVenE-Enc (KB)",
}

DEFAULT_ARMS = list(ARM_ORDER)

# Hyperparameter overrides that make each model run in seconds rather than
# hours. Used ONLY by `smoke_test` -- these settings prove the wiring, they
# prove nothing about accuracy.
SMOKE_OVERRIDES = {
    "logreg": {"max_epochs": 3, "patience": 2},
    "strats": {"max_epochs": 1, "patience": 1, "hid_dim": 16,
               "num_heads": 2, "num_layers": 1},
    "intervene_enc": {
        "model": {"embed_dim": 32, "n_head": 2, "n_layer": 1, "time2vec_dim": 8},
        "training": {"phase1_n_epochs": 1, "phase2_n_epochs": 1, "phase3_n_epochs": 1,
                     "early-stop-patience": 1, "batch_size": 8},
    },
}
SMOKE_OVERRIDES["intervene_enc_std"] = SMOKE_OVERRIDES["intervene_enc"]


def arm_label(arm_key: str) -> str:
    """Purpose: The display name for one arm."""
    return ARM_LABELS.get(arm_key, arm_key)


def resolve_arms(arms=None) -> list:
    """
    Purpose: Normalise an arm selection into (arm_key, model_key, use_qa) triples.
    Method:  Accepts arm keys, or None for the full ladder. Raises on an unknown
             key rather than silently skipping it -- a typo in the notebook would
             otherwise look like a deliberately shorter study.

    Args:
        arms (list|None): Arm keys, or None for all.

    Returns:
        list[tuple[str, str, bool]]: (arm_key, model_key, use_qa).
    """
    keys = list(arms) if arms is not None else list(ARM_ORDER)
    unknown = [k for k in keys if k not in ARM_SPEC]
    if unknown:
        raise ValueError(f"Unknown arm(s): {unknown}. Valid arms: {ARM_ORDER}")
    return [(k, *ARM_SPEC[k]) for k in keys]


# ---------------------------------------------------------------------------
# Device and memory
# ---------------------------------------------------------------------------

def resolve_device(device=None, verbose=True):
    """
    Purpose: Pick the device every model in this session will train on.
    Method:  Honour an explicit request; otherwise CUDA when available. Reports
             what it picked and, on CPU, says so loudly -- a silent fallback to
             CPU on a GPU box usually means a broken install, and the difference
             is hours per cell.

    Args:
        device  (str|None): Explicit torch device string.
        verbose (bool):     Print the resolved device.

    Returns:
        str: Torch device string.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.cuda.current_device()
            name = torch.cuda.get_device_name(index)
            total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
            print(f"[device] {device} -- {name} ({total:.1f} GiB)")
        else:
            print(f"[device] {device}"
                  + ("" if not torch.cuda.is_available() else
                     "  (CUDA is available but was not selected)"))
            if not torch.cuda.is_available():
                print("[device] WARNING: training on CPU. On a GPU box this "
                      "usually means a CPU-only torch build -- check "
                      "`torch.__version__` and reinstall with the CUDA wheel.")
    return str(device)


def free_memory(verbose=False):
    """
    Purpose: Return a finished cell's memory before the next one starts.
    Method:  Python garbage collection, then release the CUDA caching
             allocator's unused blocks.

             The grid trains a model, scores it, writes the results to disk and
             then has no further use for it -- results live in `outputs/`, not in
             RAM. Without this the whole sweep's models and their processed
             dataframes accumulate, and the last cells OOM on exactly the runs
             you most wanted.

    Args:
        verbose (bool): Print the reclaimed GPU memory.
    """
    # The harmonised raw table is several GB of object-dtype strings and is
    # needed only by LogReg, ss-STraTS and the sigma-bin preprocessor. Held
    # across the whole ladder it sat in RAM through all three INTERVenE arms,
    # which never touch it. It is a cache: dropping it costs one re-read the
    # next time an arm asks, and buys that memory back for the arm running now.
    _RAW_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved() / 1024 ** 3
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if verbose:
                after = torch.cuda.memory_reserved() / 1024 ** 3
                print(f"[memory] GPU reserved {before:.2f} -> {after:.2f} GiB")
    except ImportError:
        pass


def clear_caches():
    """
    Purpose: Drop the session-level dataframe caches.
    Method:  The raw temporal table is cached to avoid re-reading it per cell,
             which is the right trade while the sweep runs and the wrong one
             once it is done. Call this between phases of a long notebook.
    """
    _RAW_CACHE.clear()
    free_memory()


# ---------------------------------------------------------------------------
# Raw-table cache
# ---------------------------------------------------------------------------

_RAW_CACHE = {}


def raw_with_hours(cohort: Cohort) -> pd.DataFrame:
    """
    Purpose: The raw temporal table, harmonised and anchored, cached per session.
    Method:  Read once, put onto the canonical event support, then anchored to
             each patient's admission. It is the largest read in the pipeline and
             both LogReg and the STraTS preprocessor want it, so doing the work
             once here keeps it off the per-cell path.

    Args:
        cohort (Cohort): Shared cohort artefact (supplies the admission anchors).

    Returns:
        pd.DataFrame: Raw rows for cohort patients with an `hours` column.
    """
    # Keyed on the COHORT, not just the file: the cohort supplies the patient
    # set, the canonical events and the KB-containment list, so a rebuilt cohort
    # must not be served the previous one's harmonised table.
    key = (paths.RAW_TEMPORAL_FILE, len(cohort.patients), len(cohort.events),
           len(cohort.kb_event_names), bool(C.EVENTS_AS_INPUTS),
           bool(getattr(C, "KB_EVENTS_FOR_KB_ARMS_ONLY", True)))
    if key not in _RAW_CACHE:
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        raw = raw[raw["PatientId"].isin(set(cohort.patients["PatientId"]))].copy()
        # kb_events=False: this stream feeds LogReg, ss-STraTS and the sigma-bin
        # arm, none of which may see the KB's own conclusions.
        raw = cohort.harmonise_events(raw, patient_ids=cohort.patients["PatientId"],
                                      kb_events=False, verbose=True)
        admission = cohort.patients.set_index("PatientId")["admission_time"]
        raw["hours"] = ((raw["StartDateTime"] - raw["PatientId"].map(admission))
                        .dt.total_seconds() / 3600.0)
        raw.attrs["kineret_events_harmonised"] = False
        _RAW_CACHE.clear()          # one cohort in flight at a time; bound the memory
        _RAW_CACHE[key] = raw
    return _RAW_CACHE[key]


# ---------------------------------------------------------------------------
# Running one cell
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Artefact helpers (figures live in kineret.figures)
# ---------------------------------------------------------------------------

def save_figure(fig, name: str, subdir: str = None, dpi: int = 200,
                also_pdf: bool = True, verbose: bool = True) -> str:
    """
    Purpose: Persist a figure so it can go into the paper without re-running.
    Method:  Writes `outputs/figures/<name>.png` at print resolution and a
             vector `.pdf` beside it, and returns the PNG path. The figure is
             left open so a notebook still renders it inline.

    Args:
        fig      (matplotlib.figure.Figure): Figure to write.
        name     (str):       Basename, no extension.
        subdir   (str|None):  Optional subdirectory under `outputs/figures/`.
        dpi      (int):       Raster resolution.
        also_pdf (bool):      Also write a vector PDF.
        verbose  (bool):      Print the path.

    Returns:
        str: Path to the written PNG.
    """
    directory = paths.FIGURE_DIR if subdir is None else os.path.join(paths.FIGURE_DIR, subdir)
    os.makedirs(directory, exist_ok=True)
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name)
    png = os.path.join(directory, f"{safe}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.join(directory, f"{safe}.pdf"), bbox_inches="tight")
    if verbose:
        print(f"[figure] {png}")
    return png


def save_table(df: pd.DataFrame, name: str, verbose: bool = True) -> str:
    """
    Purpose: Persist a reported table next to the figures.
    Method:  Writes `outputs/figures/<name>.csv`, so an exported figure and the
             numbers behind it travel together.

    Args:
        df      (pd.DataFrame): Table to write.
        name    (str):          Basename, no extension.
        verbose (bool):         Print the path.

    Returns:
        str: Path written.
    """
    os.makedirs(paths.FIGURE_DIR, exist_ok=True)
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name)
    path = os.path.join(paths.FIGURE_DIR, f"{safe}.csv")
    df.to_csv(path, index=False)
    if verbose:
        print(f"[table]  {path}")
    return path


def run_one(arm, cohort=None, output_root=None, device=None, seed=C.SEED,
            bootstrap_resamples=2000, overrides=None, checkpoint_root=None,
            resume=True):
    """
    Purpose: Train and score a single arm of the ladder.
    Method:  Resolve the arm to (model, QA flag) and route to that model's own
             trainer. Each owns its preprocessing, checkpointing and scoring;
             all of them write the same artefacts.

    Args:
        arm                 (str):    Arm key from `ARMS`.
        cohort              (Cohort|None): Shared cohort; loaded when None.
        output_root         (str|None): Defaults to `outputs/`.
        device              (str|None): Torch device string.
        seed                (int):    RNG seed.
        bootstrap_resamples (int):    Resamples for the intervals.
        overrides           (dict|None): Hyperparameter overrides.
        checkpoint_root     (str|None): Checkpoint root (INTERVenE-Enc only).
        resume              (bool):   Resume from existing checkpoints.

    Returns:
        dict: {'output_dir': str, 'scores': dict}
    """
    if arm not in ARM_SPEC:
        raise ValueError(f"Unknown arm {arm!r}. Valid arms: {ARM_ORDER}")
    model, use_qa = ARM_SPEC[arm]
    cohort = cohort or Cohort.load()

    if model == "logreg":
        from kineret.logreg.train import run
        return run(use_qa=use_qa, cohort=cohort, raw=raw_with_hours(cohort),
                   output_root=output_root, device=device, seed=seed,
                   bootstrap_resamples=bootstrap_resamples, overrides=overrides)

    if model == "strats":
        from kineret.strats.train import run
        return run(use_qa=use_qa, output_root=output_root, device=device,
                   seed=seed, overrides=overrides,
                   bootstrap_resamples=bootstrap_resamples)

    if model in MODEL_ABSTRACTION:
        from kineret.intervene.train import run
        overrides = overrides or {}
        return run(use_qa=use_qa, cohort=cohort, output_root=output_root,
                   seed=seed, resume=resume,
                   bootstrap_resamples=bootstrap_resamples,
                   model_overrides=overrides.get("model"),
                   training_overrides=overrides.get("training"),
                   checkpoint_root=checkpoint_root, device=device,
                   abstraction=MODEL_ABSTRACTION[model])

    raise ValueError(f"Arm {arm!r} maps to unknown model {model!r}.")


def run_dir_for(arm: str, output_root=None) -> str:
    """Purpose: Where one arm's artefacts live."""
    model, use_qa = ARM_SPEC[arm]
    root = output_root or paths.OUTPUT_DIR
    return os.path.join(root, model, "qa" if use_qa else "noqa")


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

# Config keys a cached cohort is keyed on. If any of these changed since the
# pickle was written, the cohort on disk answers a different study question than
# the one now configured -- most importantly `eval_context_days`, which section 2
# of the notebook exists to revise.
COHORT_CONFIG_KEYS = (
    "horizon_end_days", "train_context_days", "eval_context_days", "split_seed",
    "outcome_support_threshold", "targets", "date_range_start", "date_range_end",
    "date_range_require_full_horizon", "event_source", "event_match_tolerance_h",
    "events_as_inputs", "raw_temporal_file", "abstract_file", "context_file",
    "kb_events_for_kb_arms_only", "event_align_across_files",
    "event_align_tolerance_min", "split_group_by_person",
)


def _current_cohort_config() -> dict:
    """Purpose: The cohort-defining config as it stands right now."""
    return {
        "horizon_end_days": float(C.HORIZON_END_DAYS),
        "train_context_days": list(C.TRAIN_CONTEXT_DAYS),
        "eval_context_days": C.EVAL_CONTEXT_DAYS,
        "split_seed": C.SPLIT_SEED,
        "outcome_support_threshold": C.OUTCOME_SUPPORT_THRESHOLD,
        "targets": list(C.OUTCOMES),
        "date_range_start": C.DATE_RANGE_START,
        "date_range_end": C.DATE_RANGE_END,
        "date_range_require_full_horizon": C.DATE_RANGE_REQUIRE_FULL_HORIZON,
        "event_source": C.EVENT_SOURCE,
        "event_match_tolerance_h": C.EVENT_MATCH_TOLERANCE_H,
        "events_as_inputs": C.EVENTS_AS_INPUTS,
        "kb_events_for_kb_arms_only": getattr(C, "KB_EVENTS_FOR_KB_ARMS_ONLY", True),
        "event_align_across_files": getattr(C, "EVENT_ALIGN_ACROSS_FILES", False),
        "event_align_tolerance_min": getattr(C, "EVENT_ALIGN_TOLERANCE_MIN", 1.0),
        "split_group_by_person": bool(getattr(C, "SPLIT_GROUP_BY_PERSON", False)),
        "raw_temporal_file": paths.RAW_TEMPORAL_FILE,
        "abstract_file": paths.ABSTRACT_FILE,
        "context_file": paths.CONTEXT_FILE,
    }


def cohort_drift(cohort) -> list:
    """
    Purpose: Report which study settings a cached cohort was built under that no
             longer match the live configuration.
    Method:  Compare `cohort.meta` field by field against the current config. A
             cohort written before this check existed carries no entry for a key;
             those are skipped rather than reported, so an older pickle does not
             look uniformly stale.

    Args:
        cohort (Cohort): The loaded cohort.

    Returns:
        list[tuple]: (key, cached_value, current_value) for each mismatch.
    """
    current = _current_cohort_config()
    drift = []
    for key in COHORT_CONFIG_KEYS:
        if key not in cohort.meta:
            continue
        was, now = cohort.meta[key], current[key]
        if was != now:
            drift.append((key, was, now))
    return drift


def clear_derived_artefacts(verbose=True):
    """
    Purpose: Delete every artefact derived from the cohort.
    Method:  Removes the ss-STraTS pickles and the sigma-bin interval table. They
             are keyed to the cohort's samples and labels, so a rebuilt cohort
             must not be paired with the previous run's derived files -- that
             would reintroduce exactly the cross-arm mismatch the contract exists
             to prevent.

    Args:
        verbose (bool): Print each removal.
    """
    from kineret.strats import config as SC
    from kineret.abstraction import std_table_path

    stale = [SC.dataset_pkl(True), SC.dataset_pkl(False), std_table_path(),
             os.path.join(paths.PROCESSED_DIR, "tak_repo_std.json")]
    for path in stale:
        if os.path.exists(path):
            os.remove(path)
            if verbose:
                print(f"[prepare] removed stale artefact: {path}")


def ensure_prepared(arms=None, rebuild=False, verbose=True) -> Cohort:
    """
    Purpose: Build every artefact the requested arms need, before training.
    Method:  Build (or reuse) `cohort.pkl`; emit the ss-STraTS pickle for each
             QA arm that needs one; build the knowledge-free interval table if
             any INTERVenE-sigma arm is requested. INTERVenE and LogReg read the
             cohort directly and need nothing further.

             Idempotent, so it is safe to leave at the top of the notebook.

    Args:
        arms    (list|None): Arm keys. None prepares for the whole ladder.
        rebuild (bool):      Rebuild the cohort even if one exists.
        verbose (bool):      Print progress.

    Returns:
        Cohort: The prepared cohort artefact.
    """
    say = print if verbose else (lambda *a, **kw: None)
    from kineret.cohort import build_cohort
    from kineret.strats import config as SC
    from kineret.strats.preprocess import build_pickle

    paths.ensure_dirs()
    triples = resolve_arms(arms)

    cohort = None
    if os.path.exists(paths.COHORT_PKL) and not rebuild:
        cached = Cohort.load()
        drift = cohort_drift(cached)
        if drift:
            # The cached cohort answers a different question than the one now
            # configured. Rebuilding is the only safe move: reusing it would
            # score every arm against labels the config no longer describes.
            say(f"[prepare] {paths.COHORT_PKL} was built under different "
                f"settings -- rebuilding. Changed:")
            for key, was, now in drift:
                say(f"           {key}: {was!r} -> {now!r}")
            clear_derived_artefacts(verbose=verbose)
            rebuild = True
        else:
            say(f"[prepare] Reusing {paths.COHORT_PKL}")
            cohort = cached
    if cohort is None:
        cohort = build_cohort(verbose=verbose)
        cohort.save()

    needs_qa = any(qa for _k, _m, qa in triples)
    if needs_qa and not cohort.qa_by_k:
        raise RuntimeError(
            "[prepare] A QA arm was requested but the cohort carries no QA "
            f"block. Check {paths.QA_FILE} and rebuild with rebuild=True.")

    strats_arms = sorted({qa for _k, m, qa in triples if m == "strats"})
    missing = [qa for qa in strats_arms if not os.path.exists(SC.dataset_pkl(qa))]
    if missing:
        say(f"[prepare] Building {len(missing)} ss-STraTS pickle(s)...")
        raw = raw_with_hours(cohort)
        for use_qa in missing:
            build_pickle(cohort, use_qa, raw=raw, verbose=verbose)
    elif strats_arms:
        say("[prepare] ss-STraTS pickles already present.")

    if any(m == "intervene_enc_std" for _k, m, _qa in triples):
        from kineret.abstraction import ensure_std_temporal
        ensure_std_temporal(cohort, raw=raw_with_hours(cohort),
                            rebuild=rebuild, verbose=verbose)

    samples = cohort.samples()
    train = samples[samples["split"] == "train"]
    per_window = train.groupby("k").size()
    scorable = int((cohort.patients["trajectory_hours"]
                    > C.EVAL_CONTEXT_DAYS * 24.0).sum())
    say(f"[prepare] Ready: {len(cohort.patients):,} patients, "
        f"{len(cohort.outcome_names)} targets, "
        f"{len(train):,} training samples from "
        f"{train['PatientId'].nunique():,} patients "
        f"({len(train) / max(train['PatientId'].nunique(), 1):.1f} per patient), "
        f"evaluation at K={C.EVAL_CONTEXT_DAYS}.")
    say(f"[prepare] {scorable:,}/{len(cohort.patients):,} patients reach the "
        f"evaluation window and can be scored; the rest are training-only.")
    say("[prepare] Samples per training window (ragged -- a patient contributes "
        "a window only if their stay reaches it):")
    say("            " + "  ".join(f"K{k}:{n}" for k, n in per_window.items()))
    return cohort


def target_support(cohort: Cohort, windows=None) -> pd.DataFrame:
    """
    Purpose: The support table the notebook shows BEFORE any training starts.
    Method:  For every candidate window, the per-target positive count and
             prevalence in that window's label range -- computed for EVERY
             candidate target, including the ones the support filter dropped,
             because the point of the cell is to make that decision visible
             rather than to show its result.

             This is what the evaluation window is chosen from: a larger K gives
             the model more history but leaves a shorter label window, and the
             rare complications lose positives first.

    Args:
        cohort  (Cohort):    Shared cohort artefact.
        windows (list|None): Windows to report. Defaults to the training set
                             plus the evaluation window.

    Returns:
        pd.DataFrame: k, target, n_pos, prevalence, n_pos_train,
                      prevalence_train, clears_threshold, has_head.
    """
    windows = (sorted(set(windows)) if windows is not None
               else sorted(set(C.TRAIN_CONTEXT_DAYS) | {C.EVAL_CONTEXT_DAYS}))

    # Every candidate the cohort resolved, kept or dropped. Computed straight
    # off the event table so a dropped target still gets a row.
    candidates = list(cohort.outcome_names) + sorted(cohort.dropped_outcomes)
    with_head = set(cohort.outcome_names)

    patients = cohort.patients
    is_train = patients["split"] == "train"
    hi = C.HORIZON_END_DAYS * 24.0

    rows = []
    for k in windows:
        # Measured over the admissions that REACH this window -- the same
        # population `build_cohort` applies the support filter to, and the only
        # one a head at this K is ever judged on.
        #
        # Dividing by the whole cohort instead makes every prevalence look
        # smaller than the filter's, by the fraction of admissions long enough
        # to be scored -- 62 % at K=4, 2 % at K=13. The table then disagrees
        # with the filter that actually runs (INFECTION read 0.6 % here while
        # clearing the 1 % floor in the cohort) and understates the head count
        # at every window, worst where the decision is hardest.
        reaches = patients["trajectory_hours"] > k * 24.0
        scorable = set(patients.loc[reaches, "PatientId"])
        train_ids = set(patients.loc[reaches & is_train, "PatientId"])
        n_all, n_train = max(len(scorable), 1), max(len(train_ids), 1)

        window = cohort.events[(cohort.events["hours"] > k * 24.0)
                               & (cohort.events["hours"] <= hi)]
        by_target = window.groupby("outcome")["PatientId"]
        positives = by_target.apply(set) if len(window) else {}
        for target in candidates:
            pids = positives.get(target, set()) if len(window) else set()
            pids = pids & scorable
            n_pos_train = len(pids & train_ids)
            prevalence_train = n_pos_train / n_train
            rows.append({
                "k": k,
                "target": target,
                "n_pos": len(pids),
                "prevalence": len(pids) / n_all,
                "n_pos_train": n_pos_train,
                "prevalence_train": prevalence_train,
                "n_scorable_train": len(train_ids),
                "clears_threshold": prevalence_train >= C.OUTCOME_SUPPORT_THRESHOLD,
                "has_head": target in with_head,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(arms=None, device=None, keep_outputs=False,
               verbose=True) -> pd.DataFrame:
    """
    Purpose: Prove the wiring before committing GPU hours to the real ladder.
    Method:  Run every arm with deliberately crippled hyperparameters into a
             throwaway output directory, then assert the property the whole
             study rests on: that every arm was scored on byte-identical labels
             for byte-identical test patients.

             A failure here is a plumbing bug -- a missing pickle, a column
             mismatch, a checkpoint that never got written. Metrics from this
             run are meaningless and are not reported.

    Args:
        arms         (list|None): Arm keys. None checks the whole ladder.
        device       (str|None):  Torch device string.
        keep_outputs (bool):      Keep the throwaway tree for inspection.
        verbose      (bool):      Print progress.

    Returns:
        pd.DataFrame: One row per arm with status, runtime and error.

    Raises:
        AssertionError: If the arms did not agree on labels or test patients.
    """
    say = print if verbose else (lambda *a, **kw: None)
    triples = resolve_arms(arms)
    cohort = ensure_prepared([k for k, *_ in triples], verbose=verbose)
    device = resolve_device(device, verbose=verbose)

    smoke_root = os.path.join(paths.OUTPUT_DIR, "_smoke")
    if os.path.exists(smoke_root):
        shutil.rmtree(smoke_root)

    say(f"\n[smoke] {len(triples)} arm(s), crippled hyperparameters -- wiring "
        f"only, the metrics are meaningless.\n")

    rows = []
    for arm_key, model, _use_qa in triples:
        started = time.time()
        result = None
        try:
            # Its own checkpoint root, and no resuming: a 1-epoch toy-width run
            # must never be resumed by -- or resume from -- a real one.
            result = run_one(arm_key, cohort=cohort, output_root=smoke_root,
                             device=device, bootstrap_resamples=0, resume=False,
                             checkpoint_root=os.path.join(smoke_root, "checkpoints"),
                             overrides=SMOKE_OVERRIDES.get(model))
            status, detail = "ok", result["output_dir"]
        except Exception as exc:
            if verbose:
                traceback.print_exc()
            status, detail = "FAILED", f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started
        say(f"[smoke] {arm_label(arm_key):<34} {status:<7} ({elapsed:.0f}s)")
        rows.append({"arm": arm_key, "label": arm_label(arm_key), "status": status,
                     "seconds": round(elapsed, 1), "detail": detail})
        del result
        free_memory()

    report = pd.DataFrame(rows)
    failed = report[report["status"] != "ok"]
    if len(failed):
        say("\n[smoke] FAILURES:")
        for _, row in failed.iterrows():
            say(f"  {row['label']}: {row['detail']}")
        return report

    agreement = check_label_agreement(smoke_root, verbose=verbose)
    if not keep_outputs:
        shutil.rmtree(smoke_root, ignore_errors=True)

    if not agreement["labels_identical"]:
        raise AssertionError(
            "[smoke] Arms were scored on DIFFERENT labels. The comparison "
            "would be meaningless. " + agreement["detail"])

    say(f"\n[smoke] All {len(report)} arms wired correctly, and every arm "
        f"agrees on labels and test patients.")
    return report


def check_label_agreement(output_root=None, verbose=True) -> dict:
    """
    Purpose: Verify that every arm was scored on the same targets and patients.
    Method:  Read each run's `test_predictions.csv`, compare the `label_*`
             columns and the patient index against the first one. This is the
             invariant that makes the headline table a comparison rather than
             seven unrelated numbers.

    Args:
        output_root (str|None): Root to scan. Defaults to `outputs/`.
        verbose     (bool):     Print the verdict.

    Returns:
        dict: {'labels_identical': bool, 'pairs': DataFrame, 'detail': str}
    """
    output_root = output_root or paths.OUTPUT_DIR
    runs = []
    for dirpath, _dirs, files in os.walk(output_root):
        if "test_predictions.csv" not in files or "run_meta.json" not in files:
            continue
        if _is_scratch(dirpath, output_root):
            continue
        meta = json.load(open(os.path.join(dirpath, "run_meta.json")))
        df = pd.read_csv(os.path.join(dirpath, "test_predictions.csv"))
        runs.append({"name": os.path.relpath(dirpath, output_root),
                     "model": meta.get("model"), "use_qa": meta.get("use_qa"),
                     "df": df})

    if len(runs) < 2:
        return {"labels_identical": True, "pairs": pd.DataFrame(),
                "detail": "fewer than two runs to compare"}

    reference = runs[0]
    ref = reference["df"].set_index("PatientId").sort_index()
    ref_cols = sorted(c for c in ref.columns if c.startswith("label_"))

    rows, problems = [], []
    for other in runs[1:]:
        cur = other["df"].set_index("PatientId").sort_index()
        cur_cols = sorted(c for c in cur.columns if c.startswith("label_"))
        same_patients = ref.index.equals(cur.index)
        same_targets = ref_cols == cur_cols
        same_labels = (same_patients and same_targets
                       and ref[ref_cols].equals(cur[cur_cols]))
        rows.append({"reference": reference["name"], "other": other["name"],
                     "same_patients": same_patients, "same_targets": same_targets,
                     "identical_labels": same_labels, "n_test": len(ref)})
        if not same_labels:
            problems.append(f"{reference['name']} vs {other['name']}")

    table = pd.DataFrame(rows)
    ok = bool(len(table)) and bool(table["identical_labels"].all())
    if verbose:
        if ok:
            print(f"[check] {len(table)} arm pair(s) compared -- labels and test "
                  f"patients identical in every one.")
        else:
            print(f"[check] LABEL MISMATCH in {len(problems)} pair(s): "
                  + "; ".join(problems))
    return {"labels_identical": ok, "pairs": table, "detail": "; ".join(problems)}


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def run_ladder(arms=None, device=None, seed=C.SEED, bootstrap_resamples=2000,
               skip_existing=True, overrides=None, verbose=True) -> pd.DataFrame:
    """
    Purpose: Train and score the whole ladder in one pass.
    Method:  Sequential, in ladder order. A failed arm is logged and the run
             continues, so one bad configuration does not cost the rest.

             Resumable: an arm with an existing `test_predictions.csv` is
             skipped, so re-running after a crash continues from where it
             stopped. Memory is released after each arm -- results live in
             `outputs/`, not in RAM -- and the log is written after every arm so
             a crash still leaves a record.

    Args:
        arms                (list|None): Arm keys. None runs the whole ladder.
        device              (str|None):  Torch device string.
        seed                (int):       RNG seed.
        bootstrap_resamples (int):       Resamples for the intervals.
        skip_existing       (bool):      Skip arms that already have predictions.
        overrides           (dict|None): {model_key: overrides} per model.
        verbose             (bool):      Print progress.

    Returns:
        pd.DataFrame: One row per arm with status, runtime and output path.
    """
    say = print if verbose else (lambda *a, **kw: None)
    triples = resolve_arms(arms)
    cohort = ensure_prepared([k for k, *_ in triples], verbose=verbose)
    device = resolve_device(device, verbose=verbose)
    overrides = overrides or {}

    os.makedirs(paths.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(paths.OUTPUT_DIR, "run_ladder_log.csv")
    say(f"[ladder] {len(triples)} arms: {[k for k, *_ in triples]}")

    log = []
    for i, (arm_key, model, _use_qa) in enumerate(triples, start=1):
        tag = f"{arm_label(arm_key)}"
        run_dir = run_dir_for(arm_key)
        if skip_existing and os.path.exists(os.path.join(run_dir, "test_predictions.csv")):
            say(f"[{i}/{len(triples)}] {tag} -- already done, skipping.")
            log.append({"arm": arm_key, "label": tag, "status": "skipped",
                        "minutes": 0.0, "detail": run_dir})
            continue

        say(f"\n{'=' * 78}\n[{i}/{len(triples)}] {tag}\n{'=' * 78}")
        started = time.time()
        result = None
        try:
            result = run_one(arm_key, cohort=cohort, device=device, seed=seed,
                             bootstrap_resamples=bootstrap_resamples,
                             overrides=overrides.get(model))
            status, detail = "ok", result["output_dir"]
        except Exception as exc:
            if verbose:
                traceback.print_exc()
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
        elapsed = (time.time() - started) / 60.0
        say(f"[ladder] {tag} -> {status} in {elapsed:.1f} min")
        log.append({"arm": arm_key, "label": tag, "status": status,
                    "minutes": round(elapsed, 2), "detail": detail})

        # Train, score, write, forget. Everything worth keeping is on disk.
        del result
        free_memory(verbose=verbose)
        pd.DataFrame(log).to_csv(log_path, index=False)

    report = pd.DataFrame(log)
    report.to_csv(log_path, index=False)
    n_failed = int((report["status"] == "failed").sum())
    say(f"\n[ladder] {len(report) - n_failed}/{len(report)} arms completed.")
    if n_failed:
        say("[ladder] Failed arms:")
        for _, row in report[report["status"] == "failed"].iterrows():
            say(f"  {row['label']}: {row['detail']}")
    return report


# ---------------------------------------------------------------------------
# Reading the results
# ---------------------------------------------------------------------------

def _arm_of(meta: dict) -> str:
    """Purpose: Recover the arm key from a run's metadata."""
    model, use_qa = meta.get("model"), bool(meta.get("use_qa"))
    for key, (m, qa) in ARM_SPEC.items():
        if m == model and qa == use_qa:
            return key
    return f"{model}{'_qa' if use_qa else ''}"


def load_results(output_root=None, average="weighted") -> pd.DataFrame:
    """
    Purpose: One tidy frame with every finished arm's headline numbers.
    Method:  Walk `outputs/` for runs carrying both `run_meta.json` and
             `test_overall_metrics.csv`, attach the bootstrap intervals where
             they exist, and order the rows by the ladder.

    Args:
        output_root (str|None): Root to scan. Defaults to `outputs/`.
        average     (str):      'weighted' or 'macro'.

    Returns:
        pd.DataFrame: arm, label, rung, n_test + per-metric mean/lo/hi columns.
    """
    output_root = output_root or paths.OUTPUT_DIR
    rows = []
    for dirpath, _dirs, files in os.walk(output_root):
        if "run_meta.json" not in files or "test_overall_metrics.csv" not in files:
            continue
        if _is_scratch(dirpath, output_root):
            continue
        meta = json.load(open(os.path.join(dirpath, "run_meta.json")))
        overall = pd.read_csv(os.path.join(dirpath, "test_overall_metrics.csv"))
        subset = overall[overall["average"] == average]
        if subset.empty:
            continue
        point = subset.iloc[0]

        boot_path = os.path.join(dirpath, "test_bootstrap.json")
        boot = json.load(open(boot_path)) if os.path.exists(boot_path) else {}

        arm = _arm_of(meta)
        row = {"arm": arm, "label": arm_label(arm),
               "rung": ARM_ORDER.index(arm) + 1 if arm in ARM_ORDER else 99,
               "n_test": meta.get("n_test"),
               "n_train_samples": meta.get("n_train_samples"),
               "context_days": meta.get("context_days"),
               "run_dir": os.path.relpath(dirpath, output_root)}

        for metric in METRIC_NAMES:
            block = boot.get("overall", {}).get(metric, {}).get(average) if boot else None
            row[metric] = float(point.get(metric, np.nan))
            row[f"{metric}_lo"] = float(block["lo"]) if block else np.nan
            row[f"{metric}_hi"] = float(block["hi"]) if block else np.nan

        row["onset_mae_h"] = float(point.get("time_mae_h", np.nan))
        onset = boot.get("onset", {}).get(average) if boot else None
        row["onset_mae_h_lo"] = float(onset["lo"]) if onset else np.nan
        row["onset_mae_h_hi"] = float(onset["hi"]) if onset else np.nan

        row["los_mae_h"] = float(point.get("los_mae_hours", np.nan))
        los = boot.get("los") if boot else None
        row["los_mae_h_lo"] = float(los["lo"]) if los else np.nan
        row["los_mae_h_hi"] = float(los["hi"]) if los else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["arm", "label", "rung"] + list(METRIC_NAMES))
    return pd.DataFrame(rows).sort_values("rung").reset_index(drop=True)


def _fmt_ci(mean, lo, hi, digits=3):
    """Purpose: Render one metric as `point [lo, hi]`, or `--` when absent."""
    if not np.isfinite(mean):
        return "--"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return f"{mean:.{digits}f}"


def ladder_table(average="weighted", metrics=("auroc", "auprc", "best_f1"),
                 output_root=None) -> pd.DataFrame:
    """
    Purpose: The study's central claim, as one ordered table.
    Method:  One row per arm in ladder order, each metric rendered as
             `point [95% interval]`. Read it as a monotonicity claim and check
             the intervals before believing any step: adjacent rungs whose
             intervals overlap have not been separated by this data.

    Args:
        average     (str):       'weighted' or 'macro'.
        metrics     (tuple):     Risk metrics to show.
        output_root (str|None):  Root to scan.

    Returns:
        pd.DataFrame: Rung / arm x metrics + onset and LoS error.
    """
    results = load_results(output_root, average=average)
    if results.empty:
        return results

    out = pd.DataFrame({
        "Rung": results["rung"],
        "Arm": results["label"],
        "n_test": results["n_test"],
    })
    for metric in metrics:
        out[metric.upper()] = [
            _fmt_ci(r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"])
            for _, r in results.iterrows()]
    out["Onset MAE (h)"] = [
        _fmt_ci(r["onset_mae_h"], r["onset_mae_h_lo"], r["onset_mae_h_hi"], 1)
        for _, r in results.iterrows()]
    out["LoS MAE (h)"] = [
        _fmt_ci(r["los_mae_h"], r["los_mae_h_lo"], r["los_mae_h_hi"], 1)
        for _, r in results.iterrows()]
    return out.reset_index(drop=True)


def qa_delta_table(average="weighted", metrics=("auroc", "auprc", "best_f1"),
                   output_root=None) -> pd.DataFrame:
    """
    Purpose: Answer the ablation question directly -- does the QA signal help?
    Method:  Pair each model's two arms. They differ by exactly the aggregated
             `QA_<pattern>` context columns (plus, for INTERVenE only, the
             pattern token stream it can natively represent) -- same patients,
             same labels, same events, same seed -- so the delta is attributable
             to the treatment-quality signal rather than to run-to-run variance.

             A delta smaller than the interval width in the headline table is
             noise, not a finding.

    Args:
        average     (str):      'weighted' or 'macro'.
        metrics     (tuple):    Metrics to difference.
        output_root (str|None): Root to scan.

    Returns:
        pd.DataFrame: Model / metric / no QA / with QA / delta.
    """
    results = load_results(output_root, average=average).set_index("arm")
    pairs = [("LogReg", "logreg", "logreg_qa"),
             ("ss-STraTS", "strats", "strats_qa"),
             ("INTERVenE-Enc (KB)", "intervene_kb", "intervene_kb_qa")]
    rows = []
    for name, off_arm, on_arm in pairs:
        if off_arm not in results.index or on_arm not in results.index:
            continue
        for metric in metrics:
            off = float(results.loc[off_arm, metric])
            on = float(results.loc[on_arm, metric])
            rows.append({"Model": name, "Metric": metric.upper(),
                         "no QA": off, "with QA": on, "delta": on - off})
    return pd.DataFrame(rows)


def per_outcome_table(metric="auprc", arms=None, output_root=None) -> pd.DataFrame:
    """
    Purpose: Break the headline down to individual complications.
    Method:  One column per arm, one row per target, plus the positive support
             and the interval-reliability flag -- a strong number on a
             single-digit-support target is noise, and the table should say so
             without a second lookup.

    Args:
        metric      (str):       Metric to tabulate.
        arms        (list|None): Arms to include, in ladder order.
        output_root (str|None):  Root to scan.

    Returns:
        pd.DataFrame: target x arm, with n_pos and ci_reliable.
    """
    output_root = output_root or paths.OUTPUT_DIR
    wanted = [k for k, *_ in resolve_arms(arms)]

    columns, support, reliable = {}, None, None
    for dirpath, _dirs, files in os.walk(output_root):
        if "run_meta.json" not in files or "test_per_outcome_metrics.csv" not in files:
            continue
        if _is_scratch(dirpath, output_root):
            continue
        meta = json.load(open(os.path.join(dirpath, "run_meta.json")))
        arm = _arm_of(meta)
        if arm not in wanted:
            continue
        table = pd.read_csv(os.path.join(dirpath, "test_per_outcome_metrics.csv"))
        table = table.set_index("outcome")
        if metric not in table.columns:
            continue
        columns[arm] = table[metric]
        if support is None:
            support = table["n_pos"]
            reliable = table.get("ci_reliable")

    if not columns:
        return pd.DataFrame()

    out = pd.DataFrame({arm_label(a): columns[a] for a in wanted if a in columns})
    out.insert(0, "n_pos", support)
    if reliable is not None:
        out.insert(1, "ci_reliable", reliable)
    return out.sort_values("n_pos", ascending=False)
