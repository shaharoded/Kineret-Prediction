"""
Task definition for the Kineret study — DEFAULTS ONLY.

Every value here is a shared, study-wide decision, and the notebook is where
those decisions are actually made: `notebooks/benchmark.ipynb` calls
`configure_study(...)` in one visible cell and overrides whatever it needs.
Per-model hyperparameters do NOT live here — they belong to the model:

    kineret/strats/config.py                      ss-STraTS
    kineret/intervene/config/model_config.py      INTERVenE-Enc
    kineret/logreg/train.py::LOGREG_SETTINGS      logistic regression

Keeping the split that way means the notebook shows the whole experimental
design on one screen, and a model's own knobs stay with the model.
"""

# ---------------------------------------------------------------------------
# Context windows.
#
# K = number of CONTEXT days the model observes, counted from ADMISSION.
# For a given K the task is:
#
#   input window : [0, K*24]                    hours
#   label window : (K*24, HORIZON_END_DAYS*24]  hours
#
# TRAIN_CONTEXT_DAYS is a data-AUGMENTATION set, not a sweep. Each training
# admission contributes one sample per K, so the model learns to forecast from
# however much history it happens to have. It is the same patient and the same
# trajectory each time, cut at a different point, with the label window
# re-derived to match.
#
# The augmentation is RAGGED on purpose: a patient contributes a sample at K
# only if their trajectory actually reaches day K. A patient discharged on day 5
# has no day-7 cut to make, and inventing one would train the model on a window
# that does not exist. Long stays therefore contribute more samples than short
# ones -- which is the correct weighting, since they contain more forecastable
# history.
#
# The default spans the whole horizon. There is no cost to a wide range: every
# window is a real cut of data already collected, and cohort membership does not
# depend on it (see below).
#
# EVAL_CONTEXT_DAYS is a SINGLE window. Validation, model selection, the test
# pass and every reported number use only this K, so each model produces exactly
# one score per outcome and the comparison is one table rather than a surface.
# Pick it from the support cell at the top of the notebook, before training:
# a larger K gives the model more history but leaves a shorter label window and
# fewer positives on the rare complications.
# ---------------------------------------------------------------------------
TRAIN_CONTEXT_DAYS = list(range(1, 14))     # 1..13
EVAL_CONTEXT_DAYS  = 4
HORIZON_END_DAYS   = 14.0   # outer bound of the prediction window, in days

# With a fixed horizon end, K sets BOTH how much history the model sees and how
# far ahead it is forecasting: K=1 predicts over 13 days, K=13 over one. Across a
# wide augmentation range those are materially different questions, so the two
# numbers are handed to the model explicitly as context features rather than
# left to be inferred from sequence length. They are constant at evaluation (one
# window), so they contribute a bias there and nothing more -- and K is known at
# prediction time by construction, so this is not leakage.
ADD_CONTEXT_LENGTH_FEATURES = True

# ---------------------------------------------------------------------------
# Study date range — applied ONCE, in `kineret.cohort.build_cohort`, to every
# source table (raw temporal, Mediator output, QA) and therefore to inputs,
# labels, length-of-stay and the QA aggregation alike. Nothing downstream
# re-filters by date; change it here and the whole study moves.
#
# REQUIRE_FULL_HORIZON drops admissions whose label window would run past
# DATE_RANGE_END. Keeping them would silently mark late complications as
# negatives simply because the extract stops — a censoring artefact that looks
# exactly like a real negative to every model. The count dropped is logged.
# ---------------------------------------------------------------------------
DATE_RANGE_START = "2022-07-01"
DATE_RANGE_END   = "2025-12-31"
DATE_RANGE_REQUIRE_FULL_HORIZON = True

# Cohort membership, and the training / evaluation asymmetry.
#
# A patient is in the study if they can furnish at least ONE training sample --
# i.e. their trajectory outlasts the shortest training window. Evaluation is a
# separate, later filter: the val and test splits contain only the patients who
# reach `eval_context_days`, because a 3-day admission cannot be scored at K=4.
#
# So training uses every patient it can, and the reporting window decides only
# who is scored. Raising eval K therefore shrinks val/test but leaves training
# untouched -- the alternative would let a reporting decision quietly evict
# short admissions from training whose early cuts are perfectly good
# supervision.
#
# `min_trajectory_hours` is an additional floor guarding against one-event
# admissions.
MIN_TRAJECTORY_HOURS = 24.0

# ---------------------------------------------------------------------------
# Outcomes.
#
# Canonical names are the Mediator-output spelling (`<NAME>_EVENT`). The raw
# source table (Mediator *input*) spells the same pathologies WITHOUT the
# suffix and occasionally with different punctuation — e.g. Mediator emits
# `CARDIO-VASCULAR_DISORDER_EVENT` where the ETL emitted `CARDIOVASCULAR_DISORDER`.
# `OUTCOME_ALIAS_REGEX` bridges the two spellings; it is used to
#   (a) find outcome rows in either table, and
#   (b) strip those rows out of the STraTS / LogReg input stream so the raw
#       feed cannot leak a label.
# Outcomes below OUTCOME_SUPPORT_THRESHOLD prevalence in the *label window* are
# dropped from the head and logged — they stay in the input vocabulary.
# ---------------------------------------------------------------------------
# CANDIDATE targets: every clinical complication event the Mediator knowledge
# base can emit, minus the structural framing (ADMISSION / RELEASE) and the
# `*_BITZUA_EVENT` treatment-administration events, which are interventions
# rather than outcomes.
#
# >>> These are the study's TARGETS. <<<
# The notebook sets `targets=[...]` on `configure_study` to the study's actual
# target list, and the support cell prints which candidates clear
# `outcome_support_threshold` on the train split at the evaluation window. Only
# those survive into the prediction head — so the head layout is decided by the
# data, visibly, before any training starts.
OUTCOMES = [
    "CARDIO-VASCULAR_DISORDER_EVENT",
    "DEATH_EVENT",
    "HYPERGLYCEMIA_EVENT",
    "HYPEROSMOLALITY_EVENT",
    "HYPOGLYCEMIA_EVENT",
    "INFECTION_EVENT",
    "KETOACIDOSIS_EVENT",
    "KIDNEY_COMPLICATION_EVENT",
    "SEVERE_HYPERGLYCEMIA_EVENT",
    "SEVERE_HYPOGLYCEMIA_EVENT",
]

# Canonical name -> regex matched (fullmatch, case-insensitive) against
# ConceptName in EITHER source table. `-?` absorbs the CARDIO-VASCULAR /
# CARDIOVASCULAR split; `(?:_EVENT)?` absorbs the Mediator suffix.
OUTCOME_ALIAS_REGEX = {
    "ACIDOSIS_EVENT":                   r"ACIDOSIS(?:_EVENT)?",
    "ACUTE_RESPIRATORY_DISORDER_EVENT": r"ACUTE_RESPIRATORY_DISORDER(?:_EVENT)?",
    "CARDIO-VASCULAR_DISORDER_EVENT":   r"CARDIO-?VASCULAR_DISORDER(?:_EVENT)?",
    "DEATH_EVENT":                      r"DEATH(?:_EVENT)?",
    "DIABETIC_COMA_EVENT":              r"DIABETIC_COMA(?:_EVENT)?",
    "HYPERGLYCEMIA_EVENT":              r"HYPERGLYCEMIA(?:_EVENT)?",
    "HYPEROSMOLALITY_EVENT":            r"HYPEROSMOLALITY(?:_EVENT)?",
    "HYPOGLYCEMIA_EVENT":               r"HYPOGLYCEMIA(?:_EVENT)?",
    "INFECTION_EVENT":                  r"INFECTION(?:_EVENT)?",
    "KETOACIDOSIS_EVENT":               r"KETOACIDOSIS(?:_EVENT)?",
    "KIDNEY_COMPLICATION_EVENT":        r"KIDNEY_COMPLICATION(?:_EVENT)?",
    "OTHER_COMPLICATION_EVENT":         r"OTHER_COMPLICATION(?:_EVENT)?",
    "SEVERE_HYPERGLYCEMIA_EVENT":       r"SEVERE_HYPERGLYCEMIA(?:_EVENT)?",
    "SEVERE_HYPOGLYCEMIA_EVENT":        r"SEVERE_HYPOGLYCEMIA(?:_EVENT)?",
    "AKI_EVENT":                        r"AKI(?:_EVENT)?|ACUTE_KIDNEY_INJURY(?:_EVENT)?",
    "ELECTROLYTE_DERANGEMENT_EVENT":    r"ELECTROLYTE_DERANGEMENT(?:_EVENT)?",
    "MYOCARDIAL_INJURY_EVENT":          r"MYOCARDIAL_INJURY(?:_EVENT)?",
}

# When True, any `*_EVENT` ConceptName found in the Mediator output that is not
# already in OUTCOMES is appended as a candidate outcome (then support-filtered).
# Lets the Kineret KB introduce complications the MIMIC list never had.
AUTO_DISCOVER_OUTCOMES = False

# ---------------------------------------------------------------------------
# Event reconciliation.
#
# The raw table and the Mediator output disagree about complications: the raw
# table carries whatever the ETL pre-computed, the Mediator carries what its
# rules fired on. Left alone, an outcome could have 300 occurrences in one file
# and 280 in the other, and then "INTERVenE beats STraTS" might just mean
# "INTERVenE saw a different set of events".
#
# So the cohort builds ONE canonical event table (`Cohort.events`) and every
# consumer is rewritten against it: each input stream has its own event rows
# stripped and the canonical ones injected in their place. After that step the
# event support is identical in every file, for every model, by construction.
#
# EVENT_SOURCE picks how the canonical table is formed:
#   "mediator"  the Mediator's verdict wins; raw-only outcomes fall back to raw
#               (the abstraction engine is the authority on its own rules)
#   "raw"       the mirror image
#   "union"     an occurrence in either file counts
#   "intersect" only occurrences both files agree on (within MATCH_TOLERANCE_H)
EVENT_SOURCE = "mediator"

# Two occurrences of the same outcome for the same patient within this many
# hours are treated as the same clinical event when reconciling the two files.
EVENT_MATCH_TOLERANCE_H = 24.0

# Whether canonical events occurring INSIDE the observation window are given to
# the models as input tokens. They are observed history, not the forecast
# target, so this is not leakage — the label window starts strictly after the
# input window. It is True because INTERVenE's abstraction stream contains them
# natively; withholding them from STraTS / LogReg would hand INTERVenE an
# advantage that has nothing to do with abstraction quality.
EVENTS_AS_INPUTS = True

# ---------------------------------------------------------------------------
# Knowledge containment.
#
# The Mediator emits `*_EVENT` concepts that are NOT prediction targets --
# AKI_EVENT, ELECTROLYTE_DERANGEMENT_EVENT, MYOCARDIAL_INJURY_EVENT and the
# like. Those are knowledge-base *products*: the KB read raw measurements and
# concluded something. They are legitimate context for a model that is being
# credited with using a knowledge base, and they are exactly the wrong thing to
# hand a model that is not.
#
# Injecting them into the raw stream would let LogReg and ss-STraTS read a
# derived clinical judgement they could never have computed themselves, and the
# ladder's whole claim -- that KB abstractions beat raw data -- would be
# measuring a leak instead of an abstraction.
#
# So: TARGET events are harmonised into every stream (they are the labels, and
# in-window occurrences are observed history every arm must share). NON-TARGET
# KB events reach only the arms whose input IS the knowledge base.
KB_EVENTS_FOR_KB_ARMS_ONLY = True

# ---------------------------------------------------------------------------
# Cross-file event alignment.
#
# For an outcome BOTH files carry, an occurrence is only credited when the two
# files put it at the same instant (within the tolerance below). This removes
# the small support drift -- 3096 vs 3098, 2849 vs 2854 -- that otherwise makes
# "which events did each arm see" a confound rather than a constant.
#
# Applied ONLY to outcomes present in both files. An outcome the raw ETL never
# pre-computed (HYPERGLYCEMIA_EVENT, KIDNEY_COMPLICATION_EVENT, ...) exists
# solely in the Mediator output; intersecting it against an empty raw view would
# delete the target entirely, so those pass through untouched.
EVENT_ALIGN_ACROSS_FILES = True

# Two timestamps this far apart are the same instant. Minutes, deliberately:
# this is a clock-skew allowance between two exports of the same event, not the
# clinical merge window (that is EVENT_MATCH_TOLERANCE_H).
EVENT_ALIGN_TOLERANCE_MIN = 1.0



# ---------------------------------------------------------------------------
# Raw-side event derivation.
#
# `mediator_input.csv` holds measurements, not complications -- the Mediator is
# what turns a glucose reading into HYPERGLYCEMIA_EVENT. So the raw file carries
# 0 occurrences of the lab-threshold targets while the Mediator output carries
# tens of thousands, and there is nothing for the cross-file intersection to
# intersect on exactly the targets that matter most.
#
# These rules rebuild the raw file's own view of those events by applying the
# very rules the Mediator ran. Every target is derived, including the ones whose
# rule is a plain observation pass-through, so one code path produces the whole
# raw-side view and the intersection is a real agreement test for all ten.
DERIVE_RAW_EVENTS = True

# The rules are NOT written here. They are compiled from the Mediator's own
# knowledge base by `kineret.mediator_rules` and executed verbatim by
# `kineret.raw_events`, because a hand-transcribed threshold is exactly how a
# silent mismatch gets introduced -- and any mismatch here is a defect, not a
# disagreement: the Mediator is deterministic and produced mediator_output.csv
# from mediator_input.csv with these very rules.
#
#   python -m kineret.mediator_rules <path-to>/Mediator/core/knowledge-base
#
# ships the result as kineret/config/event_rules.json. Point MEDIATOR_KB_PATH at
# a live checkout to compile from the XML at run time instead, which is the
# safer choice if the knowledge base has moved on.
MEDIATOR_KB_PATH = None
EVENT_RULES_PATH = None

# The derived view must reproduce the Mediator exactly. Anything less means a
# rule is wrong, so the cohort build stops rather than quietly training on
# labels that two halves of the pipeline disagree about.
RULE_AGREEMENT_MIN = 1.0
RAISE_ON_RULE_MISMATCH = True

# Minimum patient-level prevalence, measured on the TRAIN split inside the
# label window, for an outcome to keep a prediction head.
OUTCOME_SUPPORT_THRESHOLD = 0.01   # 1 %

# Structural tokens. These frame the admission; they are never model inputs for
# STraTS/LogReg (they would leak the terminus) and DEATH doubles as an outcome.
ADMISSION_TOKEN = "ADMISSION_EVENT"
RELEASE_TOKEN   = "RELEASE_EVENT"
DEATH_TOKEN     = "DEATH_EVENT"
STRUCTURAL_REGEX = {
    "ADMISSION_EVENT": r"ADMISSION(?:_EVENT)?",
    "RELEASE_EVENT":   r"RELEASE(?:_EVENT)?|DISCHARGE(?:_EVENT)?",
    "DEATH_EVENT":     r"DEATH(?:_EVENT)?",
}

# ---------------------------------------------------------------------------
# QA ablation.
#
# Every model runs twice: once with QA compliance features folded into its
# static context vector, once without. The aggregation is identical for all
# three models (mean ComplianceScore per PatternName over [0, K*24] hours), so
# the with/without arms differ by exactly those columns and nothing else.
# The notebook sets this per arm via `configure_study`; the value here is the
# default for ad-hoc single runs.
# ---------------------------------------------------------------------------
# The QA arm is deliberately ASYMMETRIC, because the models are:
#
#   LogReg / ss-STraTS          static context vector only.
#       They have no notion of a pattern token, so the only way to give them the
#       treatment-quality signal is the aggregated compliance score appended to
#       their static feature vector.
#
#   INTERVenE-Enc               static context vector AND `%_PATTERN%` tokens.
#       It is an interval model over Mediator output, so it can consume the
#       compliance patterns natively as temporal tokens, on top of the same
#       aggregated vector everyone else gets. `temporal_filters()` drops the
#       `%_PATTERN%` concepts when the arm is off and keeps them when it is on.
#
# So "with QA" means the same aggregated block for every model, plus the extra
# token stream for the one model that can represent it. That is the honest
# comparison: each model gets the treatment signal in the richest form its
# architecture admits.
USE_QA_DATA = False
# ---------------------------------------------------------------------------
# Context table identity.
#
# The Mediator carries exactly one id column, so the temporal tables key on the
# ADMISSION (`PatientId` there is really a visit id). The context table, which
# the Mediator did not have to squeeze through that constraint, keys on
# `VisitId` and additionally carries `person_id` -- the human, who may have
# several admissions in the extract.
#
# CONTEXT_ID_COLUMN is renamed to PatientId on load and is the join key.
# CONTEXT_PERSON_COLUMN is never a feature (it is an identifier; handing a model
# a person id invites it to memorise people) but it IS used to group the split.
# ---------------------------------------------------------------------------
# Temporal table identity and framing.
#
# `mediator_input.csv` carries BOTH ids -- `PatientId` is the person, `VisitId`
# is the admission -- plus the admission window it was cut from. The Mediator
# output carries only the admission, as `PatientId`, because the engine accepts
# one id column. `normalise_temporal` re-keys everything onto the admission.
#
# It also carries `relevant_admission`, marking rows that belong to the
# admission the export was built around. Rows explicitly flagged False are
# dropped; blank/True rows are kept, because the ADMISSION row itself is blank.
RAW_RELEVANT_ADMISSION_COLUMN = "relevant_admission"
DROP_IRRELEVANT_ADMISSION_ROWS = True

# Prefer the export's own AdmissionStart/AdmissionEnd over inferring the window
# from ADMISSION / RELEASE concept rows. The columns are what the extract was
# actually cut on, so they anchor t=0 exactly and give a length of stay that
# does not depend on a RELEASE row having been emitted.
ADMISSION_WINDOW_FROM_COLUMNS = True

CONTEXT_ID_COLUMN     = "VisitId"
CONTEXT_PERSON_COLUMN = "person_id"

# Keep every admission of the same person on ONE side of the split. Two
# admissions of one patient share comorbidities, baseline physiology and often
# the same complication; scoring one while having trained on the other measures
# memorisation of that person, not generalisation to a new one.
SPLIT_GROUP_BY_PERSON = True

QA_PATTERN_COLUMN    = "PatternName"
QA_SCORE_COLUMN      = "ComplianceScore"
QA_COLUMN_PREFIX     = "QA_"

# How per-pattern compliance is summarised over the [0, K*24] observation window
# to form the context columns. This is the single shared definition -- every
# model that puts QA in its context vector uses exactly this.
#
# "mean" alone reproduces the aggregation used in the geriatric quality-of-care
# work (Shalom et al., JBI 2024) and in the INTERVenE thesis code: the mean
# compliance score per pattern over the window. Adding more aggregations widens
# the block for every model at once, so the arms stay comparable.
# With a single "mean" the columns are `QA_<pattern>`; with several they become
# `QA_<pattern>_<agg>`.
QA_AGGREGATIONS = ["mean"]          # any of: mean, min, max, last, count, std

# ---------------------------------------------------------------------------
# Abstraction ablation — the second axis of the study.
#
# INTERVenE-Enc is an interval model, so "is the knowledge base worth it?" is
# answered by feeding it intervals built WITHOUT knowledge and comparing:
#
#   std : distribution-derived bins over the raw measurements (this block)
#   kb  : the Mediator's knowledge-based abstractions
#
# Combined with the QA axis that gives the three-rung ladder the study argues:
#
#   raw measurements (LogReg / ss-STraTS — no abstraction at all)
#     < std intervals            (INTERVenE-Enc, abstraction="std")
#     < KB abstractions          (INTERVenE-Enc, abstraction="kb", QA off)
#     < KB + treatment patterns  (INTERVenE-Enc, abstraction="kb", QA on)
#
# Ported from autoresearch/autoresearch-encoder/ablation/preprocess_std_bins.py.
# Seven bins at ±0.5σ, ±1σ, ±2σ. Under a true N(0,1) that gives roughly
# NORMAL 38 %, each ±0.5–1σ band 15 %, each ±1–2σ band 14 %, each tail 2.3 %.
# ---------------------------------------------------------------------------
ABSTRACTION_SOURCES = ["kb", "std"]

STD_BIN_EDGES = [float("-inf"), -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, float("inf")]
STD_BIN_LABELS = ["VERY_LOW", "LOW", "SLIGHTLY_LOW", "NORMAL",
                  "SLIGHTLY_HIGH", "HIGH", "VERY_HIGH"]

# Consecutive same-concept observations no more than this far apart merge into
# one interval. Without it the std arm would be a stream of points and the
# encoder would have no interval structure to model at all.
STD_COLLAPSE_HOURS = 24.0

# Whether the std arm also receives the Mediator's derived complication events
# as input tokens. False (default) is the honest baseline: with no knowledge
# base you do not get HYPERGLYCEMIA_EVENT either. Set True to hold event
# availability constant and isolate the representation difference alone.
STD_INCLUDE_KB_EVENTS = False

# ---------------------------------------------------------------------------
# Splits — computed once in `build_cohort` and reused verbatim by
# every arm and every K, so no comparison is ever confounded by a re-split.
# ---------------------------------------------------------------------------
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 2023
SEED = 2023

# ---------------------------------------------------------------------------
# Inclusion / exclusion filters, INTERVenE's SQL-like dialect. Applied to the
# Mediator-output table before abstraction tokenisation.
# `%_PATTERN%` rows carry treatment-quality signal that only the QA arm
# consumes, so they are dropped when USE_QA_DATA is False.
# ---------------------------------------------------------------------------
def temporal_filters(use_qa: bool = None):
    """
    Purpose: Build the INTERVenE inclusion/exclusion criteria for a run.
    Method:  Always drop `Steady` values (non-informative abstraction states);
             additionally drop `%_PATTERN%` concepts when the QA arm is off.

    Args:
        use_qa (bool|None): QA arm flag. None falls back to USE_QA_DATA.

    Returns:
        dict: {"temporal": [conditions], "context": []} for DataProcessor.
    """
    use_qa = USE_QA_DATA if use_qa is None else use_qa
    conditions = ["WHERE Value NOT LIKE '%Steady%'"]
    if not use_qa:
        conditions.append("WHERE ConceptName NOT LIKE '%_PATTERN%'")
    return {"temporal": conditions, "context": []}


# ---------------------------------------------------------------------------
# One entry point for every shared decision.
# ---------------------------------------------------------------------------

# Names `configure_study` will accept, mapped to their module-level globals.
# Anything not listed here is a per-model hyperparameter and belongs in that
# model's own config, not in the study config.
_STUDY_KEYS = {
    "train_context_days": "TRAIN_CONTEXT_DAYS",
    "eval_context_days": "EVAL_CONTEXT_DAYS",
    "horizon_end_days": "HORIZON_END_DAYS",
    "date_range_start": "DATE_RANGE_START",
    "date_range_end": "DATE_RANGE_END",
    "date_range_require_full_horizon": "DATE_RANGE_REQUIRE_FULL_HORIZON",
    "min_trajectory_hours": "MIN_TRAJECTORY_HOURS",
    "add_context_length_features": "ADD_CONTEXT_LENGTH_FEATURES",
    "targets": "OUTCOMES",
    "auto_discover_outcomes": "AUTO_DISCOVER_OUTCOMES",
    "outcome_support_threshold": "OUTCOME_SUPPORT_THRESHOLD",
    "event_source": "EVENT_SOURCE",
    "event_match_tolerance_h": "EVENT_MATCH_TOLERANCE_H",
    "events_as_inputs": "EVENTS_AS_INPUTS",
    "kb_events_for_kb_arms_only": "KB_EVENTS_FOR_KB_ARMS_ONLY",
    "event_align_across_files": "EVENT_ALIGN_ACROSS_FILES",
    "event_align_tolerance_min": "EVENT_ALIGN_TOLERANCE_MIN",
    "derive_raw_events": "DERIVE_RAW_EVENTS",
    "mediator_kb_path": "MEDIATOR_KB_PATH",
    "event_rules_path": "EVENT_RULES_PATH",
    "rule_agreement_min": "RULE_AGREEMENT_MIN",
    "raise_on_rule_mismatch": "RAISE_ON_RULE_MISMATCH",
    "drop_irrelevant_admission_rows": "DROP_IRRELEVANT_ADMISSION_ROWS",
    "admission_window_from_columns": "ADMISSION_WINDOW_FROM_COLUMNS",
    "context_id_column": "CONTEXT_ID_COLUMN",
    "context_person_column": "CONTEXT_PERSON_COLUMN",
    "split_group_by_person": "SPLIT_GROUP_BY_PERSON",
    "qa_aggregations": "QA_AGGREGATIONS",
    "std_include_kb_events": "STD_INCLUDE_KB_EVENTS",
    "std_collapse_hours": "STD_COLLAPSE_HOURS",
    "seed": "SEED",
    "split_seed": "SPLIT_SEED",
    "split_fractions": "SPLIT_FRACTIONS",
}


def configure_study(**overrides):
    """
    Purpose: Set every shared study decision in one place, from the notebook.
    Method:  Rebind the module globals named in `_STUDY_KEYS`, then push the new
             values into any already-imported `kineret.*` module that holds a
             copy. The push matters because several modules pull this config in
             with `from ... import *`, which snapshots the values at import
             time; rebinding here alone would leave them stale.

             Unknown keys raise rather than being silently ignored — a typo in
             the notebook's config cell would otherwise look like it worked and
             quietly run the default study.

    Args:
        overrides: Any of the keys in `_STUDY_KEYS`.

    Returns:
        dict: The full resolved study configuration, for printing/logging.
    """
    import sys

    unknown = set(overrides) - set(_STUDY_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown study setting(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(_STUDY_KEYS)}. Per-model hyperparameters are "
            f"not study settings -- pass those to the model's own config."
        )

    module = sys.modules[__name__]
    for key, value in overrides.items():
        setattr(module, _STUDY_KEYS[key], value)

    # `train_context_days` must contain the evaluation window: the model has to
    # have been trained at the K it is judged on.
    if EVAL_CONTEXT_DAYS not in TRAIN_CONTEXT_DAYS:
        raise ValueError(
            f"eval_context_days={EVAL_CONTEXT_DAYS} is not in "
            f"train_context_days={TRAIN_CONTEXT_DAYS}. The evaluation window "
            f"must be one the model was trained on."
        )
    # A window at or past the horizon has an empty label range -- nothing to
    # forecast, so nothing to learn from.
    too_late = [k for k in TRAIN_CONTEXT_DAYS if k >= HORIZON_END_DAYS]
    if too_late:
        raise ValueError(
            f"train_context_days {too_late} are at or past "
            f"horizon_end_days={HORIZON_END_DAYS}, leaving an empty label "
            f"window (K, {HORIZON_END_DAYS}]. Drop them or extend the horizon."
        )

    resolved = {key: getattr(module, name) for key, name in _STUDY_KEYS.items()}
    # Re-export into every module that snapshotted these names.
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("kineret") or mod is None or mod is module:
            continue
        for name in _STUDY_KEYS.values():
            if hasattr(mod, name):
                setattr(mod, name, getattr(module, name))
    return resolved


def study_summary() -> str:
    """
    Purpose: A printable echo of the shared design, for the notebook header and
             the run logs.

    Returns:
        str: Multi-line summary.
    """
    train = ", ".join(str(k) for k in TRAIN_CONTEXT_DAYS)
    return "\n".join([
        f"study window        : {DATE_RANGE_START or '-inf'} .. {DATE_RANGE_END or '+inf'}"
        f"  (full horizon required: {DATE_RANGE_REQUIRE_FULL_HORIZON})",
        f"training windows    : K in [{train}] days  (augmentation: one sample "
        f"per K the patient reaches)",
        f"evaluation window   : K = {EVAL_CONTEXT_DAYS} days  (single; all reported numbers)",
        f"label window        : (K*24, {HORIZON_END_DAYS:.0f}*24] hours from admission",
        f"candidate targets   : {len(OUTCOMES)}  (support-filtered at "
        f"{OUTCOME_SUPPORT_THRESHOLD:.0%} on train)",
        f"event reconciliation: {EVENT_SOURCE!r}, tolerance {EVENT_MATCH_TOLERANCE_H:.0f} h,"
        f" events as inputs: {EVENTS_AS_INPUTS}",
        f"QA aggregation      : {QA_AGGREGATIONS}",
        f"seeds               : split={SPLIT_SEED}, run={SEED}",
    ])
