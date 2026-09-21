"""
The shared cohort layer.

This module is the reason the benchmark is comparable. It resolves the patient
set, the train/val/test split, the outcome label matrix for every K in the
sweep, and the QA context block ONCE -- and every model then consumes those
artefacts rather than deriving its own. INTERVenE-Enc, ss-STraTS and the
logistic-regression baseline therefore always answer the exact same question on
the exact same patients.

Label contract, for context window K days and horizon end N days:

    inputs  : events in [0, K*24]              hours from ADMISSION
    label   : outcome occurs in (K*24, N*24]   hours from ADMISSION
    LoS     : hours from ADMISSION to RELEASE  (NaN if died / no terminus)

Two further things happen here and nowhere else:

* **The study date range.** Every source table is clipped to
  `DATE_RANGE_START .. DATE_RANGE_END` in one place, so inputs, labels,
  length-of-stay and the QA aggregation cannot drift apart.
* **Event reconciliation.** The raw file and the Mediator output disagree about
  how often each complication fired. The cohort reconciles them into one
  canonical table, uses it for labels, and re-injects it into every model's
  input stream (`harmonise_events`), so afterwards the event support is
  identical everywhere by construction.
"""

import json
import os
import re
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kineret.config import paths
from kineret.config import data_config as C
from kineret.io_utils import load_table, normalise_temporal, match_concepts

# A sample is a (patient, context window) pair, and its id encodes both. The
# format lives here alone: every model writes its predictions keyed by the
# PATIENT, so `patient_of_sample` is what turns a run's rows back into
# something the cross-arm label check can compare.
SAMPLE_SEP = "@k"


def make_sample_id(patient_id, k) -> str:
    """Purpose: The canonical id for one (patient, window) sample."""
    return f"{patient_id}{SAMPLE_SEP}{k}"


def patient_of_sample(sample_ids):
    """
    Purpose: Recover the patient a sample belongs to.
    Method:  Split on the separator and restore the original dtype, so a run's
             prediction file is keyed the same way whatever produced it.

    Args:
        sample_ids (array-like): Sample ids.

    Returns:
        np.ndarray: PatientIds, numeric where the originals were numeric.
    """
    raw = pd.Series(sample_ids, dtype="object").astype(str)             .str.split(SAMPLE_SEP).str[0]
    numeric = pd.to_numeric(raw, errors="coerce")
    return (numeric.to_numpy() if numeric.notna().all() else raw.to_numpy())


@dataclass
class Cohort:
    """
    Container for every shared artefact, pickled to ``data/processed/cohort.pkl``.

    Attributes:
        patients (pd.DataFrame): PatientId, admission_time, los_hours,
            trajectory_hours, split.
        events (pd.DataFrame): PatientId, outcome, hours -- every canonical
            outcome occurrence, in hours from that patient's admission.
        outcome_names (list[str]): Canonical outcomes that survived the support
            filter, measured on the TRAIN split at the *largest* K in the grid
            (the tightest label window, so an outcome kept here is learnable at
            every K and the head layout never changes between runs).
        dropped_outcomes (dict): outcome -> train prevalence, for the ones cut.
        context (pd.DataFrame): PatientId-indexed static features, unscaled.
        qa_by_k (dict[int, pd.DataFrame]): K -> PatientId-indexed QA_<pattern>
            columns aggregated over [0, K*24] hours.
        outcome_aliases (dict[str, list[str]]): canonical -> observed
            ConceptName spellings across both source tables. Doubles as the
            leakage blocklist for the STraTS / LogReg input builders.
        structural_aliases (dict[str, list[str]]): same, for ADMISSION /
            RELEASE / DEATH framing tokens.
        event_audit (pd.DataFrame): per-outcome reconciliation -- how many
            occurrences each source file carried and what the canonical table
            ended up with. This is the evidence that the two files were brought
            onto identical event support.
        meta (dict): provenance -- row counts, config echo, source paths.
    """
    patients: pd.DataFrame
    events: pd.DataFrame
    outcome_names: list
    dropped_outcomes: dict
    context: pd.DataFrame
    qa_by_k: dict
    outcome_aliases: dict
    structural_aliases: dict
    event_audit: pd.DataFrame = None
    # `*_EVENT` concepts the Mediator emits that are NOT prediction targets.
    # KB conclusions: available to the KB arms, withheld from everyone else.
    kb_event_names: list = field(default_factory=list)
    # ConceptNames the raw file uses for a target that `raw_events` has since
    # re-derived under the canonical name (`INFECTION`, `KETOACIDOSIS`,
    # `KIDNEY_COMPLICATION_OBS`, ...). The cohort's own copy was rewritten, but
    # any stream read fresh from disk still spells them the old way and must be
    # stripped the same as the canonical ones.
    replaced_spellings: list = field(default_factory=list)
    # VisitId -> person_id, when the context table supplies it. Used to keep all
    # admissions of one person on the same side of the split.
    person_of_patient: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- labels

    def labels_for_k(self, k: int) -> pd.DataFrame:
        """
        Purpose: Build the per-patient supervision block for one context window.
        Method:  Positive iff the outcome has at least one occurrence strictly
                 after ``k*24`` h and at or before ``HORIZON_END_DAYS*24`` h.
                 ``<outcome>__first_hour`` records the earliest such occurrence
                 (``inf`` when negative), used for time-head MAE and traceability.

        Args:
            k (int): Context-window length in days.

        Returns:
            pd.DataFrame: PatientId + one 0/1 column and one `__first_hour`
                          column per kept outcome + `length_of_stay_hours`.
        """
        lo, hi = k * 24.0, C.HORIZON_END_DAYS * 24.0
        pids = self.patients["PatientId"].to_numpy()
        out = pd.DataFrame({"PatientId": pids})

        window = self.events[(self.events["hours"] > lo) & (self.events["hours"] <= hi)]
        # One groupby builds the whole matrix; a per-outcome filter loop would
        # re-scan the event table K times per outcome.
        first = window.groupby(["PatientId", "outcome"])["hours"].min().unstack("outcome")
        first = first.reindex(index=pids, columns=self.outcome_names)

        for name in self.outcome_names:
            col = first[name]
            out[name] = col.notna().to_numpy().astype(int)
            out[f"{name}__first_hour"] = col.fillna(np.inf).to_numpy()

        out = out.merge(
            self.patients[["PatientId", "los_hours"]].rename(
                columns={"los_hours": "length_of_stay_hours"}),
            on="PatientId", how="left",
        )
        return out

    def samples(self, split=None, windows=None) -> pd.DataFrame:
        """
        Purpose: Enumerate the (patient, K) samples a model trains and is scored on.
        Method:  TRAINING is augmented -- each train patient contributes one
                 sample per K in `TRAIN_CONTEXT_DAYS`, same trajectory cut at a
                 different point with the label window re-derived to match, so
                 the model learns to forecast from however much history it has.

                 VALIDATION and TEST are not augmented: one sample per patient
                 at `EVAL_CONTEXT_DAYS`. Model selection and every reported
                 number therefore describe a single, stated context window --
                 the alternative would make "which K" a hidden degree of
                 freedom in the results.

        Args:
            split   (str|None):  'train' | 'val' | 'test'. None returns all.
            windows (list|None): Override the training windows (diagnostics).

        Returns:
            pd.DataFrame: sample_id, PatientId, k, split.
        """
        train_windows = list(windows) if windows is not None else list(C.TRAIN_CONTEXT_DAYS)
        eval_k = C.EVAL_CONTEXT_DAYS
        reach = self.patients.set_index("PatientId")["trajectory_hours"]

        frames = []
        for name, group in self.patients.groupby("split", sort=False):
            ks = sorted(train_windows) if name == "train" else [eval_k]
            for k in ks:
                # Ragged: only patients whose trajectory actually reaches day K
                # contribute a sample there. Cutting a 5-day admission at day 7
                # would train the model on a window that does not exist.
                eligible = group[reach.reindex(group["PatientId"]).to_numpy() > k * 24.0]
                if eligible.empty:
                    continue
                frames.append(pd.DataFrame({
                    "sample_id": eligible["PatientId"].astype(str) + f"{SAMPLE_SEP}{k}",
                    "PatientId": eligible["PatientId"].to_numpy(),
                    "k": k,
                    "split": name,
                }))
        out = pd.concat(frames, ignore_index=True)
        if split is not None:
            out = out[out["split"] == split].reset_index(drop=True)
        return out.sort_values(["split", "k", "PatientId"]).reset_index(drop=True)

    def labels_for_samples(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose: The supervision block for a set of augmented samples.
        Method:  Each sample carries its own K, so its label window is
                 (k*24, HORIZON_END_DAYS*24]. Grouping by K and reusing
                 `labels_for_k` keeps one definition of a label in the codebase.

        Args:
            samples (pd.DataFrame): Output of `samples()`.

        Returns:
            pd.DataFrame: sample_id, PatientId, k, one 0/1 and one
                          `__first_hour` column per kept outcome,
                          `length_of_stay_hours`.
        """
        blocks = []
        for k, group in samples.groupby("k", sort=True):
            block = self.labels_for_k(int(k)).set_index("PatientId")
            block = block.reindex(group["PatientId"]).reset_index(drop=True)
            block.insert(0, "sample_id", group["sample_id"].to_numpy())
            block.insert(1, "PatientId", group["PatientId"].to_numpy())
            block.insert(2, "k", int(k))
            blocks.append(block)
        out = pd.concat(blocks, ignore_index=True)
        # Restore the caller's ordering so it lines up with their feature rows.
        return out.set_index("sample_id").reindex(samples["sample_id"]).reset_index()

    def context_for_samples(self, samples: pd.DataFrame, use_qa: bool) -> pd.DataFrame:
        """
        Purpose: The static feature block for a set of augmented samples.
        Method:  Base context is per patient; the QA block is per (patient, K),
                 aggregated over that sample's own [0, k*24] window. A K=2
                 sample therefore sees only two days of compliance -- QA can
                 never describe history the sample was not given.

        Args:
            samples (pd.DataFrame): Output of `samples()`.
            use_qa  (bool):         QA ablation arm.

        Returns:
            pd.DataFrame: sample_id-indexed, float32, unscaled.
        """
        blocks = []
        for k, group in samples.groupby("k", sort=True):
            block = self.context_for_k(int(k), use_qa).reindex(group["PatientId"])
            block.index = pd.Index(group["sample_id"].to_numpy(), name="sample_id")
            blocks.append(block)
        out = pd.concat(blocks)
        return out.reindex(samples["sample_id"]).astype("float32")

    def sample_ids(self, split: str) -> np.ndarray:
        """Purpose: Just the sample ids for one split, in canonical order."""
        return self.samples(split)["sample_id"].to_numpy()

    def split_ids(self, k: int = None):
        """
        Purpose: PATIENT-level split, unchanged by the augmentation.
        Method:  Every sample of a patient inherits that patient's split, so the
                 split is a property of the patient and `k` is accepted only for
                 call-site compatibility.

        Args:
            k (int|None): Ignored; the split does not depend on the window.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (train, val, test) ids.
        """
        p = self.patients
        return tuple(
            p.loc[p["split"] == s, "PatientId"].to_numpy() for s in ("train", "val", "test")
        )

    def context_for_k(self, k: int, use_qa: bool) -> pd.DataFrame:
        """
        Purpose: The static feature block a model conditions on for one run.
        Method:  Base context, optionally column-concatenated with the QA block
                 aggregated over the same [0, K*24] window the model observes --
                 so QA features can never describe the future being predicted.

        Args:
            k      (int):  Context-window length in days.
            use_qa (bool): QA ablation arm.

        Returns:
            pd.DataFrame: PatientId-indexed, float32, unscaled.
        """
        ctx = self.context.copy()
        if use_qa:
            qa = self.qa_by_k.get(k)
            if qa is None or qa.shape[1] == 0:
                raise RuntimeError(
                    f"[Cohort] use_qa=True but no QA features exist for K={k}. "
                    f"Check {paths.QA_FILE} and re-run the preparation step."
                )
            ctx = ctx.join(qa.reindex(ctx.index).fillna(0.0), how="left")

        if C.ADD_CONTEXT_LENGTH_FEATURES:
            # How much history this sample got, and how far ahead it is being
            # asked to forecast. Across a wide augmentation range those are
            # materially different questions -- K=1 predicts over 13 days, K=13
            # over one -- and leaving the model to infer it from sequence length
            # is asking it to solve a second problem for free. Constant at
            # evaluation, where there is a single window.
            ctx["context_days"] = float(k)
            ctx["label_window_days"] = float(C.HORIZON_END_DAYS) - float(k)

        return ctx.astype("float32")

    def canonical_event_rows(self, patient_ids=None) -> pd.DataFrame:
        """
        Purpose: The canonical events, shaped like a temporal table for injection.
        Method:  Convert `self.events` back into
                 (PatientId, ConceptName, StartDateTime, EndDateTime, Value)
                 using each patient's admission anchor, with the canonical
                 `<NAME>_EVENT` spelling and Value="True" -- exactly the shape
                 the Mediator emits.

        Args:
            patient_ids (array-like|None): Restrict to these patients.

        Returns:
            pd.DataFrame: Instantaneous event rows.
        """
        events = self.events
        if patient_ids is not None:
            events = events[events["PatientId"].isin(set(np.asarray(patient_ids).tolist()))]
        if len(events) == 0:
            return pd.DataFrame(columns=["PatientId", "ConceptName", "StartDateTime",
                                         "EndDateTime", "Value"])
        admission = self.patients.set_index("PatientId")["admission_time"]
        stamps = (events["PatientId"].map(admission)
                  + pd.to_timedelta(events["hours"], unit="h"))
        return pd.DataFrame({
            "PatientId": events["PatientId"].to_numpy(),
            "ConceptName": events["outcome"].to_numpy(),
            "StartDateTime": stamps.to_numpy(),
            "EndDateTime": stamps.to_numpy(),
            "Value": "True",
        })

    def harmonise_events(self, table: pd.DataFrame, patient_ids=None,
                         kb_events: bool = False, verbose=False) -> pd.DataFrame:
        """
        Purpose: Force one input stream onto the canonical event support.
        Method:  Strip every row whose ConceptName is any spelling of any
                 outcome, then splice the canonical event rows back in.

                 This is the step that makes the models comparable at the input
                 level. The raw file and the Mediator output disagree about how
                 many times each complication fired; after this call they carry
                 the identical set, so a difference in results cannot be a
                 difference in which events the model happened to see.

                 Honours `EVENTS_AS_INPUTS`: when False the events are stripped
                 and not re-injected, so no model sees any outcome token.

                 `kb_events` decides what happens to the Mediator's NON-target
                 `*_EVENT` concepts (AKI_EVENT, ELECTROLYTE_DERANGEMENT_EVENT,
                 ...). They are knowledge-base conclusions, so they stay only in
                 the streams of the arms being credited with a knowledge base.
                 Pass True for the KB abstraction stream; leave it False for
                 every raw-data stream, or LogReg and ss-STraTS silently read a
                 derived clinical judgement they could not have computed.

        Args:
            table       (pd.DataFrame): Temporal table in canonical columns.
            patient_ids (array-like|None): Restrict the injected rows.
            kb_events   (bool):         Keep non-target KB events in this stream.
            verbose     (bool):         Print what was swapped.

        Returns:
            pd.DataFrame: The table with harmonised event rows.
        """
        # Idempotent: the grid hands the same raw table to several builders, and
        # re-splicing a multi-million-row frame per cell is pure waste.
        if table.attrs.get("kineret_events_harmonised") == bool(kb_events):
            return table

        all_spellings = {s for ss in self.outcome_aliases.values() for s in ss}
        # Plus whatever the raw file still calls a target on disk. Without this
        # a stream read fresh from the source carries BOTH the ETL's own event
        # rows and the injected canonical ones -- a duplicate signal the KB arm
        # never sees, which is exactly the asymmetry harmonisation exists to
        # remove.
        all_spellings |= set(self.replaced_spellings)
        is_event = table["ConceptName"].isin(all_spellings)

        # Non-target KB events: dropped unless this stream is a KB stream.
        drop_kb = (not kb_events) and getattr(C, "KB_EVENTS_FOR_KB_ARMS_ONLY", True)
        if drop_kb and self.kb_event_names:
            is_kb = table["ConceptName"].isin(set(self.kb_event_names))
            if verbose and int(is_kb.sum()):
                print(f"[Cohort] harmonise_events: removed {int(is_kb.sum()):,} "
                      f"non-target KB event rows "
                      f"({len(self.kb_event_names)} concepts) -- this stream "
                      f"feeds a model with no knowledge base.")
            is_event = is_event | is_kb
        stripped = table[~is_event].copy()

        if not C.EVENTS_AS_INPUTS:
            if verbose:
                print(f"[Cohort] harmonise_events: removed {int(is_event.sum()):,} "
                      f"outcome rows; EVENTS_AS_INPUTS is off so none re-injected.")
            stripped.attrs["kineret_events_harmonised"] = bool(kb_events)
            return stripped

        pids = patient_ids if patient_ids is not None else table["PatientId"].unique()
        injected = self.canonical_event_rows(pids)
        # Match the source table's columns so downstream code is unaffected.
        for col in table.columns:
            if col not in injected.columns:
                # Match the source dtype: an all-NA object column makes pandas
                # infer the concat result's dtype from the wrong side.
                injected[col] = pd.Series(index=injected.index,
                                          dtype=table[col].dtype)
        injected = injected[table.columns]

        out = pd.concat([stripped, injected], ignore_index=True)
        if verbose:
            print(f"[Cohort] harmonise_events: {int(is_event.sum()):,} outcome rows "
                  f"-> {len(injected):,} canonical rows.")
        out = out.sort_values(["PatientId", "StartDateTime"]).reset_index(drop=True)
        out.attrs["kineret_events_harmonised"] = bool(kb_events)
        return out

    def leakage_blocklist(self, kb_events: bool = False) -> set:
        """
        Purpose: ConceptName spellings that must never enter any input stream.
        Method:  Always blocks the structural RELEASE / DEATH terminus markers,
                 which announce the end of the admission.

                 Outcome events are blocked only when `EVENTS_AS_INPUTS` is
                 False. When it is True they are instead HARMONISED -- stripped
                 and replaced with the canonical set by `harmonise_events` --
                 and then clipped to the observation window by each model's own
                 truncation. That is not leakage: the label window starts
                 strictly after the input window, so an event the model can see
                 is observed history, never the thing it is being asked to
                 predict. INTERVenE's abstraction stream carries these natively,
                 so withholding them from the raw-data models would bias the
                 comparison rather than protect it.

                 Never blocked either way: the measurements outcomes are derived
                 from (glucose, creatinine, ...). Those are legitimate
                 predictors, and dropping them would penalise exactly the models
                 that read raw data.

        Args:
            kb_events (bool): True for a knowledge-base stream, which keeps the
                              Mediator's non-target `*_EVENT` concepts. False
                              for every raw-data stream, which must not see them.

        Returns:
            set[str]: Blocked ConceptName values.
        """
        blocked = set()
        for token in (C.RELEASE_TOKEN, C.DEATH_TOKEN):
            blocked.update(self.structural_aliases.get(token, []))
        if not C.EVENTS_AS_INPUTS:
            for name in self.outcome_names:
                blocked.update(self.outcome_aliases.get(name, []))
        # Non-target KB conclusions are withheld from the arms that have no KB.
        # `harmonise_events` already strips them; blocking them here as well
        # keeps any feature builder that filters on this list honest.
        if not kb_events and getattr(C, "KB_EVENTS_FOR_KB_ARMS_ONLY", True):
            blocked.update(self.kb_event_names)
        if not C.EVENTS_AS_INPUTS:
            blocked.update(self.replaced_spellings)
        return blocked

    # ---------------------------------------------------------------- storage

    def save(self, path: str = None) -> str:
        """Purpose: Persist the cohort so every training script reads one file."""
        path = path or paths.COHORT_PKL
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        print(f"[Cohort] Saved -> {path}")
        return path

    @staticmethod
    def load(path: str = None) -> "Cohort":
        """Purpose: Reload the shared cohort artefact."""
        path = path or paths.COHORT_PKL
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[Cohort] {path} not found -- run the cohort cell in notebooks/benchmark.ipynb first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# Builder internals
# ===========================================================================

def _resolve_aliases(alias_map: dict, *concept_pools) -> dict:
    """
    Purpose: Map each canonical name to the ConceptName spellings that actually
             occur in the data.
    Method:  Union the observed-concept pools, then fullmatch every alias regex
             against it.

    Args:
        alias_map     (dict): canonical -> regex body.
        concept_pools       : iterables of observed ConceptName values.

    Returns:
        dict[str, list[str]]: canonical -> observed spellings (possibly empty).
    """
    observed = set()
    for pool in concept_pools:
        observed.update(str(c) for c in pool)
    return {canon: match_concepts(observed, rx) for canon, rx in alias_map.items()}


def _resolve_context_identity(context: pd.DataFrame, verbose=True):
    """
    Purpose: Put the context table on the same key as the temporal tables, and
             recover the person behind each admission.
    Method:  The Mediator carries exactly one id column, so its temporal exports
             key on the ADMISSION -- `PatientId` there is really a visit id. The
             context table did not go through that constraint and keys on
             `CONTEXT_ID_COLUMN` (`VisitId`), carrying `CONTEXT_PERSON_COLUMN`
             (`person_id`) alongside: the human, who may appear several times.

             The id column is renamed to PatientId so the join works. The person
             column is REMOVED from the feature block -- it is an identifier,
             and a model handed a person id will happily memorise people -- but
             returned separately so the split can be grouped by it.

    Args:
        context (pd.DataFrame): The context table as loaded.
        verbose (bool):         Print what was resolved.

    Returns:
        tuple[pd.DataFrame, dict]: (context keyed on PatientId, visit -> person).
    """
    say = print if verbose else (lambda *a, **kw: None)
    lower = {str(c).lower(): c for c in context.columns}

    # The CONFIGURED id column wins even when a `PatientId` column also exists.
    # A context export can carry both, and the one named PatientId is often the
    # person rather than the visit -- joining on it would attach one person's
    # covariates to every admission they ever had.
    configured = lower.get(str(getattr(C, "CONTEXT_ID_COLUMN", "VisitId")).lower())
    if configured is not None and configured != "PatientId":
        if "PatientId" in context.columns:
            say(f"[Cohort] context has both {configured!r} and 'PatientId'; "
                f"joining on {configured!r} (the admission) and dropping the other.")
            context = context.drop(columns=["PatientId"])
        context = context.rename(columns={configured: "PatientId"})
        say(f"[Cohort] context key: {configured!r} -> PatientId "
            f"(the temporal tables key on the admission, not the person).")
    elif "PatientId" not in context.columns:
        for candidate in ("visitid", "visit_id", "patient_id"):
            column = lower.get(candidate)
            if column is not None:
                context = context.rename(columns={column: "PatientId"})
                say(f"[Cohort] context key: {column!r} -> PatientId.")
                break
    if "PatientId" not in context.columns:
        raise ValueError(
            f"[Cohort] context table has no id column. Looked for 'PatientId' and "
            f"CONTEXT_ID_COLUMN={getattr(C, 'CONTEXT_ID_COLUMN', 'VisitId')!r}; "
            f"found {list(context.columns)}. Set context_id_column in the "
            f"notebook's configure_study call.")

    person_of_patient = {}
    person_column = lower.get(str(getattr(C, "CONTEXT_PERSON_COLUMN",
                                          "person_id")).lower())
    if person_column and person_column in context.columns:
        pairs = context[["PatientId", person_column]].dropna()
        person_of_patient = dict(zip(pairs["PatientId"], pairs[person_column]))
        n_people = pairs[person_column].nunique()
        n_visits = pairs["PatientId"].nunique()
        context = context.drop(columns=[person_column])
        say(f"[Cohort] {n_visits:,} admissions belong to {n_people:,} people "
            f"({n_visits / max(n_people, 1):.2f} admissions per person). "
            f"{person_column!r} dropped from the feature block (identifier).")
    return context, person_of_patient


def _admission_window(*tables):
    """
    Purpose: The admission start/end each export was cut on, per admission.
    Method:  Read `AdmissionStart` / `AdmissionEnd` wherever a table carries
             them and take the earliest start and latest end per PatientId
             (= VisitId after re-keying). These columns are what the extract was
             actually built around, so they beat inferring the window from an
             ADMISSION or RELEASE concept row that may simply not have been
             emitted for a given admission.

    Args:
        *tables (pd.DataFrame): Temporal tables, canonicalised.

    Returns:
        tuple[pd.Series|None, pd.Series|None]: (start, end), None when absent.
    """
    starts, ends = [], []
    for table in tables:
        if table is None:
            continue
        if "AdmissionStart" in table.columns:
            starts.append(table.groupby("PatientId")["AdmissionStart"].min())
        if "AdmissionEnd" in table.columns:
            ends.append(table.groupby("PatientId")["AdmissionEnd"].max())
    start = pd.concat(starts).groupby(level=0).min() if starts else None
    end = pd.concat(ends).groupby(level=0).max() if ends else None
    return start, end


def _drop_irrelevant_rows(table: pd.DataFrame, label: str, verbose=True):
    """
    Purpose: Drop rows the export flagged as belonging to another admission.
    Method:  Remove rows whose `relevant_admission` is explicitly False. Blank
             values are KEPT -- the ADMISSION row itself carries no flag, and
             discarding blanks would delete the anchor.

    Args:
        table   (pd.DataFrame): Temporal table.
        label   (str):          Name for the log line.
        verbose (bool):         Print what was dropped.

    Returns:
        pd.DataFrame: Filtered table.
    """
    column = getattr(C, "RAW_RELEVANT_ADMISSION_COLUMN", None)
    if not column or column not in table.columns:
        return table
    flag = table[column]
    if flag.dtype == object:
        flag = flag.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
    else:
        flag = flag.astype("boolean")
    drop = flag.eq(False).fillna(False)
    if int(drop.sum()) and verbose:
        print(f"[Cohort] {label}: dropping {int(drop.sum()):,} row(s) flagged "
              f"{column}=False (they belong to another admission).")
    return table[~drop].copy()


def _admission_times(raw: pd.DataFrame, abstract: pd.DataFrame,
                     admission_aliases: list) -> pd.Series:
    """
    Purpose: Anchor every patient's clock at their ADMISSION.
    Method:  Prefer an explicit ADMISSION row; fall back to the earliest
             timestamp seen for that patient across both tables, so a cohort
             missing the framing token still gets a usable t=0.

    Args:
        raw               (pd.DataFrame): Raw temporal table.
        abstract          (pd.DataFrame): Mediator output table.
        admission_aliases (list[str]):    Observed ADMISSION spellings.

    Returns:
        pd.Series: PatientId -> admission timestamp.
    """
    both = pd.concat([raw[["PatientId", "ConceptName", "StartDateTime"]],
                      abstract[["PatientId", "ConceptName", "StartDateTime"]]],
                     ignore_index=True)
    earliest = both.groupby("PatientId")["StartDateTime"].min()

    adm_rows = both[both["ConceptName"].isin(admission_aliases)]
    if len(adm_rows):
        explicit = adm_rows.groupby("PatientId")["StartDateTime"].min()
        n_missing = len(earliest.index.difference(explicit.index))
        if n_missing:
            print(f"[Cohort] {n_missing} patients have no ADMISSION row -- "
                  f"anchoring them at their earliest event instead.")
        return explicit.reindex(earliest.index).fillna(earliest)

    print("[Cohort] No ADMISSION rows in either table -- "
          "anchoring every patient at their earliest event.")
    return earliest


def _hours_from_admission(df: pd.DataFrame, admission: pd.Series) -> pd.Series:
    """Purpose: Elapsed hours from each patient's admission to each event row."""
    anchor = df["PatientId"].map(admission)
    return (df["StartDateTime"] - anchor).dt.total_seconds() / 3600.0


def date_range_bounds():
    """
    Purpose: The study window as timestamps.
    Method:  Reads `DATE_RANGE_START` / `DATE_RANGE_END` from the shared config.
             Either may be None to leave that side open.

    Returns:
        tuple[pd.Timestamp|None, pd.Timestamp|None]: (start, end).
    """
    start = pd.Timestamp(C.DATE_RANGE_START) if C.DATE_RANGE_START else None
    # An end date is inclusive of that whole day.
    end = (pd.Timestamp(C.DATE_RANGE_END) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
           if C.DATE_RANGE_END else None)
    return start, end


def clip_to_date_range(df: pd.DataFrame, label: str, column="StartDateTime",
                       verbose=True) -> pd.DataFrame:
    """
    Purpose: Trim one table to the study window.
    Method:  Row-level clip on the timestamp column. Called once per source
             table inside `build_cohort`, and nowhere else -- the whole point is
             that the range lives in exactly one place, so inputs, labels,
             length-of-stay and the QA aggregation can never drift apart.

    Args:
        df      (pd.DataFrame): Table to trim.
        label   (str):          Name for the log line.
        column  (str):          Timestamp column to filter on.
        verbose (bool):         Print how many rows were dropped.

    Returns:
        pd.DataFrame: Rows inside the window.
    """
    start, end = date_range_bounds()
    if start is None and end is None:
        return df

    before = len(df)
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df[column] >= start
    if end is not None:
        keep &= df[column] <= end
    out = df[keep].copy()
    if verbose and len(out) != before:
        say_start = start.date() if start is not None else "-inf"
        say_end = end.date() if end is not None else "+inf"
        print(f"[Cohort] Date range [{say_start} .. {say_end}] on {label}: "
              f"{before:,} -> {len(out):,} rows.")
    return out


def _dedupe_occurrences(df: pd.DataFrame, tolerance_h: float) -> pd.DataFrame:
    """
    Purpose: Collapse restatements of the same clinical event into one row.
    Method:  Within (PatientId, outcome), sort by time and start a new
             occurrence only when the gap from the previous one exceeds
             `tolerance_h`. Abstraction intervals and repeated measurements both
             restate the same event several times; without this the two source
             files could never agree on a count.

    Args:
        df          (pd.DataFrame): ['PatientId', 'outcome', 'hours', ...].
        tolerance_h (float):        Merge window in hours.

    Returns:
        pd.DataFrame: One row per distinct occurrence, earliest time kept.
    """
    if len(df) == 0:
        return df
    out = df.sort_values(["PatientId", "outcome", "hours"]).reset_index(drop=True)
    prev = out.groupby(["PatientId", "outcome"])["hours"].shift()
    gap = out["hours"] - prev
    out["_occ"] = (gap.isna() | (gap > tolerance_h)).cumsum()
    out = out.groupby("_occ", as_index=False).first().drop(columns="_occ")
    return out.sort_values(["PatientId", "outcome", "hours"]).reset_index(drop=True)


def _default_alias_regex(name: str) -> str:
    """
    Purpose: Spelling pattern for a target the alias table does not cover.
    Method:  Match the canonical name with the Mediator's `_EVENT` suffix
             optional, so a target named in the notebook but absent from
             `OUTCOME_ALIAS_REGEX` still resolves against both files instead of
             silently matching nothing.

    Args:
        name (str): Canonical target name.

    Returns:
        str: Regex, fullmatched case-insensitively against ConceptName.
    """
    stem = re.escape(name[:-len("_EVENT")] if name.upper().endswith("_EVENT") else name)
    return rf"{stem}(?:_EVENT)?"


def _assert_rule_agreement(raw_events: pd.DataFrame, abstract_events: pd.DataFrame,
                           targets: list, verbose=True):
    """
    Purpose: Require the derived raw view to reproduce the Mediator exactly.
    Method:  Compare the two event sets occurrence by occurrence and print the
             per-target agreement. The Mediator produced `mediator_output.csv`
             from `mediator_input.csv` using the very rules `kineret.raw_events`
             executes, so a mismatch is a defect in the reproduction -- never a
             disagreement between two sources -- and it must not be absorbed by
             the intersection, which would silently delete real labels.

             `raw_only` events were invented here; `med_only` events were missed.
             Both name the rule to go and look at.

    Args:
        raw_events      (pd.DataFrame): Derived raw view, pre-alignment.
        abstract_events (pd.DataFrame): The Mediator's view, pre-alignment.
        targets         (list):         Targets to check.
        verbose         (bool):         Print the table.

    Returns:
        dict: outcome -> agreement fraction.

    Raises:
        RuntimeError: Agreement below `RULE_AGREEMENT_MIN` on any target, unless
                      `raise_on_rule_mismatch` is off.
    """
    from kineret.raw_events import agreement_report
    say = print if verbose else (lambda *a, **kw: None)
    report = agreement_report(raw_events, abstract_events, targets)

    say("[Cohort] Rule agreement -- the derived view vs the Mediator's own "
        "(both should be identical):")
    say(f"    {'outcome':<34} {'raw':>9} {'mediator':>9} {'matched':>9} "
        f"{'raw_only':>9} {'med_only':>9}  agree")
    for _, row in report.iterrows():
        say(f"    {row['outcome']:<34} {row['raw']:>9,} {row['mediator']:>9,} "
            f"{row['matched']:>9,} {row['raw_only']:>9,} {row['med_only']:>9,}  "
            f"{row['agreement']:6.2%}")

    threshold = float(getattr(C, "RULE_AGREEMENT_MIN", 1.0))
    failed = report[report["agreement"] < threshold]
    if len(failed) and getattr(C, "RAISE_ON_RULE_MISMATCH", True):
        worst = failed.iloc[0]
        raise RuntimeError(
            f"[Cohort] {len(failed)} target(s) do not reproduce the Mediator "
            f"exactly (worst: {worst['outcome']} at {worst['agreement']:.2%}, "
            f"{worst['raw_only']:,} invented / {worst['med_only']:,} missed).\n"
            f"  The Mediator derived mediator_output.csv from mediator_input.csv "
            f"with the rules in kineret/config/event_rules.json, so this is a "
            f"reproduction defect, not a source disagreement.\n"
            f"  Inspect the rule:  python -m kineret.mediator_rules <kb-path>\n"
            f"  Check it early:    from kineret.raw_events import validate_rules\n"
            f"  To proceed anyway (labels WILL be filtered by the intersection): "
            f"configure_study(raise_on_rule_mismatch=False).")
    if len(failed):
        say(f"[Cohort] WARNING: {len(failed)} target(s) below "
            f"{threshold:.0%} agreement; the intersection will drop the "
            f"difference. Results for them are not trustworthy.")
    return dict(zip(report["outcome"], report["agreement"]))


def _align_event_times(raw_events: pd.DataFrame, abstract_events: pd.DataFrame,
                       tolerance_min: float, verbose=True):
    """
    Purpose: Keep only the occurrences the two files agree on, to the instant.
    Method:  For each outcome BOTH files carry, an occurrence survives in either
             file only if the other file records the same outcome for the same
             patient within `tolerance_min` minutes. Outcomes only one file
             carries pass through untouched -- intersecting a Mediator-derived
             complication against an empty raw view would delete the target.

             Run BEFORE the clinical dedupe: deduping first would pick one
             representative timestamp per file independently, and two files
             could then hold non-matching representatives of an event they
             actually agree on.

             By the time this runs the two views should already be identical:
             `raw_events` executes the Mediator's own rules, so alignment is a
             final assertion rather than a filter. Attrition here means a rule
             is wrong, and `build_cohort` stops on it.

    Args:
        raw_events      (pd.DataFrame): ['PatientId','outcome','hours',...].
        abstract_events (pd.DataFrame): Same shape, Mediator's view.
        tolerance_min   (float):        Agreement window in minutes.
        verbose         (bool):         Print per-outcome attrition.

    Returns:
        tuple: (raw_aligned, abstract_aligned, audit_rows).
    """
    say = print if verbose else (lambda *a, **kw: None)
    tolerance_h = float(tolerance_min) / 60.0
    shared = sorted(set(raw_events["outcome"]) & set(abstract_events["outcome"]))
    if not shared:
        return raw_events, abstract_events, []

    keep_raw, keep_abs, rows = [], [], []
    for outcome in shared:
        r = raw_events[raw_events["outcome"] == outcome]
        a = abstract_events[abstract_events["outcome"] == outcome]
        # Cross join within (PatientId), then keep pairs inside the tolerance.
        pairs = r[["PatientId", "hours"]].merge(
            a[["PatientId", "hours"]], on="PatientId", suffixes=("_r", "_a"))
        pairs = pairs[(pairs["hours_r"] - pairs["hours_a"]).abs() <= tolerance_h]
        matched_r = set(map(tuple, pairs[["PatientId", "hours_r"]].to_numpy()))
        matched_a = set(map(tuple, pairs[["PatientId", "hours_a"]].to_numpy()))
        r_keep = r[[tuple(x) in matched_r
                    for x in r[["PatientId", "hours"]].to_numpy()]]
        a_keep = a[[tuple(x) in matched_a
                    for x in a[["PatientId", "hours"]].to_numpy()]]
        keep_raw.append(r_keep)
        keep_abs.append(a_keep)
        rows.append({"outcome": outcome, "raw_before": len(r), "raw_after": len(r_keep),
                     "mediator_before": len(a), "mediator_after": len(a_keep)})
        if len(r) != len(r_keep) or len(a) != len(a_keep):
            say(f"    {outcome:<34} raw {len(r):>6} -> {len(r_keep):<6} "
                f"mediator {len(a):>6} -> {len(a_keep):<6}")

    # Outcomes only one file carries are passed through unchanged.
    raw_only = raw_events[~raw_events["outcome"].isin(shared)]
    abs_only = abstract_events[~abstract_events["outcome"].isin(shared)]
    raw_out = pd.concat(keep_raw + [raw_only], ignore_index=True)
    abs_out = pd.concat(keep_abs + [abs_only], ignore_index=True)
    return raw_out, abs_out, rows


def _per_file_events(table: pd.DataFrame, aliases: dict, source: str,
                     tolerance_h: float, dedupe: bool = True) -> pd.DataFrame:
    """
    Purpose: Extract one file's view of every outcome, in canonical form.
    Method:  Map each observed spelling back to its canonical name, drop
             pre-admission rows, then collapse restatements.

    Args:
        table       (pd.DataFrame): Temporal table carrying an `hours` column.
        aliases     (dict):         canonical -> observed spellings.
        source      (str):          Provenance tag.
        tolerance_h (float):        Occurrence merge window.

    Returns:
        pd.DataFrame: ['PatientId', 'outcome', 'hours', 'source'].
    """
    spelling_to_canon = {s: canon for canon, ss in aliases.items() for s in ss}
    hit = table[table["ConceptName"].isin(spelling_to_canon)]
    if len(hit) == 0:
        return pd.DataFrame(columns=["PatientId", "outcome", "hours", "source"])
    out = pd.DataFrame({
        "PatientId": hit["PatientId"].to_numpy(),
        "outcome": hit["ConceptName"].map(spelling_to_canon).to_numpy(),
        "hours": hit["hours"].to_numpy(),
        "source": source,
    })
    out = out[out["hours"] >= 0.0]
    # Deferred when cross-file alignment runs first: see `_align_event_times`.
    return _dedupe_occurrences(out, tolerance_h) if dedupe else out.reset_index(drop=True)


def _reconcile_events(raw_events: pd.DataFrame, abstract_events: pd.DataFrame,
                      mode: str, tolerance_h: float, verbose=True):
    """
    Purpose: Produce ONE canonical event table both source files will be held to.
    Method:  Each file contributes its own view; `mode` decides how they combine.
             Whatever the mode, the result is the single truth used for labels
             AND re-injected into every model's input stream, so afterwards the
             two files have identical event support by construction rather than
             by luck.

             "mediator" / "raw" prefer one file per outcome and fall back to the
             other when the preferred one has nothing for it (an outcome the KB
             does not re-derive, or one the ETL never pre-computed). "union"
             takes an occurrence in either. "intersect" keeps only occurrences
             both files agree on within `tolerance_h`.

    Args:
        raw_events      (pd.DataFrame): Raw file's view.
        abstract_events (pd.DataFrame): Mediator file's view.
        mode            (str):          EVENT_SOURCE.
        tolerance_h     (float):        Agreement window.
        verbose         (bool):         Print the per-outcome reconciliation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (canonical events, audit table).
    """
    say = print if verbose else (lambda *a, **kw: None)
    outcomes = sorted(set(raw_events["outcome"]) | set(abstract_events["outcome"]))

    chosen, audit = [], []
    for outcome in outcomes:
        r = raw_events[raw_events["outcome"] == outcome]
        a = abstract_events[abstract_events["outcome"] == outcome]

        if mode == "union":
            picked = _dedupe_occurrences(pd.concat([a, r], ignore_index=True), tolerance_h)
            source = "union"
        elif mode == "intersect":
            merged = a.merge(r, on=["PatientId", "outcome"], suffixes=("", "_r"))
            merged = merged[(merged["hours"] - merged["hours_r"]).abs() <= tolerance_h]
            picked = _dedupe_occurrences(
                merged[["PatientId", "outcome", "hours"]].assign(source="intersect"),
                tolerance_h)
            source = "intersect"
        else:
            preferred, fallback = ((a, r) if mode == "mediator" else (r, a))
            picked = preferred if len(preferred) else fallback
            source = ("mediator" if mode == "mediator" else "raw") if len(preferred) \
                else ("raw" if mode == "mediator" else "mediator")

        if len(picked):
            chosen.append(picked.assign(source=source))

        audit.append({
            "outcome": outcome,
            "n_raw": len(r), "n_mediator": len(a),
            "n_canonical": len(picked), "source": source,
            "patients_raw": r["PatientId"].nunique(),
            "patients_mediator": a["PatientId"].nunique(),
            "patients_canonical": picked["PatientId"].nunique() if len(picked) else 0,
        })

    audit_df = pd.DataFrame(audit)
    disagreements = audit_df[audit_df["n_raw"] != audit_df["n_mediator"]]
    if len(disagreements):
        say(f"[Cohort] The two files disagree on {len(disagreements)} outcome(s); "
            f"EVENT_SOURCE={mode!r} resolves each, and the canonical table is "
            f"then injected into BOTH input streams so support matches:")
        for _, row in disagreements.iterrows():
            say(f"    {row['outcome']:<34} raw={row['n_raw']:>5}  "
                f"mediator={row['n_mediator']:>5}  -> {row['n_canonical']:>5} "
                f"({row['source']})")

    if not chosen:
        empty = pd.DataFrame(columns=["PatientId", "outcome", "hours", "source"])
        return empty, audit_df
    events = pd.concat(chosen, ignore_index=True)
    return (events.sort_values(["PatientId", "outcome", "hours"]).reset_index(drop=True),
            audit_df)


def _build_qa_block(qa_df: pd.DataFrame, admission: pd.Series,
                    patient_ids: np.ndarray, k: int) -> pd.DataFrame:
    """
    Purpose: Aggregate treatment-quality compliance into static features for one K.
    Method:  Mean ComplianceScore per (PatientId, PatternName) over [0, k*24]
             hours from admission, pivoted wide and reindexed to the canonical
             pattern set taken from the FULL qa file -- so the column layout is
             identical across K, across splits and across models. Patients with
             no QA row in the window are zero-filled.

    Args:
        qa_df       (pd.DataFrame): QA table with PatternName / ComplianceScore.
        admission   (pd.Series):    PatientId -> admission timestamp.
        patient_ids (np.ndarray):   Cohort patients.
        k           (int):          Context-window length in days.

    Returns:
        pd.DataFrame: PatientId-indexed QA_<pattern> columns, float32.
    """
    canonical = sorted(qa_df[C.QA_PATTERN_COLUMN].dropna().astype(str).unique())
    aggregations = list(C.QA_AGGREGATIONS) or ["mean"]
    single = len(aggregations) == 1
    index = pd.Index(patient_ids, name="PatientId")

    def column_name(pattern, agg):
        """A lone `mean` keeps the plain `QA_<pattern>` name used in the thesis code."""
        return (f"{C.QA_COLUMN_PREFIX}{pattern}" if single
                else f"{C.QA_COLUMN_PREFIX}{pattern}_{agg}")

    cols = [column_name(p, a) for a in aggregations for p in canonical]

    qa = qa_df.copy()
    qa["hours"] = _hours_from_admission(qa, admission)
    qa = qa[(qa["hours"] >= 0.0) & (qa["hours"] <= k * 24.0)]
    qa[C.QA_SCORE_COLUMN] = pd.to_numeric(qa[C.QA_SCORE_COLUMN], errors="coerce")
    qa = qa.dropna(subset=[C.QA_SCORE_COLUMN])

    if len(qa) == 0:
        return pd.DataFrame(0.0, index=index, columns=cols).astype("float32")

    # `last` needs chronological order within each (patient, pattern).
    qa = qa.sort_values(["PatientId", C.QA_PATTERN_COLUMN, "hours"])
    grouped = qa.groupby(["PatientId", C.QA_PATTERN_COLUMN])[C.QA_SCORE_COLUMN]

    blocks = []
    for agg in aggregations:
        series = grouped.last() if agg == "last" else grouped.agg(agg)
        wide = series.unstack(C.QA_PATTERN_COLUMN)
        wide = wide.reindex(columns=canonical, fill_value=0.0)
        wide.columns = [column_name(p, agg) for p in canonical]
        blocks.append(wide.reindex(index, fill_value=0.0))

    out = pd.concat(blocks, axis=1).reindex(columns=cols)
    # A patient with no QA row in the window scores 0 -- "no recorded
    # compliance", which is the same thing the models see for an absent pattern.
    return out.fillna(0.0).astype("float32")


# ===========================================================================
# Public builder
# ===========================================================================

def build_cohort(verbose: bool = True) -> Cohort:
    """
    Purpose: Read the four source tables and produce every shared artefact.
    Method:  (1) load + canonicalise, (2) anchor each patient at admission,
             (3) resolve outcome/structural aliases across both spellings,
             (4) extract outcome occurrences and length-of-stay,
             (5) apply the cohort-length filter, (6) freeze the patient split,
             (7) support-filter outcomes on TRAIN at the tightest label window,
             (8) precompute the QA block for every K in the grid.

    Args:
        verbose (bool): Print a provenance report as it goes.

    Returns:
        Cohort: The populated artefact (caller decides whether to save).
    """
    say = print if verbose else (lambda *a, **kw: None)
    paths.ensure_dirs()

    # --- 1. load ---------------------------------------------------------
    say(f"[Cohort] raw temporal : {paths.RAW_TEMPORAL_FILE}")
    say(f"[Cohort] mediator out : {paths.ABSTRACT_FILE}")
    say(f"[Cohort] context      : {paths.CONTEXT_FILE}")
    raw      = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
    abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
    if getattr(C, "DROP_IRRELEVANT_ADMISSION_ROWS", False):
        raw = _drop_irrelevant_rows(raw, "raw temporal", verbose=verbose)
        abstract = _drop_irrelevant_rows(abstract, "mediator output", verbose=verbose)

    context  = load_table(paths.CONTEXT_FILE)
    context, person_of_patient = _resolve_context_identity(context, verbose=verbose)
    say(f"[Cohort] rows: raw={len(raw):,}  abstract={len(abstract):,}  context={len(context):,}")

    qa_df = None
    if os.path.exists(paths.QA_FILE):
        qa_df = load_table(paths.QA_FILE).rename(columns={"StartTime": "StartDateTime"})
        qa_df["StartDateTime"] = pd.to_datetime(
            qa_df["StartDateTime"], utc=True, errors="coerce").dt.tz_convert(None)
        missing = {"PatientId", C.QA_PATTERN_COLUMN, C.QA_SCORE_COLUMN} - set(qa_df.columns)
        if missing:
            raise ValueError(f"[Cohort] QA file missing columns: {sorted(missing)}")
        say(f"[Cohort] qa           : {paths.QA_FILE} ({len(qa_df):,} rows)")
    else:
        say(f"[Cohort] qa           : ABSENT ({paths.QA_FILE}) -- QA arm unavailable.")

    # --- 1a. raw-side event derivation -----------------------------------
    # BEFORE the study window is applied, and before anything else touches the
    # tables. Two of the Mediator's rules read a reading's neighbours -- the
    # `sustained` gate looks at the previous measurement, and the creatinine
    # ratio divides by the admission's FIRST one -- so deriving from a truncated
    # series silently changes both. The Mediator saw the whole admission; so
    # must this. The derived rows are then clipped by the study window along
    # with everything else, exactly as the Mediator's own events are.
    replaced_spellings = []
    if getattr(C, "DERIVE_RAW_EVENTS", False):
        from kineret.raw_events import derive_raw_events, target_spellings
        say("[Cohort] Rebuilding the raw file's own event view from the "
            "Mediator's compiled rules (on the full, unclipped series):")
        new_rows = derive_raw_events(raw, targets=list(C.OUTCOMES), verbose=verbose)
        if len(new_rows):
            # Whatever the raw file already spelled for these targets is
            # replaced: the rules produce the complete view, and keeping both
            # would double any target the ETL happened to pre-compute.
            derived_targets = sorted(new_rows["ConceptName"].unique())
            stale = target_spellings(derived_targets, raw["ConceptName"].unique())
            replaced_spellings = sorted(stale - set(derived_targets))
            raw = raw[~raw["ConceptName"].isin(stale)]
            raw = pd.concat([raw, new_rows], ignore_index=True)
            say(f"[Cohort]   {len(new_rows):,} derived rows replace "
                f"{len(stale)} raw spelling(s) of {len(derived_targets)} target(s).")
            if replaced_spellings:
                say(f"[Cohort]   the raw file's own spellings "
                    f"{replaced_spellings} are stripped from every input "
                    f"stream, so no arm sees them alongside the canonical event.")

    # --- 1b. date range --------------------------------------------------
    # The ONLY place the study window is applied. Every table is clipped here,
    # so inputs, labels, length-of-stay and QA can never disagree about which
    # period the study covers.
    range_start, range_end = date_range_bounds()
    if range_start is not None or range_end is not None:
        say(f"[Cohort] Study window: "
            f"{C.DATE_RANGE_START or '-inf'} .. {C.DATE_RANGE_END or '+inf'}")
        raw = clip_to_date_range(raw, "raw temporal", verbose=verbose)
        abstract = clip_to_date_range(abstract, "mediator output", verbose=verbose)
        if qa_df is not None:
            qa_df = clip_to_date_range(qa_df, "qa", verbose=verbose)
        if raw.empty and abstract.empty:
            raise RuntimeError(
                f"[Cohort] The study window "
                f"[{C.DATE_RANGE_START} .. {C.DATE_RANGE_END}] contains no rows. "
                f"Check DATE_RANGE_* in kineret/config/data_config.py against the "
                f"actual timestamps in your source tables."
            )

    # --- 2. anchor -------------------------------------------------------
    structural = _resolve_aliases(C.STRUCTURAL_REGEX,
                                  raw["ConceptName"].unique(),
                                  abstract["ConceptName"].unique())
    window_start, window_end = (_admission_window(raw, abstract)
                                if getattr(C, "ADMISSION_WINDOW_FROM_COLUMNS", False)
                                else (None, None))
    # The raw export names the person alongside the visit, so the split can be
    # grouped by person even when the context table omits it.
    if "PersonId" in raw.columns:
        pairs = raw[["PatientId", "PersonId"]].dropna().drop_duplicates("PatientId")
        from_raw = dict(zip(pairs["PatientId"], pairs["PersonId"]))
        added = len(set(from_raw) - set(person_of_patient))
        if added:
            say(f"[Cohort] {added:,} admission(s) got their person from the raw "
                f"export's PersonId column.")
        from_raw.update(person_of_patient)      # context wins where both exist
        person_of_patient = from_raw

    admission = _admission_times(raw, abstract, structural[C.ADMISSION_TOKEN])
    if window_start is not None:
        covered = window_start.reindex(admission.index)
        n_used = int(covered.notna().sum())
        admission = covered.fillna(admission)
        say(f"[Cohort] Anchored {n_used:,} admission(s) on the export's own "
            f"AdmissionStart column; {len(admission) - n_used:,} fell back to "
            f"the ADMISSION row / earliest event.")
    raw["hours"] = _hours_from_admission(raw, admission)
    abstract["hours"] = _hours_from_admission(abstract, admission)

    # Their content is now folded into `admission`, `los` and the person map.
    # On a 23-million-row export these columns are hundreds of MB that every
    # downstream copy would otherwise carry.
    spent = ["PersonId", "AdmissionStart", "AdmissionEnd",
             getattr(C, "RAW_RELEVANT_ADMISSION_COLUMN", "relevant_admission")]
    raw = raw.drop(columns=[c for c in spent if c in raw.columns])
    abstract = abstract.drop(columns=[c for c in spent if c in abstract.columns])

    # --- 3. outcome aliases ---------------------------------------------
    # The TARGET LIST is the authority on what a target is. `OUTCOME_ALIAS_REGEX`
    # only says how each name is spelled across the two files -- it is a
    # spelling table, not a target list. Deriving the targets from it (as this
    # used to) meant `targets=[...]` in the notebook was silently ignored and
    # every regex it happened to contain got a prediction head.
    alias_map = {name: C.OUTCOME_ALIAS_REGEX.get(name, _default_alias_regex(name))
                 for name in C.OUTCOMES}
    if C.AUTO_DISCOVER_OUTCOMES:
        already = {a.upper() for lst in
                   _resolve_aliases(alias_map, abstract["ConceptName"].unique()).values()
                   for a in lst}
        reserved = {C.ADMISSION_TOKEN.upper(), C.RELEASE_TOKEN.upper()}
        extra = sorted({str(c) for c in abstract["ConceptName"].unique()
                        if str(c).upper().endswith("_EVENT")
                        and str(c).upper() not in already
                        and str(c).upper() not in reserved})
        for name in extra:
            alias_map.setdefault(name, re.escape(name))
        if extra:
            say(f"[Cohort] Auto-discovered {len(extra)} extra *_EVENT outcomes: {extra}")

    outcome_aliases = _resolve_aliases(alias_map,
                                       raw["ConceptName"].unique(),
                                       abstract["ConceptName"].unique())
    unmatched = [c for c, s in outcome_aliases.items() if not s]
    if unmatched:
        say(f"[Cohort] Configured targets absent from both tables: {unmatched}")

    # Every other `*_EVENT` concept in the Mediator output: a KB conclusion that
    # is not a prediction target. Recorded so the input streams of the non-KB
    # arms can be stripped of them -- see `Cohort.harmonise_events`.
    target_spellings = {sp.upper() for ss in outcome_aliases.values() for sp in ss}
    reserved = {C.ADMISSION_TOKEN.upper(), C.RELEASE_TOKEN.upper(),
                C.DEATH_TOKEN.upper()}
    kb_event_names = sorted({
        str(c) for c in abstract["ConceptName"].unique()
        if str(c).upper().endswith("_EVENT")
        and str(c).upper() not in target_spellings
        and str(c).upper() not in reserved})
    if kb_event_names:
        say(f"[Cohort] {len(kb_event_names)} non-target *_EVENT concept(s) in the "
            f"Mediator output are KB products, not targets.")
        say(f"[Cohort]   kept for the KB arms, stripped from raw-data arms "
            f"(KB_EVENTS_FOR_KB_ARMS_ONLY={C.KB_EVENTS_FOR_KB_ARMS_ONLY}): "
            f"{kb_event_names}")

    # --- 3b. event reconciliation ---------------------------------------
    # Build each file's view separately, then reconcile into one canonical
    # table. That table is what every model is trained against AND what gets
    # injected back into every input stream, so the two files end up with
    # identical event support no matter how they disagreed to begin with.
    # --- 3a. raw-side derivation ----------------------------------------
    # The raw file holds measurements, not complications. Rebuild its own view
    # of the lab-threshold targets BEFORE reconciling, so the intersection has
    # something to intersect on for every target and not just for the ones the
    # ETL happened to pre-compute.
    rule_agreement = None
    align = bool(getattr(C, "EVENT_ALIGN_ACROSS_FILES", False))
    raw_events = _per_file_events(raw, outcome_aliases, "raw",
                                  C.EVENT_MATCH_TOLERANCE_H, dedupe=not align)
    abstract_events = _per_file_events(abstract, outcome_aliases, "mediator",
                                       C.EVENT_MATCH_TOLERANCE_H, dedupe=not align)
    # Snapshot BEFORE alignment. Alignment forces the two files onto the same
    # occurrences, so a report computed after it always reads 1.00 -- which says
    # nothing about whether the derivation rule matches the Mediator's.
    pre_raw = _dedupe_occurrences(raw_events, C.EVENT_MATCH_TOLERANCE_H) if align         else raw_events
    pre_abstract = _dedupe_occurrences(abstract_events, C.EVENT_MATCH_TOLERANCE_H)         if align else abstract_events

    align_rows = []
    if align:
        say(f"[Cohort] Aligning target events across files "
            f"(+/-{C.EVENT_ALIGN_TOLERANCE_MIN:g} min on StartDateTime); "
            f"outcomes carried by only one file pass through:")
        raw_events, abstract_events, align_rows = _align_event_times(
            raw_events, abstract_events, C.EVENT_ALIGN_TOLERANCE_MIN,
            verbose=verbose)
        raw_events = _dedupe_occurrences(raw_events, C.EVENT_MATCH_TOLERANCE_H)
        abstract_events = _dedupe_occurrences(abstract_events,
                                              C.EVENT_MATCH_TOLERANCE_H)
    if getattr(C, "DERIVE_RAW_EVENTS", False):
        rule_agreement = _assert_rule_agreement(
            pre_raw, pre_abstract,
            [o for o in alias_map if outcome_aliases.get(o)], verbose=verbose)

    events, event_audit = _reconcile_events(
        raw_events, abstract_events, C.EVENT_SOURCE,
        C.EVENT_MATCH_TOLERANCE_H, verbose=verbose)
    if align_rows:
        event_audit = event_audit.merge(pd.DataFrame(align_rows), on="outcome",
                                        how="left")
    say(f"[Cohort] Canonical outcome occurrences: {len(events):,} across "
        f"{events['outcome'].nunique() if len(events) else 0} outcomes "
        f"(EVENT_SOURCE={C.EVENT_SOURCE!r}).")

    # --- 4. per-patient framing -----------------------------------------
    all_pids = pd.Index(sorted(set(raw["PatientId"]) | set(abstract["PatientId"])),
                        name="PatientId")

    # Trajectory length is measured per SOURCE TABLE and then taken as the
    # minimum, not over their union.
    #
    # The two streams do not end together: abstractions are derived, so the last
    # Mediator interval routinely closes before the last raw measurement. Using
    # the union would admit a patient whose raw data runs to day 10 but whose
    # abstractions stop at day 5 -- the raw-data models would score them and
    # INTERVenE would silently drop them inside `_cut_after_k_days`, leaving the
    # models on different test sets. Requiring BOTH streams to reach the
    # threshold costs some cohort size and buys a guarantee that every model can
    # score every patient.
    raw_traj = raw.groupby("PatientId")["hours"].max().reindex(all_pids)
    abstract_traj = abstract.groupby("PatientId")["hours"].max().reindex(all_pids)
    trajectory = pd.concat([raw_traj, abstract_traj], axis=1).min(axis=1, skipna=False)

    rel_frames = [
        raw.loc[raw["ConceptName"].isin(structural[C.RELEASE_TOKEN]), ["PatientId", "hours"]],
        abstract.loc[abstract["ConceptName"].isin(structural[C.RELEASE_TOKEN]),
                     ["PatientId", "hours"]],
    ]
    rel_rows = pd.concat(rel_frames)
    los = (rel_rows.groupby("PatientId")["hours"].min().reindex(all_pids)
           if len(rel_rows) else pd.Series(np.nan, index=all_pids))

    # AdmissionEnd is the discharge the extract was cut on. It beats a RELEASE
    # row, which is simply absent for some admissions -- and a missing LoS is a
    # masked training target, so those admissions teach the LoS head nothing.
    if window_end is not None:
        anchored = admission.reindex(all_pids)
        from_columns = ((window_end.reindex(all_pids) - anchored)
                        .dt.total_seconds() / 3600.0)
        n_recovered = int(from_columns.notna().sum() - los.notna().sum())
        los = from_columns.fillna(los)
        say(f"[Cohort] Length of stay from AdmissionEnd for "
            f"{int(from_columns.notna().sum()):,} admission(s)"
            + (f" ({n_recovered:,} that no RELEASE row covered)"
               if n_recovered > 0 else "") + ".")

    patients = pd.DataFrame({
        "PatientId": all_pids,
        "admission_time": admission.reindex(all_pids).to_numpy(),
        "los_hours": los.to_numpy(),
        "trajectory_hours": trajectory.to_numpy(),
        # Kept for diagnostics: when the cohort comes out smaller than expected,
        # these say which stream was the binding constraint.
        "raw_trajectory_hours": raw_traj.to_numpy(),
        "abstract_trajectory_hours": abstract_traj.to_numpy(),
    })

    # --- 5. cohort filter ------------------------------------------------
    # "fixed" requires every patient to survive past the LARGEST K so the whole
    # sweep shares one patient set; "per_k" only needs the smallest.
    # Keyed on the SHORTEST training window: a patient is in the study if they
    # can furnish at least one training sample. Evaluation is a separate, later
    # filter -- `samples()` gives the val/test splits only patients who reach the
    # evaluation window, because you cannot score a 3-day admission at K=4.
    #
    # Deliberately NOT keyed on the evaluation window. Doing that would let the
    # reporting decision silently shrink the training set: raising eval K from 4
    # to 6 would evict every 5-day admission from training too, even though
    # their K=1..4 cuts are perfectly good supervision.
    anchor_k = min(C.TRAIN_CONTEXT_DAYS)
    min_hours = max(C.MIN_TRAJECTORY_HOURS, anchor_k * 24.0)
    before = len(patients)
    keeps = patients["trajectory_hours"] > min_hours
    # Report which stream did the excluding, so a surprisingly small cohort is
    # diagnosable without re-running anything.
    raw_ok = patients["raw_trajectory_hours"] > min_hours
    abs_ok = patients["abstract_trajectory_hours"] > min_hours
    only_raw = int((raw_ok & ~abs_ok).sum())
    only_abs = int((abs_ok & ~raw_ok).sum())
    patients = patients[keeps].reset_index(drop=True)
    say(f"[Cohort] kept {len(patients):,}/{before:,} patients with trajectory "
        f"> {min_hours:.0f} h (= shortest training window, K={anchor_k}) in "
        f"BOTH source tables.")
    scorable = int((patients["trajectory_hours"] > C.EVAL_CONTEXT_DAYS * 24.0).sum())
    say(f"[Cohort]   of those, {scorable:,} reach the evaluation window "
        f"(K={C.EVAL_CONTEXT_DAYS}) and can appear in val/test; the remaining "
        f"{len(patients) - scorable:,} are training-only.")
    if only_raw or only_abs:
        say(f"[Cohort]   excluded for one-sided coverage: {only_raw:,} reach the "
            f"threshold only in the raw table, {only_abs:,} only in the Mediator "
            f"output. Admitting them would leave the models on different test sets.")

    # An admission too close to the end of the extract cannot have a complete
    # label window: complications after the cut-off are invisible and would be
    # scored as negatives. That is administrative censoring, and it looks
    # identical to a real negative to every model, so those patients leave.
    if C.DATE_RANGE_REQUIRE_FULL_HORIZON and range_end is not None:
        horizon = pd.Timedelta(hours=C.HORIZON_END_DAYS * 24.0)
        complete = patients["admission_time"] + horizon <= range_end
        n_censored = int((~complete).sum())
        if n_censored:
            say(f"[Cohort] Dropping {n_censored:,} admission(s) whose "
                f"{C.HORIZON_END_DAYS:.0f}-day label window would run past "
                f"{C.DATE_RANGE_END} (incomplete follow-up).")
            patients = patients[complete].reset_index(drop=True)
        if len(patients) == 0:
            raise RuntimeError(
                f"[Cohort] Every admission was dropped for incomplete follow-up. "
                f"Extend DATE_RANGE_END, shorten HORIZON_END_DAYS, or set "
                f"DATE_RANGE_REQUIRE_FULL_HORIZON=False (accepting censoring)."
            )
    if len(patients) == 0:
        raise RuntimeError(
            "[Cohort] Every patient was filtered out. Shorten the smallest "
            "train_context_days, or lower min_trajectory_hours."
        )

    keep = set(patients["PatientId"])
    events = events[events["PatientId"].isin(keep)].reset_index(drop=True)

    # --- 6. frozen split -------------------------------------------------
    rng = np.random.default_rng(C.SPLIT_SEED)
    group_by_person = (bool(getattr(C, "SPLIT_GROUP_BY_PERSON", False))
                       and bool(person_of_patient))
    if group_by_person:
        # Split PEOPLE, then assign every admission of a person to their side.
        # Two admissions of one patient share comorbidities, baseline physiology
        # and often the same recurring complication; training on one and scoring
        # the other measures memorisation of that person, not generalisation.
        person = patients["PatientId"].map(person_of_patient)
        # An admission with no person row cannot be grouped -- give it its own
        # group rather than pooling every unknown into one giant group.
        person = person.where(person.notna(),
                              "__ungrouped__" + patients["PatientId"].astype(str))
        units = pd.unique(person)
    else:
        person = None
        units = patients["PatientId"].to_numpy().copy()

    shuffled = np.asarray(units).copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(C.SPLIT_FRACTIONS["train"] * n))
    n_val = int(round(C.SPLIT_FRACTIONS["val"] * n))
    assignment = {}
    assignment.update({p: "train" for p in shuffled[:n_train]})
    assignment.update({p: "val" for p in shuffled[n_train:n_train + n_val]})
    assignment.update({p: "test" for p in shuffled[n_train + n_val:]})
    if group_by_person:
        patients["split"] = person.map(assignment).to_numpy()
        say(f"[Cohort] Split grouped by person: {n:,} people -> "
            f"{len(patients):,} admissions; no person spans two splits.")
    else:
        patients["split"] = patients["PatientId"].map(assignment)
        if getattr(C, "SPLIT_GROUP_BY_PERSON", False):
            say("[Cohort] split_group_by_person is on but the context table "
                "supplied no person column -- splitting by admission. If one "
                "person can have several admissions here, that is leakage.")
    say("[Cohort] Split: " + ", ".join(
        f"{s}={int((patients['split'] == s).sum()):,}" for s in ("train", "val", "test")))

    # --- 7. outcome support filter ---------------------------------------
    # Measured on TRAIN at the EVALUATION window. That is the window every
    # reported number comes from, so an outcome that cannot clear the bar there
    # has no business occupying a prediction head -- even if a shorter training
    # window would have given it more positives.
    eval_k = C.EVAL_CONTEXT_DAYS
    lo, hi = eval_k * 24.0, C.HORIZON_END_DAYS * 24.0
    # Restricted to train patients who REACH the evaluation window: that is the
    # population the head will be judged on, so it is the population its
    # learnability should be measured over. Including shorter stays -- which can
    # never contribute a K=eval sample -- would dilute the prevalence with
    # patients the head never sees in that condition.
    scorable_train = patients[(patients["split"] == "train")
                              & (patients["trajectory_hours"] > lo)]
    train_ids = set(scorable_train["PatientId"])
    n_train_pat = max(len(train_ids), 1)
    window = events[(events["hours"] > lo) & (events["hours"] <= hi)
                    & (events["PatientId"].isin(train_ids))]
    support = window.groupby("outcome")["PatientId"].nunique() / n_train_pat

    outcome_names, dropped = [], {}
    for name in [o for o in alias_map if outcome_aliases.get(o)]:
        prevalence = float(support.get(name, 0.0))
        if prevalence >= C.OUTCOME_SUPPORT_THRESHOLD:
            outcome_names.append(name)
        else:
            dropped[name] = prevalence
    say(f"[Cohort] Targets kept ({len(outcome_names)}) at the evaluation window "
        f"K={eval_k}: {outcome_names}")
    if dropped:
        say(f"[Cohort] Targets dropped (train prevalence at K={eval_k} below "
            f"{C.OUTCOME_SUPPORT_THRESHOLD:.1%}): "
            + ", ".join(f"{k}={v:.3%}" for k, v in sorted(dropped.items(),
                                                          key=lambda kv: -kv[1])))
    if not outcome_names:
        raise RuntimeError(
            "[Cohort] No target cleared outcome_support_threshold at "
            f"K={eval_k}. Lower the threshold, shorten eval_context_days, or "
            "widen horizon_end_days."
        )

    # --- 8. context + QA -------------------------------------------------
    # Identity was resolved at load time by `_resolve_context_identity`.
    context = context.drop(columns=[c for c in ("index", "Unnamed: 0") if c in context.columns])
    if not context["PatientId"].is_unique:
        say("[Cohort] Duplicate PatientIds in context -- aggregating by max.")
        context = context.groupby("PatientId", as_index=False).max()
    context = context.set_index("PatientId").reindex(patients["PatientId"])
    n_missing_ctx = int(context.isna().all(axis=1).sum())
    if n_missing_ctx:
        say(f"[Cohort] {n_missing_ctx} cohort patients have no context row -- "
            f"filling with the column median.")
    context = context.apply(pd.to_numeric, errors="coerce")
    context = context.fillna(context.median(numeric_only=True)).fillna(0.0).astype("float32")

    qa_by_k = {}
    if qa_df is not None:
        # One block per window the study will ever ask for: every training
        # window (augmentation) plus the evaluation window.
        windows = sorted(set(C.TRAIN_CONTEXT_DAYS) | {C.EVAL_CONTEXT_DAYS})
        for k in windows:
            qa_by_k[k] = _build_qa_block(qa_df, admission,
                                         patients["PatientId"].to_numpy(), k)
        say(f"[Cohort] QA blocks built for K={windows} "
            f"({qa_by_k[windows[0]].shape[1]} pattern columns each).")

    meta = {
        "raw_temporal_file": paths.RAW_TEMPORAL_FILE,
        "abstract_file": paths.ABSTRACT_FILE,
        "context_file": paths.CONTEXT_FILE,
        "qa_file": paths.QA_FILE if qa_df is not None else None,
        "n_raw_rows": int(len(raw)),
        "n_abstract_rows": int(len(abstract)),
        "context_columns": list(context.columns),

        "horizon_end_days": float(C.HORIZON_END_DAYS),
        "train_context_days": list(C.TRAIN_CONTEXT_DAYS),
        "eval_context_days": C.EVAL_CONTEXT_DAYS,
        "split_seed": C.SPLIT_SEED,
        "outcome_support_threshold": C.OUTCOME_SUPPORT_THRESHOLD,
        "targets": list(C.OUTCOMES),
        "support_measured_at_k": eval_k,
        "date_range_start": C.DATE_RANGE_START,
        "date_range_end": C.DATE_RANGE_END,
        "date_range_require_full_horizon": C.DATE_RANGE_REQUIRE_FULL_HORIZON,
        "event_source": C.EVENT_SOURCE,
        "event_match_tolerance_h": C.EVENT_MATCH_TOLERANCE_H,
        "events_as_inputs": C.EVENTS_AS_INPUTS,
        "kb_events_for_kb_arms_only": getattr(C, "KB_EVENTS_FOR_KB_ARMS_ONLY", True),
        "event_align_across_files": getattr(C, "EVENT_ALIGN_ACROSS_FILES", False),
        "event_align_tolerance_min": getattr(C, "EVENT_ALIGN_TOLERANCE_MIN", 1.0),
        # The SETTING, so the drift guard compares like with like; whether it
        # actually took effect is recorded separately.
        "split_group_by_person": bool(getattr(C, "SPLIT_GROUP_BY_PERSON", False)),
        "split_grouped_in_practice": bool(group_by_person),
        "n_kb_events_withheld": len(kb_event_names),
        "rule_agreement": rule_agreement,
    }

    return Cohort(
        patients=patients, events=events, outcome_names=outcome_names,
        dropped_outcomes=dropped, context=context, qa_by_k=qa_by_k,
        outcome_aliases=outcome_aliases, structural_aliases=structural,
        event_audit=event_audit, kb_event_names=kb_event_names,
        replaced_spellings=replaced_spellings,
        person_of_patient=person_of_patient, meta=meta,
    )
