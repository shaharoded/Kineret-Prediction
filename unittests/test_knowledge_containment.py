"""
Tests for the four invariants that keep the ladder's claim honest.

The ladder argues `raw < sigma-bins < KB abstractions < KB + treatment hints`.
That argument only measures what it says it measures if:

  1. the notebook's TARGET list is the target list (nothing else gets a head),
  2. the knowledge base's own conclusions never reach a model with no
     knowledge base,
  3. every arm is credited with the same target events, dated the same way,
  4. the split separates PEOPLE, not just admissions.

Each is cheap to get wrong and invisible in the results when it is.
"""

import re

import pandas as pd

from kineret.config import data_config as C
from kineret.config import paths
from kineret.io_utils import load_table, normalise_temporal


def _kb_stream(cohort):
    """Purpose: The abstraction stream exactly as the INTERVenE KB arm sees it."""
    abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
    abstract = abstract[abstract["PatientId"].isin(set(cohort.patients["PatientId"]))]
    return cohort.harmonise_events(abstract, patient_ids=cohort.patients["PatientId"],
                                   kb_events=True)


def _raw_stream(cohort):
    """Purpose: The raw stream exactly as LogReg / ss-STraTS see it."""
    from kineret.benchmark import raw_with_hours
    return raw_with_hours(cohort)


class TestKnowledgeContainment:
    """
    The Mediator emits `*_EVENT` concepts that are not prediction targets --
    AKI_EVENT, ELECTROLYTE_DERANGEMENT_EVENT and friends. Each is a KB verdict
    reached by reading raw measurements. Handing one to LogReg or ss-STraTS
    gives them a derived clinical judgement for free, and rungs 5-6 would then
    measure a leak rather than the value of abstraction.
    """

    def test_non_target_events_are_identified(self, cohort):
        """KB events are detected, and none of them is a target."""
        assert cohort.kb_event_names, (
            "the synthetic Mediator output carries non-target *_EVENT concepts; "
            "none were detected")
        target_spellings = {s for name in cohort.outcome_names
                            for s in cohort.outcome_aliases.get(name, [])}
        assert not (set(cohort.kb_event_names) & target_spellings)

    def test_raw_stream_carries_no_kb_conclusions(self, cohort):
        """The stream LogReg and ss-STraTS read is free of KB events."""
        present = set(_raw_stream(cohort)["ConceptName"]) & set(cohort.kb_event_names)
        assert not present, f"KB conclusions reached a raw-data arm: {sorted(present)}"

    def test_kb_stream_keeps_them(self, cohort):
        """The KB arm's own abstractions are not stripped -- they are its input."""
        present = set(_kb_stream(cohort)["ConceptName"]) & set(cohort.kb_event_names)
        assert present, "the KB arm lost the abstractions it is supposed to use"

    def test_target_support_is_identical_in_both_streams(self, cohort):
        """
        Containment must not disturb the shared target-event support.

        Measured over every RESOLVED target spelling, not just the ones that
        kept a prediction head: `cohort.events` carries the occurrences of
        support-filtered targets too, and both streams are injected with all of
        them. Restricting to `outcome_names` here would compare a subset of the
        streams against the whole canonical table.
        """
        targets = {s for spellings in cohort.outcome_aliases.values()
                   for s in spellings}
        raw_n = int(_raw_stream(cohort)["ConceptName"].isin(targets).sum())
        kb_n = int(_kb_stream(cohort)["ConceptName"].isin(targets).sum())
        assert raw_n == kb_n == len(cohort.events)

    def test_blocklist_is_stream_specific(self, cohort):
        """Blocked for the raw arms, allowed for the KB arms."""
        assert set(cohort.kb_event_names) <= cohort.leakage_blocklist(kb_events=False)
        assert not (set(cohort.kb_event_names)
                    & cohort.leakage_blocklist(kb_events=True))


class TestTargetsAreAuthoritative:
    """
    `targets=[...]` in the notebook is the target list. The alias table only
    says how each name is spelled across the two files -- deriving the targets
    from it meant the notebook's list was silently ignored and every regex it
    happened to contain got a prediction head.
    """

    def test_no_head_outside_the_configured_targets(self, cohort):
        assert set(cohort.outcome_names) <= set(C.OUTCOMES), (
            f"heads exist for non-targets: "
            f"{sorted(set(cohort.outcome_names) - set(C.OUTCOMES))}")

    def test_canonical_events_are_targets_only(self, cohort):
        assert set(cohort.events["outcome"]) <= set(C.OUTCOMES)

    def test_a_target_absent_from_the_alias_table_still_resolves(self):
        """A name the alias table never heard of falls back to `<NAME>(_EVENT)?`."""
        from kineret.cohort import _default_alias_regex
        pattern = _default_alias_regex("NOVEL_COMPLICATION_EVENT")
        for spelling in ("NOVEL_COMPLICATION", "NOVEL_COMPLICATION_EVENT"):
            assert re.fullmatch(pattern, spelling, re.IGNORECASE), spelling


class TestCrossFileEventAlignment:
    """
    For a target both files carry, only the occurrences they date identically
    are credited -- that is what removes the 3096-vs-3098 support drift. A
    target only one file carries passes through: intersecting a Mediator-derived
    complication against an empty raw view would delete it entirely.
    """

    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["PatientId", "outcome", "hours", "source"])

    def test_unmatched_occurrences_are_dropped_from_both_files(self, monkeypatch):
        from kineret.cohort import _align_event_times
        monkeypatch.setattr(C, "EVENT_ALIGN_MAX_LOSS", 1.0, raising=False)
        raw = self._frame([(1, "X_EVENT", 10.0, "raw"), (1, "X_EVENT", 99.0, "raw")])
        med = self._frame([(1, "X_EVENT", 10.005, "mediator")])   # 18 s apart
        r, a, rows = _align_event_times(raw, med, tolerance_min=1.0, verbose=False)
        assert len(r) == 1 and len(a) == 1
        assert float(r["hours"].iloc[0]) == 10.0
        assert rows[0]["raw_before"] == 2 and rows[0]["raw_after"] == 1

    def test_beyond_tolerance_is_not_a_match(self, monkeypatch):
        """
        The valve is disabled here on purpose: a one-event fixture that fails to
        match is 100 % loss, which the valve would (correctly) refuse. This test
        is about the matching rule itself.
        """
        from kineret.cohort import _align_event_times
        monkeypatch.setattr(C, "EVENT_ALIGN_MAX_LOSS", 1.0, raising=False)
        raw = self._frame([(1, "X_EVENT", 10.0, "raw")])
        med = self._frame([(1, "X_EVENT", 10.5, "mediator")])     # 30 min apart
        r, a, _ = _align_event_times(raw, med, tolerance_min=1.0, verbose=False)
        assert len(r) == 0 and len(a) == 0

    def test_single_file_outcomes_pass_through_untouched(self):
        """The Mediator-only targets are the study's main labels -- keep them."""
        from kineret.cohort import _align_event_times
        raw = self._frame([])
        med = self._frame([(1, "ONLY_MED_EVENT", 5.0, "mediator"),
                           (2, "ONLY_MED_EVENT", 7.0, "mediator")])
        _, a, _ = _align_event_times(raw, med, tolerance_min=1.0, verbose=False)
        assert len(a) == 2, "a Mediator-derived target was deleted by alignment"

    def test_alignment_never_matches_across_patients(self, monkeypatch):
        from kineret.cohort import _align_event_times
        monkeypatch.setattr(C, "EVENT_ALIGN_MAX_LOSS", 1.0, raising=False)
        raw = self._frame([(1, "X_EVENT", 10.0, "raw")])
        med = self._frame([(2, "X_EVENT", 10.0, "mediator")])
        r, a, _ = _align_event_times(raw, med, tolerance_min=1.0, verbose=False)
        assert len(r) == 0 and len(a) == 0


class TestAdmissionIdentity:
    """
    The Mediator carries one id column, so the temporal tables key on the
    ADMISSION. The context table keys on VisitId and names the person, who may
    have several admissions in the extract.
    """

    def test_context_is_keyed_on_the_admission(self, cohort):
        assert cohort.context.index.name == "PatientId"
        assert len(cohort.context) == len(cohort.patients)

    def test_person_id_is_not_a_feature(self, cohort):
        """An identifier as a feature invites the model to memorise people."""
        lowered = {str(c).lower() for c in cohort.context.columns}
        assert "person_id" not in lowered
        assert C.CONTEXT_PERSON_COLUMN.lower() not in lowered

    def test_person_map_covers_the_cohort(self, cohort):
        assert cohort.person_of_patient
        assert set(cohort.patients["PatientId"]) <= set(cohort.person_of_patient)

    def test_no_person_spans_two_splits(self, cohort):
        """
        Two admissions of one patient share comorbidities, baseline physiology
        and often the same recurring complication. Training on one and scoring
        the other measures memorisation of that person, not generalisation.
        """
        patients = cohort.patients.copy()
        patients["person"] = patients["PatientId"].map(cohort.person_of_patient)
        spans = patients.groupby("person")["split"].nunique()
        offenders = spans[spans > 1]
        assert offenders.empty, (
            f"{len(offenders)} person(s) appear in more than one split: "
            f"{list(offenders.index[:5])}")

    def test_repeat_admissions_exist_in_the_fixture(self, cohort):
        """Otherwise the test above would pass vacuously."""
        people = pd.Series(list(cohort.person_of_patient.values()))
        assert people.duplicated().any(), (
            "fixture has no readmissions, so the grouping test proves nothing")


class TestReplacedSpellingsAreStripped:
    """
    `raw_events` rewrites the raw file's own event spellings to the canonical
    name inside the cohort. Any stream read fresh from disk still spells them
    the old way -- and unless those are stripped too, the raw-data arms carry
    both the ETL's `INFECTION` row and the injected `INFECTION_EVENT` at the
    same instant, a duplicate the KB arm never sees.
    """

    def test_the_replaced_spellings_are_recorded(self, cohort):
        assert cohort.replaced_spellings, (
            "the synthetic raw file spells several targets without the _EVENT "
            "suffix; none were recorded as replaced")
        assert not (set(cohort.replaced_spellings) & set(cohort.outcome_names))

    def test_no_stream_carries_them(self, cohort):
        stale = set(cohort.replaced_spellings)
        assert not (set(_raw_stream(cohort)["ConceptName"]) & stale)
        assert not (set(_kb_stream(cohort)["ConceptName"]) & stale)

    def test_a_fresh_read_is_harmonised_the_same_way(self, cohort):
        """The regression: harmonising a table straight off disk must strip them."""
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        assert set(raw["ConceptName"]) & set(cohort.replaced_spellings), (
            "fixture no longer exercises this path")
        out = cohort.harmonise_events(raw, patient_ids=cohort.patients["PatientId"])
        assert not (set(out["ConceptName"]) & set(cohort.replaced_spellings))
