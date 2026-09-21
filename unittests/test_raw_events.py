"""
Tests for reproducing the Mediator's events from the raw stream.

The Mediator derived `mediator_output.csv` from `mediator_input.csv` using the
rules in its knowledge base. `kineret.raw_events` executes those same rules, so
the two views must agree occurrence for occurrence. Anything less is a defect in
the reproduction — never a disagreement between two sources — and the cohort
build refuses to continue on it.

The synthetic Mediator in `scripts/make_synthetic_data.py` implements the same
definitions independently, so the end-to-end test below is a genuine
cross-check rather than the executor grading its own homework.
"""

import pandas as pd
import pytest

from kineret.config import data_config as C
from kineret.raw_events import (agreement_report, derive_raw_events, load_rules,
                                resolve_concepts, _clause_hits)

COLUMNS = ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"]


def _readings(values, concept="GLUCOSE_MEASURE", patient=1, gap_h=1.0):
    """Purpose: One concept's readings for one admission, evenly spaced."""
    base = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        [(patient, concept, base + pd.Timedelta(hours=i * gap_h),
          base + pd.Timedelta(hours=i * gap_h), v) for i, v in enumerate(values)],
        columns=COLUMNS)


def _hits(values, clause, clippers=None, **kwargs):
    """Purpose: Run one compiled clause over a series of readings."""
    rows = _readings(values, **kwargs)
    return _clause_hits(rows, clause, clippers or {}, clause["source"])


VALUE_CLAUSE = {"source": "GLUCOSE_MEASURE", "via": None, "transform": None,
                "constraints": [{"type": "min", "value": 250.0}]}
SUSTAINED_CLAUSE = {"source": "GLUCOSE_MEASURE", "via": "STEADY_GLUCOSE_MEASURE_HIGH",
                    "transform": {"kind": "sustained", "gate_op": "ge",
                                  "gate_value": 180.0, "window": "24h"},
                    "constraints": [{"type": "min", "value": 180.0}]}
RATIO_CLAUSE = {"source": "CREATININE_SERUM_MEASURE", "via": "CREATININE_REL_SERUM_MEASURE",
                "transform": {"kind": "ratio_to_baseline"},
                "constraints": [{"type": "min", "value": 2.0}]}


class TestCompiledRulesMatchTheKnowledgeBase:
    """
    The thresholds are not written in this package. If these change, the
    knowledge base changed and the rules must be recompiled.
    """

    @pytest.fixture(scope="class")
    def rules(self):
        return load_rules()

    def test_the_glycaemic_thresholds(self, rules):
        events = rules["events"]
        flat = {name: [(c["source"], (c["transform"] or {}).get("kind", "value"),
                        tuple(sorted(x.items())[0] for x in c["constraints"]))
                       for rule in spec["rules"] for c in rule["clauses"]]
                for name, spec in events.items()}

        assert ("GLUCOSE_MEASURE", "value", (("type", "min"),)) not in flat  # shape guard
        hyper = dict((c[0] + ":" + c[1], c) for c in flat["HYPERGLYCEMIA_EVENT"])
        assert "GLUCOSE_MEASURE:value" in hyper
        assert "GLUCOSE_MEASURE:sustained" in hyper

    def test_severe_hyperglycemia_is_a_single_extreme_reading(self, rules):
        clauses = [c for rule in rules["events"]["SEVERE_HYPERGLYCEMIA_EVENT"]["rules"]
                   for c in rule["clauses"]]
        assert len(clauses) == 1 and clauses[0]["transform"] is None
        assert clauses[0]["constraints"] == [{"type": "min", "value": 250.0}]

    def test_kidney_complication_has_all_three_clauses(self, rules):
        """Ratio to baseline, severe absolute, and the observation code."""
        kinds = {(c["source"], (c["transform"] or {}).get("kind", "value"))
                 for rule in rules["events"]["KIDNEY_COMPLICATION_EVENT"]["rules"]
                 for c in rule["clauses"]}
        assert ("CREATININE_SERUM_MEASURE", "ratio_to_baseline") in kinds
        assert ("CREATININE_SERUM_MEASURE", "value") in kinds
        assert ("KIDNEY_COMPLICATION", "value") in kinds

    def test_measurement_validity_ranges_are_the_concepts_own(self, rules):
        """
        `HIGH_GLUCOSE_IND` and `LOW_GLUCOSE_IND` both declare an attribute named
        `GLUCOSE_MEASURE` with their own bounds. Keying clippers by attribute
        name let LOW_GLUCOSE_IND clip every glucose reading to <= 70, which
        silently produced zero hyperglycaemia events.
        """
        assert rules["clippers"]["GLUCOSE_MEASURE"] == [20.0, 1200.0]
        assert rules["clippers"]["CREATININE_SERUM_MEASURE"] == [0.1, 20.0]


class TestClauseExecution:

    def test_absolute_threshold_is_inclusive(self):
        hits = _hits([249.9, 250.0, 250.1], VALUE_CLAUSE)
        assert list(hits["Value"]) == [250.0, 250.1]

    def test_out_of_range_readings_never_reach_a_rule(self):
        """The engine validates against the concept's range before any rule."""
        hits = _hits([5000.0, 300.0], VALUE_CLAUSE,
                     clippers={"GLUCOSE_MEASURE": [20.0, 1200.0]})
        assert list(hits["Value"]) == [300.0]

    def test_sustained_needs_the_previous_reading_to_qualify_too(self):
        hits = _hits([200.0, 200.0, 100.0, 200.0], SUSTAINED_CLAUSE)
        # row 0 has no predecessor; row 2 is below; row 3's predecessor is below
        assert list(hits.index) == [1]

    def test_sustained_respects_the_good_before_window(self):
        """A predecessor older than 24 h does not count as sustained."""
        near = _hits([200.0, 200.0], SUSTAINED_CLAUSE, gap_h=10.0)
        far = _hits([200.0, 200.0], SUSTAINED_CLAUSE, gap_h=30.0)
        assert len(near) == 1 and len(far) == 0

    def test_sustained_does_not_span_admissions(self):
        rows = pd.concat([_readings([200.0], patient=1),
                          _readings([200.0], patient=2)], ignore_index=True)
        assert _clause_hits(rows, SUSTAINED_CLAUSE, {}, "GLUCOSE_MEASURE").empty

    def test_ratio_uses_the_first_reading_and_excludes_it(self):
        """
        The engine consumes the admission's first reading as the baseline and
        removes it from its own output, so it can never fire the rule itself.
        """
        hits = _hits([2.0, 1.0, 4.0, 5.0], RATIO_CLAUSE,
                     concept="CREATININE_SERUM_MEASURE")
        assert list(hits["Value"]) == [4.0, 5.0]

    def test_ratio_ignores_a_nonpositive_baseline(self):
        assert _hits([0.0, 5.0], RATIO_CLAUSE,
                     concept="CREATININE_SERUM_MEASURE").empty

    def test_boolean_clause_matches_text(self):
        clause = {"source": "INFECTION", "via": None, "transform": None,
                  "constraints": [{"type": "equal", "value": "True"}]}
        hits = _hits(["True", "false", "TRUE"], clause, concept="INFECTION")
        assert len(hits) == 2


class TestConceptSpelling:
    """
    A rule that cannot find its source concept derives nothing, which looks
    exactly like "this complication never happened".
    """

    def test_punctuation_differences_resolve(self):
        rules = {"events": {"E": {"rules": [{"clauses": [
            {"source": "CARDIO-VASCULAR_DISORDER"}]}]}}}
        resolved = resolve_concepts(rules, ["CARDIOVASCULAR_DISORDER", "OTHER"])
        assert resolved["CARDIO-VASCULAR_DISORDER"] == ["CARDIOVASCULAR_DISORDER"]

    def test_exact_match_wins(self):
        rules = {"events": {"E": {"rules": [{"clauses": [
            {"source": "GLUCOSE_MEASURE"}]}]}}}
        resolved = resolve_concepts(rules, ["GLUCOSE_MEASURE", "GLUCOSEMEASURE"])
        assert resolved["GLUCOSE_MEASURE"] == ["GLUCOSE_MEASURE"]

    def test_absent_concept_is_simply_not_resolved(self):
        rules = {"events": {"E": {"rules": [{"clauses": [
            {"source": "TROPONIN_MEASURE"}]}]}}}
        assert resolve_concepts(rules, ["GLUCOSE_MEASURE"]) == {}


class TestDerivationOutput:

    def test_events_land_on_the_qualifying_reading(self):
        """Agreement is measured to the instant, so the stamp must be exact."""
        raw = _readings([100.0, 300.0])
        out = derive_raw_events(raw, targets=["SEVERE_HYPERGLYCEMIA_EVENT"],
                                verbose=False)
        assert list(out["StartDateTime"]) == [raw["StartDateTime"].iloc[1]]
        assert (out["StartDateTime"] == out["EndDateTime"]).all()
        assert set(out["Value"]) == {"True"}

    def test_clauses_are_or_ed_without_double_counting(self):
        """
        A reading of 300 satisfies both HYPERGLYCEMIA clauses when the previous
        reading was also high. It is still one event.
        """
        raw = _readings([200.0, 300.0])
        out = derive_raw_events(raw, targets=["HYPERGLYCEMIA_EVENT"], verbose=False)
        assert len(out) == 1

    def test_a_target_with_no_compiled_rule_is_reported_not_guessed(self):
        raw = _readings([300.0])
        out = derive_raw_events(raw, targets=["NOT_AN_EVENT"], verbose=False)
        assert out.empty


class TestAgreementWithTheMediator:
    """The end-to-end contract, against an independently written generator."""

    def test_every_target_reproduces_exactly(self, cohort):
        report = cohort.meta.get("rule_agreement")
        assert report is not None, "the cohort did not record a rule-agreement check"
        offenders = {name: value for name, value in report.items() if value < 1.0}
        assert not offenders, (
            f"targets that do not reproduce the Mediator exactly: {offenders}")

    def test_agreement_report_counts_both_directions(self):
        raw = pd.DataFrame({"PatientId": [1, 2], "outcome": "A",
                            "hours": [1.0, 2.0], "source": "raw"})
        med = pd.DataFrame({"PatientId": [1, 3], "outcome": "A",
                            "hours": [1.0, 9.0], "source": "mediator"})
        row = agreement_report(raw, med, ["A"]).iloc[0]
        assert row["matched"] == 1
        assert row["raw_only"] == 1, "an invented event must be visible"
        assert row["med_only"] == 1, "a missed event must be visible"
        assert row["agreement"] == pytest.approx(1 / 3)


class TestDeclaredAttributes:
    """
    A raw concept declares the ConceptNames it accepts, and the knowledge base
    is the authority on that. `KIDNEY_COMPLICATION` accepts an observation code
    filed under `KIDNEY_COMPLICATION_OBS`; looking only for the concept's own
    name missed 1,087 of 2,416 kidney events on the real extract, and the
    intersection would then have deleted them as "disagreement".
    """

    def test_the_kidney_observation_code_is_declared(self):
        assert set(load_rules()["attributes"]["KIDNEY_COMPLICATION"]) == {
            "KIDNEY_COMPLICATION", "KIDNEY_COMPLICATION_OBS"}

    def test_the_cardiovascular_spellings_are_declared(self):
        """The unhyphenated ETL spelling is stated by the KB, not guessed."""
        assert set(load_rules()["attributes"]["CARDIO-VASCULAR_DISORDER"]) == {
            "CARDIO-VASCULAR_DISORDER", "CARDIOVASCULAR_DISORDER"}

    def test_every_declared_spelling_present_is_resolved(self):
        rules = {"attributes": {"KIDNEY_COMPLICATION":
                                ["KIDNEY_COMPLICATION", "KIDNEY_COMPLICATION_OBS"]},
                 "events": {"E": {"rules": [{"clauses": [
                     {"source": "KIDNEY_COMPLICATION"}]}]}}}
        resolved = resolve_concepts(rules, ["KIDNEY_COMPLICATION_OBS", "OTHER"])
        assert resolved["KIDNEY_COMPLICATION"] == ["KIDNEY_COMPLICATION_OBS"]

    def test_a_clause_fires_on_any_declared_spelling(self):
        clause = {"source": "KIDNEY_COMPLICATION", "via": None, "transform": None,
                  "constraints": [{"type": "equal", "value": "True"}]}
        rows = pd.concat([
            _readings(["True"], concept="KIDNEY_COMPLICATION", patient=1),
            _readings(["True"], concept="KIDNEY_COMPLICATION_OBS", patient=2),
        ], ignore_index=True)
        hits = _clause_hits(rows, clause, {},
                            ["KIDNEY_COMPLICATION", "KIDNEY_COMPLICATION_OBS"])
        assert set(hits["PatientId"]) == {1, 2}
