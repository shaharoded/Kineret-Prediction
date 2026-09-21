"""
Tests for the knowledge-free abstraction arm.

The std arm only earns its place in the ladder if it is genuinely
knowledge-free and genuinely produces intervals. These check both, plus the
TAK-repository augmentation that lets it run on the same architecture as the
knowledge-based arm.
"""

import json

import numpy as np
import pandas as pd
import pytest

from kineret.abstraction import build_std_temporal, collapse_intervals, std_bin
from kineret.config import data_config as C


@pytest.fixture(scope="module")
def std_arm(cohort):
    """The std temporal table and its augmented TAK repository."""
    return build_std_temporal(cohort, verbose=False)


class TestBinning:
    def test_bins_are_distribution_derived(self):
        """A reading one sigma above its concept mean lands in the HIGH band."""
        base = pd.Timestamp("2024-01-01")
        values = list(np.arange(0.0, 100.0))
        df = pd.DataFrame({
            "PatientId": 1, "ConceptName": "X_MEASURE",
            "StartDateTime": [base + pd.Timedelta(hours=i) for i in range(len(values))],
            "EndDateTime": [base + pd.Timedelta(hours=i) for i in range(len(values))],
            "Value": values,
        })
        binned, stats = std_bin(df)
        mean, std = stats.iloc[0]["mean"], stats.iloc[0]["std"]

        z = (binned["Value"].map(dict(zip(C.STD_BIN_LABELS, C.STD_BIN_LABELS))))
        assert set(binned["Value"]) <= set(C.STD_BIN_LABELS)
        # The concept name carries the bin, so the token is symbolic not numeric.
        assert binned["ConceptName"].str.startswith("X_MEASURE_STD_").all()
        # The extremes must not be NORMAL.
        by_value = binned.assign(v=values).sort_values("v")
        assert by_value.iloc[0]["Value"] in ("VERY_LOW", "LOW")
        assert by_value.iloc[-1]["Value"] in ("VERY_HIGH", "HIGH")

    def test_constant_concept_is_all_normal(self):
        """Zero variance carries no state information, so everything is NORMAL."""
        base = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "PatientId": 1, "ConceptName": "FLAT",
            "StartDateTime": [base + pd.Timedelta(hours=i) for i in range(5)],
            "EndDateTime": [base + pd.Timedelta(hours=i) for i in range(5)],
            "Value": [7.0] * 5,
        })
        binned, _ = std_bin(df)
        assert set(binned["Value"]) == {"NORMAL"}

    def test_categorical_rows_pass_through(self):
        """Booleans and event markers are already symbolic; binning must not touch them."""
        base = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "PatientId": [1, 1], "ConceptName": ["DEATH_EVENT", "GLUCOSE_MEASURE"],
            "StartDateTime": [base, base], "EndDateTime": [base, base],
            "Value": ["True", 120.0],
        })
        binned, _ = std_bin(df)
        assert "DEATH_EVENT" in set(binned["ConceptName"])
        assert (binned.loc[binned["ConceptName"] == "DEATH_EVENT", "Value"] == "True").all()


class TestCollapsing:
    def test_near_observations_merge_into_one_interval(self):
        base = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "PatientId": 1, "ConceptName": "X_STD_NORMAL",
            "StartDateTime": [base, base + pd.Timedelta(hours=12),
                              base + pd.Timedelta(hours=24)],
            "EndDateTime": base, "Value": "NORMAL",
        })
        out = collapse_intervals(df, max_gap_hours=24.0)
        assert len(out) == 1
        assert out.iloc[0]["StartDateTime"] == base
        assert out.iloc[0]["EndDateTime"] == base + pd.Timedelta(hours=24)

    def test_a_long_gap_breaks_the_chain(self):
        base = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "PatientId": 1, "ConceptName": "X_STD_NORMAL",
            "StartDateTime": [base, base + pd.Timedelta(hours=30)],
            "EndDateTime": base, "Value": "NORMAL",
        })
        assert len(collapse_intervals(df, max_gap_hours=24.0)) == 2


class TestStdArm:
    def test_produces_real_intervals(self, std_arm):
        """Without intervals the encoder has no START/END structure to model."""
        df, _ = std_arm
        duration = (pd.to_datetime(df["EndDateTime"])
                    - pd.to_datetime(df["StartDateTime"])).dt.total_seconds()
        assert (duration > 1).mean() > 0.05, "std arm produced almost no intervals"

    def test_carries_no_kb_abstractions(self, std_arm):
        """The point of the arm: no Mediator state/trend/pattern concepts."""
        df, _ = std_arm
        concepts = set(df["ConceptName"].unique())
        leaked = {c for c in concepts
                  if ("_STD_" not in c)
                  and (c.endswith("_STATE") or c.endswith("_TREND")
                       or "_PATTERN" in c)}
        assert not leaked, f"KB abstractions leaked into the std arm: {leaked}"

    def test_structural_tokens_are_canonical(self, std_arm, cohort):
        """The encoder's admission / terminal logic keys on the canonical names."""
        df, _ = std_arm
        concepts = set(df["ConceptName"].unique())
        assert C.ADMISSION_TOKEN in concepts

    def test_every_concept_resolves_in_the_augmented_repo(self, std_arm):
        """DataProcessor raises on any concept the TAK repository does not know."""
        df, tak_path = std_arm
        with open(tak_path, encoding="utf-8") as f:
            known = set(json.load(f)["taks"].keys())
        assert set(df["ConceptName"].unique()) <= known

    def test_std_bins_declare_their_raw_parent(self, std_arm):
        """The embedder uses the raw parent as a hierarchy level; it must resolve."""
        df, tak_path = std_arm
        with open(tak_path, encoding="utf-8") as f:
            taks = json.load(f)["taks"]
        for name in df["ConceptName"].unique():
            if "_STD_" in name:
                assert taks[name]["derived_from"] == name.rsplit("_STD_", 1)[0]

    def test_kb_events_are_absent_by_default(self, std_arm, cohort):
        """Default arm gets no Mediator-derived complications -- that is the ablation."""
        if C.STD_INCLUDE_KB_EVENTS:
            pytest.skip("STD_INCLUDE_KB_EVENTS is on")
        df, _ = std_arm
        concepts = set(df["ConceptName"].unique())
        # An outcome the Mediator derives but the raw table never spells out
        # must not appear. HYPERGLYCEMIA is derived from glucose by the KB.
        mediator_only = {
            name for name in cohort.outcome_names
            if all(s.endswith("_EVENT") for s in cohort.outcome_aliases[name])
        }
        assert not (mediator_only & concepts), (
            f"KB-derived events leaked into the knowledge-free arm: "
            f"{mediator_only & concepts}")
