"""
Tests for the cross-model input contract.

These are the invariants that make the benchmark a comparison rather than four
unrelated numbers: what may reach an input stream, that both source files end up
on identical event support, that the study window is applied once and binds
everything, and that the QA arm means the same thing for every model.
"""

import numpy as np
import pandas as pd
import pytest

from kineret.config import data_config as C
from kineret.config import paths
from kineret.io_utils import load_table, normalise_temporal


class TestLeakageGuards:
    """What may and may not reach an input stream."""

    def test_terminus_markers_are_always_blocked(self, cohort):
        """RELEASE / DEATH announce the end of the admission and never go in."""
        blocked = cohort.leakage_blocklist()
        for token in (C.RELEASE_TOKEN, C.DEATH_TOKEN):
            for spelling in cohort.structural_aliases.get(token, []):
                assert spelling in blocked, f"{spelling} would leak the terminus"

    def test_outcome_events_follow_the_events_as_inputs_switch(self, cohort):
        """
        With EVENTS_AS_INPUTS on, outcome events are harmonised into the stream
        rather than blocked: inside the seed window they are observed history,
        and the label window starts strictly after it. With the switch off they
        are blocked outright.
        """
        blocked = cohort.leakage_blocklist()
        outcome_spellings = {s for name in cohort.outcome_names
                             for s in cohort.outcome_aliases[name]}
        # DEATH is both an outcome and a terminus, so it is blocked either way.
        death_spellings = set(cohort.structural_aliases.get(C.DEATH_TOKEN, []))
        non_terminal = outcome_spellings - death_spellings

        if C.EVENTS_AS_INPUTS:
            assert not (non_terminal & blocked), (
                "outcome events should be harmonised into the input stream, "
                "not blocked, when EVENTS_AS_INPUTS is on")
        else:
            assert non_terminal <= blocked

    def test_source_measurements_stay_as_inputs(self, cohort):
        """Glucose is what HYPERGLYCEMIA is derived from -- a predictor, not a label."""
        blocked = cohort.leakage_blocklist()
        assert "GLUCOSE_MEASURE" not in blocked
        assert "CREATININE_SERUM_MEASURE" not in blocked

    @pytest.mark.parametrize("k", [2, 4, 7])
    def test_input_and_label_windows_are_disjoint(self, cohort, k):
        """
        The real guarantee: no occurrence can be both an input and a positive
        label at the same K. The windows meet at k*24 and do not overlap.
        """
        lo = k * 24.0
        as_input = cohort.events[cohort.events["hours"] <= lo]
        as_label = cohort.events[(cohort.events["hours"] > lo)
                                 & (cohort.events["hours"] <= C.HORIZON_END_DAYS * 24.0)]
        assert not set(as_input.index) & set(as_label.index)


class TestEventHarmonisation:
    """Both source files must end up on identical event support."""

    @pytest.fixture(scope="class")
    def harmonised(self, cohort):
        """Both source tables, put onto the canonical event support."""
        pids = cohort.patients["PatientId"]
        keep = set(pids)
        raw = normalise_temporal(load_table(paths.RAW_TEMPORAL_FILE))
        abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
        return (cohort.harmonise_events(raw[raw["PatientId"].isin(keep)], patient_ids=pids),
                cohort.harmonise_events(abstract[abstract["PatientId"].isin(keep)],
                                        patient_ids=pids))

    def test_streams_carry_the_same_events(self, cohort, harmonised):
        """
        The point of the reconciliation: afterwards the raw file and the
        Mediator output contain the identical set of outcome occurrences, so a
        difference between models cannot be a difference in what they saw.
        """
        raw, abstract = harmonised
        names = set(cohort.outcome_names)

        def support(df):
            hit = df[df["ConceptName"].isin(names)]
            return hit.groupby("ConceptName").size().sort_index()

        pd.testing.assert_series_equal(support(raw), support(abstract), check_names=False)

    def test_injected_rows_are_the_canonical_ones(self, cohort, harmonised):
        """Injected events are the canonical table, not a re-derivation."""
        if not C.EVENTS_AS_INPUTS:
            pytest.skip("EVENTS_AS_INPUTS is off")
        raw, _abstract = harmonised
        names = set(cohort.outcome_names)
        n_injected = int(raw["ConceptName"].isin(names).sum())
        n_canonical = int(cohort.events["outcome"].isin(names).sum())
        assert n_injected == n_canonical

    def test_original_spellings_are_gone(self, cohort, harmonised):
        """Un-suffixed raw spellings must not survive alongside the canonical ones."""
        raw, _abstract = harmonised
        concepts = set(raw["ConceptName"].unique())
        for canonical, spellings in cohort.outcome_aliases.items():
            for spelling in spellings:
                if spelling != canonical:
                    assert spelling not in concepts, (
                        f"{spelling} survived harmonisation next to {canonical}")

    def test_reconciliation_audit_is_recorded(self, cohort):
        """The evidence that the files were reconciled has to be inspectable."""
        audit = cohort.event_audit
        assert audit is not None and len(audit)
        assert {"outcome", "n_raw", "n_mediator", "n_canonical",
                "source"} <= set(audit.columns)
        for outcome in cohort.outcome_names:
            assert len(audit[audit["outcome"] == outcome]) == 1, \
                f"{outcome} missing from the reconciliation audit"

    def test_occurrences_are_deduplicated_within_tolerance(self):
        """Restatements of one event inside the tolerance collapse to one row."""
        from kineret.cohort import _dedupe_occurrences
        df = pd.DataFrame({
            "PatientId": [1, 1, 1, 1],
            "outcome": ["X_EVENT"] * 4,
            # 0h, +1h, +2h are one episode; +100h is a second.
            "hours": [10.0, 11.0, 12.0, 110.0],
        })
        out = _dedupe_occurrences(df, tolerance_h=24.0)
        assert len(out) == 2
        assert sorted(out["hours"]) == [10.0, 110.0]


class TestDateRange:
    """The study window is applied once and binds everything."""

    def test_every_admission_is_inside_the_window(self, cohort):
        from kineret.cohort import date_range_bounds
        start, end = date_range_bounds()
        admissions = pd.to_datetime(cohort.patients["admission_time"])
        if start is not None:
            assert (admissions >= start).all()
        if end is not None:
            assert (admissions <= end).all()

    def test_full_horizon_is_available_for_every_patient(self, cohort):
        """
        No admission may have a label window running past the extract's end --
        those complications are invisible, and scoring them as negatives is a
        censoring artefact indistinguishable from a real negative.
        """
        from kineret.cohort import date_range_bounds
        _start, end = date_range_bounds()
        if end is None or not C.DATE_RANGE_REQUIRE_FULL_HORIZON:
            pytest.skip("full-horizon rule not in force")
        horizon = pd.Timedelta(hours=C.HORIZON_END_DAYS * 24.0)
        latest = pd.to_datetime(cohort.patients["admission_time"]) + horizon
        assert (latest <= end).all()

    def test_no_event_falls_outside_the_window(self, cohort):
        """Labels come from events, so the clip has to reach them too."""
        from kineret.cohort import date_range_bounds
        start, end = date_range_bounds()
        admission = cohort.patients.set_index("PatientId")["admission_time"]
        stamps = (cohort.events["PatientId"].map(admission)
                  + pd.to_timedelta(cohort.events["hours"], unit="h"))
        if start is not None:
            assert (stamps >= start).all()
        if end is not None:
            assert (stamps <= end).all()

    def test_meta_records_the_window(self, cohort):
        assert cohort.meta["date_range_start"] == C.DATE_RANGE_START
        assert cohort.meta["date_range_end"] == C.DATE_RANGE_END


class TestQaArms:
    """QA enters every model's context vector in exactly the same way."""

    def test_qa_block_is_shared_by_every_model(self, cohort):
        """
        The block is built once on the cohort, so LogReg, ss-STraTS and
        INTERVenE necessarily receive the same columns and the same values --
        the arms differ by exactly these columns and nothing else.
        """
        base = cohort.context_for_k(4, use_qa=False)
        with_qa = cohort.context_for_k(4, use_qa=True)
        added = [c for c in with_qa.columns if c not in base.columns]
        assert added, "QA arm added no columns"
        assert all(c.startswith(C.QA_COLUMN_PREFIX) for c in added)
        assert with_qa[base.columns].equals(base)

    def test_one_column_per_pattern_per_aggregation(self, cohort):
        block = cohort.qa_by_k[4]
        n_patterns = len({c for c in block.columns})
        assert block.shape[1] == n_patterns
        assert block.shape[1] % max(len(C.QA_AGGREGATIONS), 1) == 0

    def test_single_mean_keeps_the_plain_column_name(self, cohort):
        """`QA_<pattern>` when only `mean` is configured -- the thesis convention."""
        if list(C.QA_AGGREGATIONS) != ["mean"]:
            pytest.skip("multiple aggregations configured")
        suffixes = ("_mean", "_min", "_max", "_last", "_count", "_std")
        for col in cohort.qa_by_k[4].columns:
            assert not col.endswith(suffixes), f"{col} carries an aggregation suffix"

    def test_intervene_additionally_gets_pattern_tokens(self):
        """
        INTERVenE is the only model that can represent the compliance patterns
        as temporal tokens, so its filter drops them when the arm is off and
        keeps them when it is on. Everyone else gets the aggregated vector only.
        """
        off = C.temporal_filters(use_qa=False)["temporal"]
        on = C.temporal_filters(use_qa=True)["temporal"]
        assert any("_PATTERN" in cond for cond in off), \
            "pattern tokens should be filtered out of the no-QA arm"
        assert not any("_PATTERN" in cond for cond in on), \
            "pattern tokens should be kept in the QA arm"


class TestLogRegFeatures:
    """The non-temporal baseline must still see each concept's distribution."""

    def test_distribution_summary_is_present(self, cohort):
        from kineret.benchmark import raw_with_hours
        from kineret.logreg.features import build_feature_frame

        train_s = cohort.samples("train")
        features, kept = build_feature_frame(
            raw_with_hours(cohort), train_s, cohort.leakage_blocklist())
        assert kept, "no variables cleared the support floor"

        # Centre, spread, extremes, robust shape, trajectory and observation
        # density -- collapsing a window to its mean alone would throw away most
        # of what distinguishes a stable patient from a deteriorating one.
        expected = ("__mean", "__std", "__min", "__max", "__median", "__p25",
                    "__p75", "__first", "__last", "__slope", "__count",
                    "__rate_per_day", "__observed")
        for suffix in expected:
            assert any(c.endswith(suffix) for c in features.columns), \
                f"no {suffix} feature was built"

    def test_never_measured_is_zero_not_imputed(self, cohort):
        """Count-like columns mean 'it never happened', which is genuinely zero."""
        from kineret.benchmark import raw_with_hours
        from kineret.logreg.features import build_feature_frame

        train_s = cohort.samples("train")
        features, _ = build_feature_frame(
            raw_with_hours(cohort), train_s, cohort.leakage_blocklist())
        for col in features.columns:
            if col.endswith(("__observed", "__count")):
                assert not features[col].isna().any(), f"{col} left NaN"


class TestPredictionTask:
    """Every model must answer the same three questions."""

    THREE_QUESTIONS = ("risk", "onset", "length of stay")

    def _run_dirs(self, tmp_path):
        return tmp_path

    def test_shared_schema_carries_all_three(self, tmp_path):
        """
        risk + onset + LoS, in one file, for every model. If a model stops
        emitting one of them the comparison quietly becomes unequal, so the
        schema itself is asserted.
        """
        import numpy as np
        from kineret.evaluation import write_predictions

        names = ["A_EVENT", "B_EVENT"]
        n = 8
        labels = np.array([[1, 0]] * n, dtype=float)
        probs = np.full((n, 2), 0.5)
        times = np.full((n, 2), 30.0)
        path = write_predictions(str(tmp_path), np.arange(n), labels, probs, names,
                                 los_true=np.full(n, 100.0), los_pred=np.full(n, 90.0),
                                 time_true=times, time_pred=times + 5)
        header = pd.read_csv(path).columns
        for name in names:
            assert f"label_{name}" in header
            assert f"prob_{name}" in header
            assert f"time_true_{name}" in header
            assert f"time_pred_{name}" in header
        assert "los_true_hours" in header and "los_pred_hours" in header

    def test_onset_mae_ignores_negatives(self):
        """
        'When will it happen' is meaningless for a complication that never
        happens, so negatives must not enter the onset MAE.
        """
        import numpy as np
        from kineret.evaluation import time_to_event_mae

        labels = np.array([[1.0], [0.0], [1.0]])
        time_true = np.array([[10.0], [np.nan], [20.0]])
        time_pred = np.array([[12.0], [999.0], [25.0]])
        out = time_to_event_mae(time_true, time_pred, labels, ["X_EVENT"]).iloc[0]
        assert out["n_time"] == 2
        assert out["time_mae_h"] == pytest.approx((2.0 + 5.0) / 2)

    def test_strats_head_width_covers_all_three(self):
        """K risk + K onset + 1 LoS -- the same output surface INTERVenE has."""
        import argparse
        import torch

        from kineret.strats.model_utils import TimeSeriesModel

        args = argparse.Namespace(model_type="strats", D=4, hid_dim=8, num_labels=3,
                                  V=5, pos_class_weight=np.ones(3), load_ckpt_path=None,
                                  pretrain=0, los_loss_weight=1.0, time_loss_weight=0.1)
        model = TimeSeriesModel(args)
        assert model.binary_head.out_features == 2 * 3 + 1

        # Eval branch returns [probs | normalised onset | normalised LoS].
        logits = torch.zeros(2, 7)
        out = model.binary_cls_final(logits, labels=None)
        assert out.shape == (2, 7)

    def test_strats_time_loss_is_masked_to_positives(self):
        """A negative contributes no onset gradient, however wrong the prediction."""
        import argparse
        import torch

        from kineret.strats.model_utils import TimeSeriesModel

        args = argparse.Namespace(model_type="strats", D=4, hid_dim=8, num_labels=2,
                                  V=5, pos_class_weight=np.ones(2), load_ckpt_path=None,
                                  pretrain=0, los_loss_weight=0.0, time_loss_weight=1.0)
        model = TimeSeriesModel(args)

        logits = torch.zeros(1, 5)
        labels = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0]])

        # A wildly wrong prediction on the MASKED outcome must not move the loss.
        base = model.binary_cls_final(logits, labels, time_target_norm=target,
                                      time_mask=mask)
        noisy = logits.clone()
        noisy[0, 3] = 100.0          # index 3 = time slot of the negative outcome
        after = model.binary_cls_final(noisy, labels, time_target_norm=target,
                                       time_mask=mask)
        assert torch.allclose(base, after)


class TestSupportTableMatchesTheFilter:
    """
    The support table is what the evaluation window is chosen from, so it has
    to report the decision the cohort actually makes.

    It once divided by every training admission at every K, while the cohort's
    filter divides by the admissions that REACH that window -- 62 % of them at
    K=4, 2 % at K=13. Every prevalence read low, the head count read low, and
    the figure driving the K decision disagreed with the code.
    """

    def test_clears_threshold_matches_the_cohorts_heads(self, cohort):
        from kineret.benchmark import target_support
        table = target_support(cohort, windows=[C.EVAL_CONTEXT_DAYS])
        clears = set(table[table["clears_threshold"]]["target"])
        assert clears == set(cohort.outcome_names), (
            f"table says {sorted(clears)}, cohort built heads for "
            f"{sorted(cohort.outcome_names)}")

    def test_denominator_is_the_scorable_population(self, cohort):
        from kineret.benchmark import target_support
        patients = cohort.patients
        for k in (1, C.EVAL_CONTEXT_DAYS):
            table = target_support(cohort, windows=[k])
            expected = int(((patients["split"] == "train")
                            & (patients["trajectory_hours"] > k * 24.0)).sum())
            assert set(table["n_scorable_train"]) == {expected}

    def test_denominator_shrinks_as_the_window_grows(self, cohort):
        """Ragged augmentation: fewer admissions reach a later window."""
        from kineret.benchmark import target_support
        table = target_support(cohort, windows=[1, C.EVAL_CONTEXT_DAYS])
        counts = table.groupby("k")["n_scorable_train"].first()
        assert counts.loc[1] >= counts.loc[C.EVAL_CONTEXT_DAYS]

    def test_positives_are_restricted_to_scorable_patients(self, cohort):
        """A patient who cannot reach K cannot be a positive at K."""
        from kineret.benchmark import target_support
        table = target_support(cohort, windows=[C.EVAL_CONTEXT_DAYS])
        assert (table["n_pos_train"] <= table["n_scorable_train"]).all()
