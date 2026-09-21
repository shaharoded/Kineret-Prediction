"""
Tests for the shared cohort layer.

These are the load-bearing invariants of the whole benchmark: if the label
window drifts, or the split moves between K values, every cross-model and
cross-K comparison in the results table silently becomes meaningless.
"""

import numpy as np
import pytest

from kineret.config import data_config as C


class TestAliasResolution:
    """The raw table and the Mediator output spell complications differently."""

    def test_alias_regexes_cover_both_spellings(self):
        """
        The alias table must still bridge the two conventions. The raw ETL
        writes `CARDIOVASCULAR_DISORDER` where the Mediator writes
        `CARDIO-VASCULAR_DISORDER_EVENT`.

        Tested on the pattern rather than on the built cohort: raw-side event
        derivation now rewrites the raw file's own spellings to the canonical
        name, so by the time a Cohort exists only one spelling survives. The
        regex is what still has to handle an extract this package has not
        derived -- and what `resolve_concepts` leans on to find a rule's source.
        """
        import re
        for canonical, raw_spelling in (
                ("CARDIO-VASCULAR_DISORDER_EVENT", "CARDIOVASCULAR_DISORDER"),
                ("KETOACIDOSIS_EVENT", "KETOACIDOSIS")):
            pattern = C.OUTCOME_ALIAS_REGEX[canonical]
            assert re.fullmatch(pattern, raw_spelling, re.IGNORECASE)
            assert re.fullmatch(pattern, canonical, re.IGNORECASE)

    def test_derivation_leaves_one_canonical_spelling(self, cohort):
        """
        After derivation the raw file carries the canonical `*_EVENT` name and
        nothing else, so both streams name every target identically.
        """
        for target in ("KETOACIDOSIS_EVENT", "CARDIO-VASCULAR_DISORDER_EVENT"):
            assert cohort.outcome_aliases[target] == [target], (
                f"{target} still has multiple spellings: "
                f"{cohort.outcome_aliases[target]}")

    def test_structural_tokens_resolve(self, cohort):
        """ADMISSION / RELEASE / DEATH are found in at least one table."""
        for token in (C.ADMISSION_TOKEN, C.RELEASE_TOKEN, C.DEATH_TOKEN):
            assert cohort.structural_aliases[token], f"no spelling found for {token}"

    def test_mediator_is_preferred_over_raw(self, cohort):
        """Where both tables carry an outcome, the Mediator's verdict wins."""
        both = cohort.events[cohort.events["outcome"] == "KETOACIDOSIS_EVENT"]
        assert set(both["source"]) == {"mediator"}


class TestLabelWindow:
    """Labels must be exactly (K*24, HORIZON_END_DAYS*24] hours from admission."""

    @pytest.mark.parametrize("k", [2, 4, 7])
    def test_positive_iff_event_in_window(self, cohort, k):
        """Every positive has an occurrence in the window; every negative has none."""
        lo, hi = k * 24.0, C.HORIZON_END_DAYS * 24.0
        labels = cohort.labels_for_k(k).set_index("PatientId")
        in_window = cohort.events[(cohort.events["hours"] > lo)
                                  & (cohort.events["hours"] <= hi)]
        for outcome in cohort.outcome_names:
            expected = set(in_window.loc[in_window["outcome"] == outcome, "PatientId"])
            actual = set(labels.index[labels[outcome] == 1])
            assert actual == expected, f"{outcome} mismatch at K={k}"

    @pytest.mark.parametrize("k", [2, 5])
    def test_events_inside_the_seed_are_not_labels(self, cohort, k):
        """An outcome that only fires inside [0, K*24] is a negative, not a positive."""
        lo = k * 24.0
        labels = cohort.labels_for_k(k).set_index("PatientId")
        for outcome in cohort.outcome_names:
            occurrences = cohort.events[cohort.events["outcome"] == outcome]
            seed_only = (occurrences.groupby("PatientId")["hours"].max() <= lo)
            for pid in seed_only.index[seed_only]:
                assert labels.loc[pid, outcome] == 0, (
                    f"{outcome} for {pid} fires only inside the K={k} seed but is labelled positive")

    def test_first_hour_is_inside_the_window(self, cohort):
        """`__first_hour` is the earliest occurrence within the window, not before it."""
        k = 4
        lo, hi = k * 24.0, C.HORIZON_END_DAYS * 24.0
        labels = cohort.labels_for_k(k)
        for outcome in cohort.outcome_names:
            hours = labels.loc[labels[outcome] == 1, f"{outcome}__first_hour"]
            assert (hours > lo).all() and (hours <= hi).all()
            negatives = labels.loc[labels[outcome] == 0, f"{outcome}__first_hour"]
            assert np.isinf(negatives).all()

    def test_windows_shrink_monotonically_with_k(self, cohort):
        """A larger K can only remove positives -- the window's left edge moves right."""
        for outcome in cohort.outcome_names:
            counts = [cohort.labels_for_k(k)[outcome].sum() for k in (2, 4, 7)]
            assert counts == sorted(counts, reverse=True), f"{outcome}: {counts}"


class TestSplits:
    """One split, frozen, reused by every model and every K."""

    def test_splits_are_disjoint_and_complete(self, cohort):
        train, val, test = cohort.split_ids(4)
        assert not (set(train) & set(val)) and not (set(val) & set(test))
        assert not (set(train) & set(test))
        assert len(train) + len(val) + len(test) == len(cohort.patients)

    def test_split_assignment_is_stable_across_k(self, cohort):
        """A patient never changes split when K changes."""
        by_k = {}
        for k in C.TRAIN_CONTEXT_DAYS:
            train, val, test = cohort.split_ids(k)
            for name, ids in (("train", train), ("val", val), ("test", test)):
                for pid in ids:
                    assert by_k.setdefault(pid, name) == name, (
                        f"patient {pid} moved to {name} at K={k}")

    def test_membership_is_keyed_on_the_evaluation_window(self, cohort):
        """
        Every patient must be scorable at the evaluation window. Membership must
        NOT depend on the longest training window -- that would make admission
        to the study depend on how wide the augmentation happens to be.
        """
        assert (cohort.patients["trajectory_hours"]
                > C.EVAL_CONTEXT_DAYS * 24.0).all()

    def test_widening_the_augmentation_does_not_shrink_the_cohort(self, cohort):
        """
        The point of keying membership on the evaluation window: adding longer
        training windows adds samples, never removes patients.
        """
        before = len(cohort.patients)
        narrow = len(cohort.samples("train"))
        original = list(C.TRAIN_CONTEXT_DAYS)
        try:
            C.configure_study(train_context_days=sorted(set(original) | {13}))
            assert len(cohort.patients) == before
            assert len(cohort.samples("train")) >= narrow
        finally:
            C.configure_study(train_context_days=original)


class TestContextAndQA:
    """The QA arm must differ from the base arm by exactly the QA columns."""

    def test_qa_adds_only_qa_columns(self, cohort):
        base = cohort.context_for_k(4, use_qa=False)
        with_qa = cohort.context_for_k(4, use_qa=True)
        added = [c for c in with_qa.columns if c not in base.columns]
        assert added, "QA arm added no columns"
        assert all(c.startswith(C.QA_COLUMN_PREFIX) for c in added)
        assert with_qa[base.columns].equals(base)

    def test_qa_column_layout_is_identical_across_k(self, cohort):
        layouts = {k: list(cohort.qa_by_k[k].columns) for k in C.TRAIN_CONTEXT_DAYS}
        assert len({tuple(v) for v in layouts.values()}) == 1

    def test_qa_uses_only_the_observation_window(self, cohort):
        """QA at K=2 cannot depend on data a K=2 model has not seen."""
        small, large = cohort.qa_by_k[2], cohort.qa_by_k[7]
        assert not small.equals(large), (
            "QA features are identical at K=2 and K=7 -- the aggregation window "
            "is not following K, which would leak future compliance into the seed.")

    def test_context_has_no_missing_values(self, cohort):
        ctx = cohort.context_for_k(4, use_qa=True)
        assert not ctx.isna().any().any()


class TestOutcomeSelection:
    """Head layout is decided once, on train, at the tightest window."""

    def test_kept_targets_clear_the_threshold_at_the_eval_window(self, cohort):
        """The head is decided at the window the results are reported from."""
        train_ids = set(cohort.patients.loc[cohort.patients["split"] == "train",
                                            "PatientId"])
        labels = cohort.labels_for_k(C.EVAL_CONTEXT_DAYS)
        labels = labels[labels["PatientId"].isin(train_ids)]
        for target in cohort.outcome_names:
            assert labels[target].mean() >= C.OUTCOME_SUPPORT_THRESHOLD

    def test_dropped_outcomes_are_below_the_threshold(self, cohort):
        for _outcome, prevalence in cohort.dropped_outcomes.items():
            assert prevalence < C.OUTCOME_SUPPORT_THRESHOLD


class TestSampleAugmentation:
    """Training is augmented over context windows; evaluation is not."""

    def test_train_has_one_sample_per_reachable_window(self, cohort):
        """
        Ragged augmentation: a patient contributes a window only if their stay
        reaches it. Cutting a 5-day admission at day 7 would train the model on
        a window that does not exist.
        """
        samples = cohort.samples("train")
        reach = cohort.patients.set_index("PatientId")["trajectory_hours"]
        for pid, group in samples.groupby("PatientId"):
            expected = {k for k in C.TRAIN_CONTEXT_DAYS if reach[pid] > k * 24.0}
            assert set(group["k"]) == expected, f"wrong windows for patient {pid}"

    def test_no_sample_extends_past_its_patient_stay(self, cohort):
        """The invariant the raggedness exists to protect."""
        samples = cohort.samples()
        reach = cohort.patients.set_index("PatientId")["trajectory_hours"]
        assert (samples["k"] * 24.0 < reach.reindex(samples["PatientId"]).to_numpy()).all()

    def test_short_stays_contribute_fewer_samples(self, cohort):
        """Longer stays offer more distinct cuts, so they weigh more. By design."""
        samples = cohort.samples("train")
        counts = samples.groupby("PatientId").size()
        reach = cohort.patients.set_index("PatientId")["trajectory_hours"].reindex(counts.index)
        # Monotone, not strictly: several windows can sit inside one stay.
        assert counts.corr(reach) > 0.5

    def test_val_and_test_use_only_the_eval_window(self, cohort):
        """
        Model selection and reporting must describe ONE stated window --
        otherwise "which K" becomes a hidden degree of freedom in the results.
        """
        for split in ("val", "test"):
            samples = cohort.samples(split)
            assert set(samples["k"]) == {C.EVAL_CONTEXT_DAYS}
            assert samples["PatientId"].is_unique

    def test_sample_ids_are_unique(self, cohort):
        samples = cohort.samples()
        assert samples["sample_id"].is_unique

    def test_a_patient_never_spans_two_splits(self, cohort):
        samples = cohort.samples()
        per_patient = samples.groupby("PatientId")["split"].nunique()
        assert (per_patient == 1).all()

    def test_sample_labels_follow_that_sample_window(self, cohort):
        """
        The whole point of the augmentation: a patient's K=2 row and K=7 row
        carry different labels, because their label windows differ.
        """
        samples = cohort.samples("train")
        labels = cohort.labels_for_samples(samples).set_index("sample_id")
        target = cohort.outcome_names[0]
        for k in (min(C.TRAIN_CONTEXT_DAYS), max(C.TRAIN_CONTEXT_DAYS)):
            rows = samples[samples["k"] == k]
            direct = cohort.labels_for_k(k).set_index("PatientId")
            got = labels.loc[rows["sample_id"], target].to_numpy()
            want = direct.loc[rows["PatientId"], target].to_numpy()
            assert (got == want).all(), f"labels wrong for K={k}"

    def test_augmentation_can_only_shrink_positives_with_k(self, cohort):
        """
        A later cut cannot create a positive a shorter window missed -- for the
        SAME patient. Compared per patient, since the ragged augmentation means
        the K=13 rows are a different (longer-staying) subset.
        """
        samples = cohort.samples("train")
        labels = cohort.labels_for_samples(samples).set_index("sample_id")
        merged = samples.set_index("sample_id").join(
            labels[cohort.outcome_names].sum(axis=1).rename("n_pos"))
        for _pid, group in merged.groupby("PatientId"):
            ordered = group.sort_values("k")["n_pos"].to_numpy()
            assert (np.diff(ordered) <= 0).all(), "positives grew with K"

    def test_sample_context_uses_that_sample_window(self, cohort):
        """
        A K=2 sample's QA block must summarise two days, not seven -- otherwise
        the augmentation would leak future compliance into the short windows.
        """
        samples = cohort.samples("train")
        ctx = cohort.context_for_samples(samples, use_qa=True)
        qa_cols = [c for c in ctx.columns if c.startswith(C.QA_COLUMN_PREFIX)]
        assert qa_cols, "no QA columns to check"
        short = samples[samples["k"] == min(C.TRAIN_CONTEXT_DAYS)]["sample_id"]
        long = samples[samples["k"] == max(C.TRAIN_CONTEXT_DAYS)]["sample_id"]
        assert not ctx.loc[short, qa_cols].to_numpy().flatten().tolist() == \
            ctx.loc[long, qa_cols].to_numpy().flatten().tolist()

    def test_eval_window_must_be_a_training_window(self):
        """A model cannot be judged at a window it was never trained on."""
        import pytest as _pytest
        original = list(C.TRAIN_CONTEXT_DAYS)
        with _pytest.raises(ValueError, match="not in"):
            C.configure_study(train_context_days=[2, 3], eval_context_days=5)
        C.configure_study(train_context_days=original, eval_context_days=4)

    def test_a_window_past_the_horizon_is_rejected(self):
        """(K, 14] is empty at K >= 14 -- nothing to forecast, nothing to learn."""
        import pytest as _pytest
        original = list(C.TRAIN_CONTEXT_DAYS)
        with _pytest.raises(ValueError, match="past"):
            C.configure_study(train_context_days=original + [14])
        C.configure_study(train_context_days=original)

    def test_context_carries_the_window_lengths(self, cohort):
        """
        With a fixed horizon end, K sets both how much history the sample got
        and how far ahead it forecasts. Both are handed over explicitly.
        """
        if not C.ADD_CONTEXT_LENGTH_FEATURES:
            pytest.skip("length features disabled")
        samples = cohort.samples("train")
        ctx = cohort.context_for_samples(samples, use_qa=False)
        assert {"context_days", "label_window_days"} <= set(ctx.columns)
        got = ctx.loc[samples["sample_id"], "context_days"].to_numpy()
        assert (got == samples["k"].to_numpy()).all()
        window = ctx.loc[samples["sample_id"], "label_window_days"].to_numpy()
        assert np.allclose(window, C.HORIZON_END_DAYS - samples["k"].to_numpy())

    def test_unknown_study_key_is_rejected(self):
        """A typo in the notebook's config cell must not silently run defaults."""
        import pytest as _pytest
        with _pytest.raises(ValueError, match="Unknown study setting"):
            C.configure_study(eval_contxt_days=4)
