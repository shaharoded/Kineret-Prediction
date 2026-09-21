"""
Tests for the shared scorer and the cross-model input contract.

The scorer is what turns four different codebases into one comparable table, so
its schema contract and its handling of degenerate columns matter as much as the
metric maths.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from kineret.config import data_config as C

from kineret.evaluation import (
    time_to_event_mae,
    aggregate, bootstrap_metrics, length_of_stay_mae,
    per_outcome_metrics, score_run, write_predictions,
)


@pytest.fixture
def toy():
    """A 3-outcome, 200-patient problem: one easy, one random, one all-negative."""
    rng = np.random.default_rng(0)
    n = 200
    labels = np.zeros((n, 3))
    labels[:60, 0] = 1
    labels[:, 1] = rng.integers(0, 2, n)
    probs = np.column_stack([
        np.where(labels[:, 0] == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)),
        rng.uniform(0, 1, n),
        rng.uniform(0, 1, n),
    ])
    return labels, probs, ["easy", "random", "empty"]


class TestMetrics:
    def test_separable_outcome_scores_near_perfect(self, toy):
        labels, probs, names = toy
        table = per_outcome_metrics(labels, probs, names).set_index("outcome")
        assert table.loc["easy", "auroc"] > 0.95
        assert 0.35 < table.loc["random", "auroc"] < 0.65

    def test_single_class_outcome_is_nan_not_a_crash(self, toy):
        labels, probs, names = toy
        table = per_outcome_metrics(labels, probs, names).set_index("outcome")
        assert np.isnan(table.loc["empty", "auroc"])
        assert table.loc["empty", "n_pos"] == 0

    def test_weighted_average_is_the_support_weighted_mean(self, toy):
        """`weighted` weights each outcome by n_pos; `macro` weights them equally."""
        labels, probs, names = toy
        table = per_outcome_metrics(labels, probs, names)
        agg = aggregate(table).set_index("average")

        finite = table[table["auroc"].notna()]
        weights = finite["n_pos"] / table["n_pos"].sum()
        expected = (finite["auroc"] * weights).sum() / weights.sum()
        assert agg.loc["weighted", "auroc"] == pytest.approx(expected)
        assert agg.loc["macro", "auroc"] == pytest.approx(finite["auroc"].mean())

    def test_all_negative_outcome_is_excluded_from_both_averages(self, toy):
        """The empty outcome contributes nothing rather than dragging the mean to NaN."""
        labels, probs, names = toy
        agg = aggregate(per_outcome_metrics(labels, probs, names)).set_index("average")
        assert np.isfinite(agg.loc["macro", "auroc"])
        assert np.isfinite(agg.loc["weighted", "auroc"])

    def test_los_mae_ignores_patients_without_a_terminus(self):
        true = np.array([100.0, 200.0, np.nan, 300.0])
        pred = np.array([110.0, 180.0, 999.0, 300.0])
        mae, n = length_of_stay_mae(true, pred)
        assert n == 3
        assert mae == pytest.approx((10 + 20 + 0) / 3)


class TestBootstrap:
    def test_ci_brackets_the_point_estimate(self, toy):
        labels, probs, names = toy
        boot = bootstrap_metrics(labels, probs, names, n_resamples=100, seed=1)
        block = boot["overall"]["auroc"]["weighted"]
        assert block["lo"] <= block["mean"] <= block["hi"]

    def test_zero_resamples_skips_the_block(self, toy):
        labels, probs, names = toy
        assert bootstrap_metrics(labels, probs, names, n_resamples=0) == {}


class TestRunArtefacts:
    def test_write_then_score_round_trips(self, tmp_path, toy):
        """The schema every model writes is the schema the scorer reads."""
        labels, probs, names = toy
        run_dir = str(tmp_path / "model" / "k4_noqa")
        pids = np.arange(len(labels))
        los_true = np.full(len(labels), 120.0)
        write_predictions(run_dir, pids, labels, probs, names,
                          los_true=los_true, los_pred=los_true + 5.0)
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump({"model": "toy", "context_days": 4, "use_qa": False,
                       "outcome_names": names, "n_test": len(labels)}, f)

        result = score_run(run_dir, n_resamples=50, verbose=False)
        assert result["los_mae"] == pytest.approx(5.0)
        assert set(result["per_outcome"]["outcome"]) == set(names)
        for artefact in ("test_per_outcome_metrics.csv", "test_overall_metrics.csv",
                         "test_bootstrap.json"):
            assert os.path.exists(os.path.join(run_dir, artefact))

    def test_collect_runs_finds_every_finished_run(self, tmp_path, toy):
        labels, probs, names = toy
        for model in ("a", "b"):
            for qa in (True, False):
                run_dir = str(tmp_path / model / f"k4_{'qa' if qa else 'noqa'}")
                write_predictions(run_dir, np.arange(len(labels)), labels, probs, names)
                with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
                    json.dump({"model": model, "context_days": 4, "use_qa": qa,
                               "outcome_names": names, "n_test": len(labels)}, f)
                score_run(run_dir, n_resamples=0, verbose=False)

        from kineret.benchmark import load_results
        table = load_results(str(tmp_path), average="weighted")
        assert len(table) == 4              # 2 models x 2 arms
        assert set(table["n_test"]) == {len(labels)}




class TestBootstrapCoverage:
    """Every reported number should carry an interval where one is meaningful."""

    @pytest.fixture
    def timed(self):
        """A toy run with risk, onset and LoS, so all three families are present."""
        rng = np.random.default_rng(3)
        n, k = 120, 2
        labels = np.zeros((n, k))
        labels[:40, 0] = 1
        labels[:20, 1] = 1
        probs = np.clip(labels * 0.6 + rng.uniform(0, 0.4, (n, k)), 0, 1)
        time_true = np.where(labels > 0, rng.uniform(50, 300, (n, k)), np.nan)
        time_pred = time_true + rng.normal(0, 10, (n, k))
        return labels, probs, time_true, time_pred, ["A_EVENT", "B_EVENT"]

    def test_onset_interval_is_produced(self, timed):
        labels, probs, time_true, time_pred, names = timed
        boot = bootstrap_metrics(labels, probs, names,
                                 time_true=time_true, time_pred=time_pred,
                                 n_resamples=200, seed=5)
        assert "onset" in boot and "per_outcome_onset" in boot
        for average in ("macro", "weighted"):
            block = boot["onset"][average]
            assert block["lo"] <= block["mean"] <= block["hi"]

    def test_onset_interval_brackets_the_point_estimate(self, timed):
        labels, probs, time_true, time_pred, names = timed
        point = time_to_event_mae(time_true, time_pred, labels, names)
        boot = bootstrap_metrics(labels, probs, names,
                                 time_true=time_true, time_pred=time_pred,
                                 n_resamples=300, seed=5)
        lo = np.asarray(boot["per_outcome_onset"]["lo"])
        hi = np.asarray(boot["per_outcome_onset"]["hi"])
        for i, value in enumerate(point["time_mae_h"]):
            assert lo[i] <= value <= hi[i], f"{point['outcome'][i]} outside its interval"

    def test_onset_is_absent_when_the_model_has_no_time_head(self, timed):
        """A model without a time head must not silently get a fabricated interval."""
        labels, probs, _tt, _tp, names = timed
        boot = bootstrap_metrics(labels, probs, names, n_resamples=50, seed=5)
        assert "onset" not in boot and "per_outcome_onset" not in boot

    def test_thin_support_is_flagged_not_hidden(self, tmp_path, timed):
        """
        An outcome with a handful of positives still gets an estimate, but the
        row says the interval cannot be trusted -- at n=1 it collapses to a
        point, which would otherwise read as precision.
        """
        labels, probs, time_true, time_pred, names = timed
        labels[:, 1] = 0
        labels[0, 1] = 1                       # exactly one positive
        run_dir = str(tmp_path / "toy" / "k4_noqa")
        write_predictions(run_dir, np.arange(len(labels)), labels, probs, names,
                          time_true=time_true, time_pred=time_pred)
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump({"model": "toy", "context_days": 4, "use_qa": False,
                       "outcome_names": names, "n_test": len(labels)}, f)

        table = score_run(run_dir, n_resamples=200, verbose=False)["per_outcome"]
        thin = table[table["outcome"] == "B_EVENT"].iloc[0]
        rich = table[table["outcome"] == "A_EVENT"].iloc[0]
        assert not bool(thin["ci_reliable"])
        assert bool(rich["ci_reliable"])
        # The estimate itself survives -- it is flagged, not dropped.
        assert np.isfinite(thin["time_mae_h"])

    def test_per_outcome_csv_carries_intervals(self, tmp_path, timed):
        labels, probs, time_true, time_pred, names = timed
        run_dir = str(tmp_path / "toy" / "k4_noqa")
        write_predictions(run_dir, np.arange(len(labels)), labels, probs, names,
                          los_true=np.full(len(labels), 100.0),
                          los_pred=np.full(len(labels), 95.0),
                          time_true=time_true, time_pred=time_pred)
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump({"model": "toy", "context_days": 4, "use_qa": False,
                       "outcome_names": names, "n_test": len(labels)}, f)
        score_run(run_dir, n_resamples=100, verbose=False)

        table = pd.read_csv(os.path.join(run_dir, "test_per_outcome_metrics.csv"))
        for metric in ("auroc", "auprc", "best_f1", "f1_0_5", "minrp", "time_mae_h"):
            assert f"{metric}_lo" in table.columns and f"{metric}_hi" in table.columns
