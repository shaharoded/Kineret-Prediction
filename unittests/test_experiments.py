"""Research-contract tests: dependence, held-out splits, frozen inference, and calibration."""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from kineret.experiments.cohorts import (date_subset, site_split,
                                       people, evaluation_window, validate_split)
from kineret.experiments.statistics import (cluster_indices, paired_contrasts, calibration,
                                           score_predictions, attach_people)
from kineret.experiments.stability import interval_agreement, summarize


@pytest.fixture
def repeated_cohort():
    from kineret.cohort import Cohort
    ids = np.arange(20)
    dates = pd.to_datetime(["2022-07-05", "2022-08-01"] + ["2024-01-01"]*14 +
                           ["2024-12-25", "2025-01-01", "2025-02-01", "2025-12-30"])
    patients = pd.DataFrame(dict(PatientId=ids, admission_time=dates, trajectory_hours=240.,
                                 los_hours=240., split=["train"]*12+["val"]*4+["test"]*4))
    return Cohort(patients=patients,
        events=pd.DataFrame(dict(PatientId=ids, outcome="A", hours=120.)),
        outcome_names=["A"], dropped_outcomes={}, context=pd.DataFrame({"age": 60., "sex": 1.}, index=pd.Index(ids, name="PatientId")),
        qa_by_k={}, outcome_aliases={"A": ["A"]}, structural_aliases={},
        person_of_patient={i: (17 if i == 2 else i) for i in ids})


def test_cluster_bootstrap_keeps_repeated_admissions_together():
    for idx in cluster_indices([1, 1, 2, 3, 3, 3], 30):
        counts = np.bincount(idx, minlength=6)
        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]


def test_date_filter_preserves_splits_and_original(repeated_cohort):
    original = repeated_cohort.patients.copy()
    result = date_subset(repeated_cohort)
    assert 0 not in set(result.patients.PatientId)
    assert 1 in set(result.patients.PatientId)
    assert 19 not in set(result.patients.PatientId)  # incomplete day-14 horizon
    pd.testing.assert_frame_equal(original, repeated_cohort.patients)
    assert result.outcome_names == repeated_cohort.outcome_names


def test_loo_retains_target_site_and_excludes_its_people(repeated_cohort, tmp_path):
    visits = tmp_path/"visits.csv"
    pd.DataFrame(dict(visit_id=range(20), hospital=["A"]*17+["B"]*3)).to_csv(visits, index=False)
    fold = site_split(repeated_cohort, "B", visits)
    assert set(fold.patients.loc[fold.patients.split == "test", "PatientId"]) == {17, 18, 19}
    assert 2 not in set(fold.patients.PatientId)  # same person as admission 17
    assert "hospital" not in fold.context
    validate_split(fold)


def _predictions():
    return pd.DataFrame(dict(PatientId=range(40), PersonId=np.repeat(range(20), 2),
        label_A=np.tile([0, 1], 20), prob_A=np.tile([.2, .8], 20)))


def test_paired_identity_and_mismatched_labels():
    a = _predictions()
    result = paired_contrasts({"a": a, "b": a.sample(frac=1, random_state=3)}, [("same", "a", "b")], 20)
    assert (result.delta == 0).all() and (result.p_value == 1).all()
    assert (result.lo == 0).all() and (result.hi == 0).all()
    bad = a.copy(); bad.loc[0, "label_A"] = 1
    with pytest.raises(ValueError, match="identical"):
        paired_contrasts({"a": a, "b": bad}, [("bad", "a", "b")], 20)


def test_calibration_never_fits_on_test(monkeypatch):
    import kineret.experiments.statistics as S
    a = _predictions()
    def forbidden(*args, **kwargs): raise AssertionError("test self-fit")
    monkeypatch.setattr(S, "fit_temperature", forbidden)
    table, curves = calibration(a)
    assert set(table.variant) == {"raw"}
    assert table.brier.iloc[0] == pytest.approx(.04)
    with pytest.raises(ValueError, match="overlap"):
        calibration(a, a)


def test_constant_probability_reliability_retains_every_row():
    from kineret.evaluation import reliability_data
    curve = reliability_data([0, 0, 1, 1], [.5]*4)
    assert curve.n.sum() == 4 and curve.mean_obs.iloc[0] == .5


def test_metric_ci_flags_single_class():
    a = _predictions(); a.label_A = 0
    result = score_predictions(a, 12)
    roc = result[(result.outcome == "A") & (result.metric == "auroc")].iloc[0]
    assert np.isnan(roc.estimate) and roc.valid_resamples == 0 and not roc.ci_reliable


def test_interval_metric_counts_label_changes_and_lost_coverage():
    result = interval_agreement([(0, 10, "low"), (10, 20, "high")], [(0, 5, "low"), (5, 15, "high")])
    assert result == dict(baseline=20, altered=15, overlap=15, matching=10, union=20)


def test_stability_repeats_are_not_independent_people():
    sample = pd.DataFrame(dict(PatientId=[1, 2], weight=[1., 1.], stratum=["a", "a"]))
    detail = pd.DataFrame([dict(condition="same", family="state", PatientId=p, repeat=r,
                               baseline=10., altered=10., overlap=10., matching=10., union=10.)
                           for p in [1, 2] for r in [0, 1]])
    result = summarize(detail, sample, 20)
    assert result.iloc[0].estimate == 1 and result.iloc[0].lo == 1
    assert (result.n_people == 2).all()


def test_notebook_python_and_single_activation_point():
    root = Path(__file__).resolve().parents[1]
    for path in (root/"notebooks").glob("*.ipynb"):
        if "reference" in path.name: continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        for c in nb["cells"]:
            if c["cell_type"] == "code": ast.parse("".join(c["source"]))


@pytest.fixture
def adapter_cohort(cohort):
    result = deepcopy(cohort)
    result.person_of_patient = {i: i for i in result.patients.PatientId}
    return result


def test_logreg_bundle_reuses_weights_at_multiple_k(adapter_cohort, tmp_path, monkeypatch):
    import torch
    from kineret.logreg.train import run
    from kineret.experiments.inference import predict_arm
    output = run(cohort=adapter_cohort, output_root=str(tmp_path), device="cpu",
                 overrides={"max_epochs": 2, "batch_size": 64}, bootstrap_resamples=0)
    checkpoint = Path(output["output_dir"])/"model_bundle.pt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def forbidden(*a, **kw): raise AssertionError("Inference called an optimizer")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    result = predict_arm("logreg", adapter_cohort, checkpoint, k=4, device="cpu")
    original = pd.read_csv(Path(output["output_dir"])/"test_predictions.csv").set_index("PatientId")
    cols = [c for c in result if c.startswith("prob_")]
    np.testing.assert_allclose(result.set_index("PatientId")[cols], original.loc[result.PatientId, cols], atol=1e-6)
    other = predict_arm("logreg", adapter_cohort, checkpoint, k=2, device="cpu")
    assert len(other) >= len(result)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_strats_bundle_inference(adapter_cohort, tmp_path, monkeypatch):
    import torch
    from kineret.strats import config as SC
    from kineret.strats.preprocess import build_pickle
    from kineret.strats.train import run
    from kineret.experiments.inference import predict_arm
    monkeypatch.setattr(SC, "PROCESSED_DIR", str(tmp_path/"processed"))
    data_path = build_pickle(adapter_cohort, False, verbose=False)
    output = run(output_root=str(tmp_path), dataset_path=data_path, device="cpu",
        overrides=dict(hid_dim=8, num_heads=2, num_layers=1, max_epochs=1, max_obs=16,
                       train_batch_size=128, eval_batch_size=64), bootstrap_resamples=0)
    checkpoint = Path(output["output_dir"])/"checkpoint_best.bin"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def forbidden(*a, **kw): raise AssertionError("Inference called an optimizer")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    result = predict_arm("strats", adapter_cohort, checkpoint, k=4, device="cpu")
    original = pd.read_csv(Path(output["output_dir"])/"test_predictions.csv").set_index("PatientId")
    cols = [c for c in result if c.startswith("prob_")]
    np.testing.assert_allclose(result.set_index("PatientId")[cols], original.loc[result.PatientId, cols], atol=1e-5)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_chunked_source_read_uses_admission_identity_and_prefix(tmp_path):
    from kineret.experiments.io import load_temporal
    source = tmp_path/"raw.csv"
    pd.DataFrame(dict(PatientId=[99, 99, 99, 55], VisitId=[1, 1, 1, 2],
                     ConceptName="GLUCOSE", Value=[120, 130, 999, 140],
                     StartTime=["2024-01-01", "2024-01-04", "2024-01-07", "2024-01-01"])).to_csv(source, index=False)
    frame = load_temporal(source, [1], pd.Series({1: pd.Timestamp("2024-01-01")}), max_hours=96, chunksize=1)
    assert set(frame.PatientId) == {1} and len(frame) == 2
    assert 999 not in set(frame.Value)


def test_cached_evaluation_preserves_source_and_skips_rescoring(repeated_cohort, tmp_path, monkeypatch):
    from kineret.experiments import workflow as W
    from kineret.experiments.inference import prediction_frame
    from kineret.benchmark import run_dir_for
    original = tmp_path/"original"
    run_dir = Path(run_dir_for("logreg", str(original)))
    run_dir.mkdir(parents=True)
    samples = repeated_cohort.samples("test")
    frame = prediction_frame(repeated_cohort, samples, np.full((len(samples), 1), .3))
    source = run_dir/"test_predictions.csv"
    frame.to_csv(source, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path/"new"
    result = W.evaluate(repeated_cohort, output, ["logreg"], source_outputs=str(original),
                        include_validation=False, n_resamples=8)
    assert "logreg" in result[0]
    def forbidden(*args, **kwargs): raise AssertionError("Cached evaluation was recomputed")
    monkeypatch.setattr(W, "score_predictions", forbidden)
    monkeypatch.setattr(W, "predict_arm", forbidden)
    cached = W.evaluate(repeated_cohort, output, ["logreg"], source_outputs=str(original),
                        include_validation=False, n_resamples=8)
    assert cached[2].source.iloc[0] == "cached_evaluation"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_deployment_includes_the_notebook_recipe():
    from scripts.package import collect
    files = {p.replace("\\", "/") for p in collect()}
    assert "kineret/config/final_recipe.json" in files
    assert "kineret/experiments/workflow.py" in files
    assert "kineret/experiments/README.md" in files
    assert "notebooks/benchmark.ipynb" in files
    assert not any(p.startswith(("results/", "outputs/", "checkpoints/", "data/")) for p in files)


@pytest.mark.parametrize("qa", [False, True])
def test_encoder_checkpoint_and_named_xai(adapter_cohort, tmp_path, monkeypatch, qa):
    """Tiny untrained fixture exercises real loading, processing, inference and XAI interfaces."""
    import torch
    from kineret.config import paths
    from kineret.intervene.config import dataset_config as D, model_config as M
    D.configure(4, qa, adapter_cohort.outcome_names)
    M.configure_checkpoints(4, qa, root=str(tmp_path))
    from kineret.intervene.train import _prepare_split, trunc_scaler_of
    from kineret.intervene.dataset import EMRTokenizer
    from kineret.intervene.embedder import EMREmbedding
    from kineret.intervene.transformer import InterveneEncoder
    from kineret.io_utils import normalise_temporal, load_table
    from kineret.experiments.inference import predict_arm
    from kineret.experiments.interpretability import sparse_xai
    abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
    abstract = adapter_cohort.harmonise_events(abstract, patient_ids=adapter_cohort.patients.PatientId,
                                              kb_events=True, verbose=False)
    samples = adapter_cohort.samples("train")
    parts = _prepare_split(abstract, adapter_cohort, samples, qa, None, None, truncate=False)
    tokenizer = EMRTokenizer.from_processed_df(parts["full"])
    tokenizer.save(str(Path(M.CHECKPOINT_PATH)/"tokenizer.pt"))
    cfg = dict(M.MODEL_CONFIG, embed_dim=16, n_head=2, n_layer=1, time2vec_dim=4,
               ctx_dim=parts["full_ctx"].shape[1], dropout=0.)
    model = InterveneEncoder(cfg, EMREmbedding(tokenizer, cfg["ctx_dim"], 4, 16, 0.))
    model.attach_task_heads(hidden=16, n_heads=2)
    checkpoint = Path(M.PHASE3_CHECKPOINT)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    arm = "intervene_kb_qa" if qa else "intervene_kb"
    result = predict_arm(arm, adapter_cohort, checkpoint, k=4, device="cpu")
    assert len(result) == len(adapter_cohort.samples("test"))
    assert np.isfinite(result.filter(like="prob_").to_numpy()).all()
    other = predict_arm(arm, adapter_cohort, checkpoint, k=2, device="cpu")
    assert len(other) >= len(result)
    # Tiny transcoder smoke, not a scientific experiment: 2 fit/val/test people, one epoch.
    attribution = sparse_xai(arm, adapter_cohort, checkpoint, tmp_path/"xai", device="cpu",
                            n_fit=2, n_validation=2, n_explain=2,
                            tokens_per_admission=8, n_epochs=1, expansion=2)
    assert not attribution.empty
    assert (pd.read_csv(tmp_path/"xai/forward_invariance.csv").max_logit_difference <= 1e-6).all()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_hospital_conflicts_excluded_from_both_sides_of_loo(repeated_cohort, tmp_path):
    from kineret.loo import load_visits_master
    from kineret.experiments.cohorts import hospital_series
    visits = tmp_path / "conflicts.csv"
    rows = pd.DataFrame(dict(visit_id=range(20), hospital=["A"]*17+["B"]*3))
    rows = pd.concat([rows, pd.DataFrame(dict(visit_id=[0, 1], hospital=["B", "A"]))])
    rows.to_csv(visits, index=False)
    master = load_visits_master(visits)
    assert 0 not in set(master.VisitId)
    assert master.attrs["conflicting_visit_ids"] == [0]
    assert master.VisitId.tolist().count(1) == 1
    sites = hospital_series(repeated_cohort, visits)
    assert 0 not in sites.index
    fold = site_split(repeated_cohort, "B", visits)
    assert 0 not in set(fold.patients.PatientId)
    assert set(fold.patients.loc[fold.patients.split == "test", "PatientId"]) == {17, 18, 19}
    validate_split(fold)
