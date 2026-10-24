"""SAX scientific contracts: global scaling, missingness, prefix safety and weight reuse."""
from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest

from kineret.sax import RECIPE, SYMBOLS, BREAKPOINTS, fit_statistics, transform, prefix_view, raw_view


def frame(patient, hours, values, concept="GLUCOSE_MEASURE"):
    origin = pd.Timestamp("2024-01-01")
    dates = [origin+pd.Timedelta(hours=h) for h in hours]
    return pd.DataFrame(dict(PatientId=patient, ConceptName=concept, StartDateTime=dates,
                             EndDateTime=dates, Value=values, hours=hours))


@pytest.fixture
def toy_cohort():
    from types import SimpleNamespace
    return SimpleNamespace(patients=pd.DataFrame(dict(PatientId=[1, 2],
        admission_time=pd.Timestamp("2024-01-01"), split=["train", "test"])),
        outcome_names=["COMPLICATION_EVENT"], kb_event_names=["OTHER_EVENT"], replaced_spellings=[],
        outcome_aliases={"COMPLICATION_EVENT": ["COMPLICATION_EVENT"]},
        structural_aliases={"ADMISSION": ["ADMISSION"]})


def test_recipe_is_fixed_four_hours_seven_gaussian_symbols():
    assert RECIPE["segment_hours"] == 4 and len(SYMBOLS) == 7
    assert len(BREAKPOINTS) == 6 and BREAKPOINTS == sorted(BREAKPOINTS)
    np.testing.assert_allclose(BREAKPOINTS, -np.asarray(BREAKPOINTS)[::-1])


def test_scaling_fits_training_only_and_preserves_cross_patient_levels(toy_cohort):
    train = frame(1, [1, 2, 3], [80, 100, 120])
    test = frame(2, [1, 2], [10000, 20000])
    stats = fit_statistics(pd.concat([train, test]), [1])
    assert stats.loc["GLUCOSE_MEASURE", "mean"] == 100
    assert stats.loc["GLUCOSE_MEASURE", "std"] == 20
    # Identical absolute values map identically even in different admissions.
    same = pd.concat([frame(1, [1], [140]), frame(2, [1], [140])])
    result = transform(same, toy_cohort, stats)
    assert result.Value.nunique() == 1 and result.Value.iloc[0] == "S7"


def test_missing_segments_are_not_filled_or_merged_across_gaps(toy_cohort):
    raw = frame(1, [1, 9], [100, 100])
    stats = fit_statistics(raw, [1])
    table = prefix_view(transform(raw, toy_cohort, stats))
    assert len(table) == 2
    assert list(table.SAXAvailableHour) == [4, 12]
    assert table.iloc[0].EndDateTime < table.iloc[1].StartDateTime


def test_identical_contiguous_symbols_merge_after_prefix_selection(toy_cohort):
    raw = frame(1, [1, 5], [100, 100])
    table = transform(raw, toy_cohort, fit_statistics(raw, [1]))
    result = prefix_view(table)
    assert len(result) == 1
    assert result.iloc[0].SAXAvailableHour == 8
    assert result.iloc[0].EndDateTime-result.iloc[0].StartDateTime == pd.Timedelta(hours=8)


def test_future_segment_starting_at_cutoff_never_leaks(toy_cohort):
    stats = pd.DataFrame(dict(mean=[100.], std=[20.]), index=["GLUCOSE_MEASURE"])
    raw = frame(2, [93, 96, 97, 99], [100, 100, 100, 100])
    changed = raw.copy(); changed.loc[changed.hours > 96, "Value"] = 100000
    first = prefix_view(transform(raw, toy_cohort, stats), k=4)
    second = prefix_view(transform(changed, toy_cohort, stats), k=4)
    pd.testing.assert_frame_equal(first, second)
    assert first.SAXAvailableHour.max() == 96
    assert first.EndDateTime.max() == pd.Timestamp("2024-01-05")


def test_boundary_measurements_belong_to_segment_ending_there(toy_cohort):
    raw = frame(1, [0, 4, 4.1], [100, 100, 200])
    stats = pd.DataFrame(dict(mean=[100.], std=[20.]), index=["GLUCOSE_MEASURE"])
    table = transform(raw, toy_cohort, stats)
    assert list(table.SAXAvailableHour) == [4, 8]
    assert table.iloc[0].Value == "S4"


def test_raw_view_withholds_qa_and_kb_events(toy_cohort):
    raw = pd.concat([frame(1, [0], [1], "ADMISSION"),
                     frame(1, [1], [120]), frame(1, [2], [1], "COMPLICATION_EVENT"),
                     frame(1, [2], [.8], "MONITOR_PATTERN"), frame(1, [2], [1], "OTHER_EVENT")])
    result = raw_view(toy_cohort, raw)
    assert set(result.ConceptName) == {"ADMISSION", "GLUCOSE_MEASURE"}
    assert result.loc[result.ConceptName == "ADMISSION", "Value"].iloc[0] == "True"


def test_notebook_model_discovery_reuses_completed_sax_without_training(tmp_path, monkeypatch):
    from kineret.experiments import workflow as W
    from kineret.experiments.inference import checkpoint_for
    directory = tmp_path/"new_primary_models"
    root = directory/"intervene_sax"
    checkpoint = checkpoint_for("intervene_sax", checkpoint_root=str(root/"checkpoints"))
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"completed checkpoint")
    def forbidden(*args, **kwargs): raise AssertionError("Completed primary model was trained again")
    monkeypatch.setattr(W, "run_holdout", forbidden)
    sources = W.train_missing_main(None, directory, ["intervene_sax"],
        source_outputs=str(tmp_path/"original_outputs"), checkpoint_root=str(tmp_path/"original_checkpoints"))
    assert "intervene_sax" in sources
    assert checkpoint_for("intervene_sax", checkpoint_root=sources["intervene_sax"]["checkpoints"]) == checkpoint


def test_sax_training_and_frozen_multi_k_reuse(cohort, tmp_path, monkeypatch):
    """Tiny synthetic fit exercises the real three-phase path and standalone SAX artifacts."""
    import torch
    from kineret.benchmark import run_one, ARM_ORDER, ARM_SPEC
    from kineret.experiments.inference import predict_arm, checkpoint_for
    c = deepcopy(cohort)
    c.person_of_patient = {i: i for i in c.patients.PatientId}
    assert ARM_ORDER[-1] == "intervene_sax" and ARM_SPEC["intervene_sax"][1] is False
    result = run_one("intervene_sax", cohort=c, output_root=str(tmp_path/"outputs"),
        checkpoint_root=str(tmp_path/"checkpoints"), device="cpu", bootstrap_resamples=0,
        overrides=dict(model=dict(embed_dim=16, n_head=2, n_layer=1, time2vec_dim=4),
                       training=dict(phase1_n_epochs=1, phase2_n_epochs=1, phase3_n_epochs=1,
                                     batch_size=128, **{"early-stop-patience": 1})))
    checkpoint = checkpoint_for("intervene_sax", checkpoint_root=str(tmp_path/"checkpoints"))
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def forbidden(*a, **kw): raise AssertionError("SAX frozen inference called training")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    predicted = predict_arm("intervene_sax", c, checkpoint, k=4, device="cpu")
    original = pd.read_csv(Path(result["output_dir"])/"test_predictions.csv").set_index("PatientId")
    cols = predicted.filter(like="prob_").columns
    np.testing.assert_allclose(predicted.set_index("PatientId")[cols], original.loc[predicted.PatientId, cols], atol=1e-5)
    other = predict_arm("intervene_sax", c, checkpoint, k=2, device="cpu")
    assert len(other) >= len(predicted)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_missing_kb_measurement_gets_plain_sax_root(tmp_path, monkeypatch):
    import json
    from kineret.sax import build_repository
    from kineret.config import paths
    base = tmp_path / "base.json"
    original = {"taks": {"KNOWN": {"name": "KNOWN", "derived_from": None}}}
    base.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(paths, "TAK_REPO_PATH", str(base))
    concept = "ALBUMIN_SERUM_MEASURE"
    raw = frame(1, [1, 2], [3., 4.], concept)
    stats = fit_statistics(raw, [1])
    repo_path, known = build_repository(stats, raw, tmp_path)
    taks = json.loads(Path(repo_path).read_text())["taks"]
    assert taks[concept]["derived_from"] is None
    assert taks[concept]["family"] == "raw-concept"
    for symbol in SYMBOLS:
        assert taks[f"{concept}_SAX_{symbol}"]["derived_from"] == concept
        assert f"{concept}_SAX_{symbol}" in known
    assert json.loads(base.read_text()) == original
    assert pd.read_csv(tmp_path / "sax_added_raw_roots.csv").ConceptName.tolist() == [concept]
