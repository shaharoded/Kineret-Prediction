"""Frozen-checkpoint adapters. These functions never call an optimizer or trainer."""
from pathlib import Path
import gc
import json
import pickle
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from kineret.config import paths
from kineret.benchmark import ARM_SPEC, run_dir_for
from kineret.io_utils import load_table, normalise_temporal
from .cohorts import evaluation_window, subset, people
from .io import load_temporal


def checkpoint_for(arm, output_root=None, checkpoint_root=None, trained_k=4):
    model, qa = ARM_SPEC[arm]
    if model.startswith("intervene"):
        root = Path(checkpoint_root or paths.CHECKPOINT_DIR)
        if arm in ("intervene_std", "intervene_sax"):
            root /= "std" if arm == "intervene_std" else "sax"
        return root / f"k{trained_k}_{'qa' if qa else 'noqa'}" / "phase3" / "ckpt_best.pt"
    return Path(run_dir_for(arm, output_root)) / ("model_bundle.pt" if model == "logreg" else "checkpoint_best.bin")


def _device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _raw(cohort):
    raw = load_temporal(paths.RAW_TEMPORAL_FILE, cohort.patients.PatientId)
    raw = cohort.harmonise_events(raw, patient_ids=cohort.patients.PatientId, verbose=False)
    origin = cohort.patients.set_index("PatientId").admission_time
    raw["hours"] = (raw.StartDateTime - raw.PatientId.map(origin)).dt.total_seconds()/3600
    return raw


def prediction_frame(cohort, samples, probs, times=None, los=None):
    block = cohort.labels_for_samples(samples).set_index("sample_id").loc[samples.sample_id]
    out = pd.DataFrame({"PatientId": samples.PatientId.to_numpy(),
                        "PersonId": people(cohort, samples.PatientId)})
    for j, outcome in enumerate(cohort.outcome_names):
        out[f"label_{outcome}"] = block[outcome].to_numpy(float)
        out[f"prob_{outcome}"] = probs[:, j]
        out[f"time_true_{outcome}"] = block[f"{outcome}__first_hour"].to_numpy(float)
        if times is not None:
            out[f"time_pred_{outcome}"] = times[:, j]
    out["los_true_hours"] = block.length_of_stay_hours.to_numpy(float)
    if los is not None:
        out["los_pred_hours"] = los
    if not np.isfinite(probs).all():
        raise ValueError("Nonfinite probabilities; do not silently drop evaluation admissions.")
    return out


def load_encoder(arm, cohort, checkpoint, k=4, device=None):
    from kineret.intervene.config import dataset_config as D
    D.configure(k, ARM_SPEC[arm][1], outcome_names=cohort.outcome_names)
    from kineret.intervene.dataset import EMRTokenizer
    from kineret.intervene.embedder import EMREmbedding
    from kineret.intervene.transformer import InterveneEncoder
    checkpoint = Path(checkpoint)
    root = checkpoint.parent.parent
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state["config"]
    tok = EMRTokenizer.load(str(root / "tokenizer.pt"))
    embedder = EMREmbedding(tok, ctx_dim=cfg["ctx_dim"], time2vec_dim=cfg["time2vec_dim"],
                           embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])
    model, *_ = InterveneEncoder.load(str(checkpoint), embedder=embedder, attach_task_heads=True)
    if not set(cohort.outcome_names).issubset(model.outcome_names):
        raise ValueError("Checkpoint outcomes differ from the frozen study heads.")
    return model.eval().to(_device(device)), tok, joblib.load(root / "scaler.pkl")


def encoder_dataset(arm, cohort, samples, tokenizer, scaler, k=4, std_table=None, std_repo=None,
                    sax_directory=None):
    from kineret.intervene.dataset import DataProcessor, EMRDataset
    from kineret.intervene.train import _build_label_frames
    qa = ARM_SPEC[arm][1]
    if arm == "intervene_sax":
        if sax_directory is None:
            raise ValueError("SAX inference requires the saved preprocessing beside its checkpoint.")
        from kineret.sax import frozen_view
        abstract, repo = frozen_view(cohort, sax_directory, k)
    elif arm == "intervene_std":
        table = Path(std_table or Path(paths.PROCESSED_DIR)/"std_temporal_data.csv")
        repo = str(std_repo or Path(paths.PROCESSED_DIR)/"tak_repo_std.json")
        if not table.exists() or not Path(repo).exists():
            raise FileNotFoundError("Frozen sigma-bin table/repository required; do not refit on evaluation data.")
        abstract = load_temporal(table, samples.PatientId)
    else:
        abstract = load_temporal(paths.ABSTRACT_FILE, samples.PatientId)
        abstract = cohort.harmonise_events(abstract, patient_ids=samples.PatientId, kb_events=True, verbose=False)
        repo = paths.TAK_REPO_PATH
    abstract = abstract[abstract.PatientId.isin(samples.PatientId)].copy()
    context = cohort.context_for_k(k, qa).reindex(samples.PatientId).reset_index()
    proc = DataProcessor(abstract, context, tak_repo_path=repo, max_input_days=k,
                         scaler=scaler, enforce_min_duration=False)
    tokens, ctx = proc.run()
    mapping = dict(zip(samples.PatientId, samples.sample_id))
    tokens.PatientId = tokens.PatientId.map(mapping)
    ctx.index = ctx.index.map(mapping)
    labels, times = _build_label_frames(cohort, samples)
    dataset = EMRDataset(tokens, ctx, tokenizer, patient_labels=labels, patient_gt_time=times)
    missing = set(samples.sample_id)-set(dataset.patient_ids)
    if missing:
        raise ValueError(f"Encoder preprocessing lost {len(missing)} eligible admissions.")
    return dataset


def encoder_predictions(arm, cohort, samples, checkpoint, k, device=None, batch_size=8, **kwargs):
    from kineret.intervene.inference import predict
    model, tokenizer, scaler = load_encoder(arm, cohort, checkpoint, k, device)
    if arm == "intervene_sax":
        kwargs["sax_directory"] = Path(checkpoint).parent.parent
    ds = encoder_dataset(arm, cohort, samples, tokenizer, scaler, k, **kwargs)
    result = predict(model, ds, batch_size=batch_size, device=_device(device)).reindex(samples.sample_id)
    outcomes = cohort.outcome_names
    probs = result[[f"P_{o}" for o in outcomes]].to_numpy(float)
    times = result[[f"T_{o}" for o in outcomes]].to_numpy(float)
    los = result["T_RELEASE_EVENT"].to_numpy(float) if "T_RELEASE_EVENT" in result else None
    return prediction_frame(cohort, samples, probs, times, los)


def logreg_predictions(arm, cohort, samples, checkpoint, device=None, **_):
    from kineret.logreg.train import MultiLabelLogisticRegression
    from kineret.logreg.features import build_feature_frame, assemble_design_matrix
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if bundle["outcomes"] != cohort.outcome_names:
        raise ValueError("LogReg outcome layout mismatch.")
    features, _ = build_feature_frame(_raw(cohort), samples, cohort.leakage_blocklist(), bundle["keep_vars"])
    context = cohort.context_for_samples(samples, ARM_SPEC[arm][1])
    X, _ = assemble_design_matrix(features, context, bundle["medians"])
    X = X.reindex(index=samples.sample_id, columns=bundle["columns"], fill_value=0.)
    X = ((X.to_numpy()-bundle["mean"])/bundle["std"]).astype(np.float32)
    model = MultiLabelLogisticRegression(X.shape[1], len(cohort.outcome_names)).to(_device(device))
    model.load_state_dict(bundle["state"]); model.eval()
    with torch.no_grad():
        risk, time, los = model(torch.as_tensor(X, device=_device(device)))
    return prediction_frame(cohort, samples, risk.sigmoid().cpu().numpy(),
        time.cpu().numpy()*bundle["time_std"]+bundle["time_mean"],
        los.cpu().numpy()*bundle["los_std"]+bundle["los_mean"])


def recover_strats_preprocessor(arm, source_root, frozen_pickle, destination, device="cpu"):
    """Replay original seeded preprocessing, never training. Requires ORIGINAL pickle."""
    from kineret.strats.train import _build_args
    from kineret.strats.dataset import Dataset
    from kineret.strats.model_utils import set_all_seeds
    meta = json.loads((Path(run_dir_for(arm, source_root))/"run_meta.json").read_text())
    args = _build_args(ARM_SPEC[arm][1], str(destination), device, meta["seed"],
                       {k: v for k, v in meta["hyperparameters"].items()
                        if k in __import__("kineret.strats.config", fromlist=["STRATS_SETTINGS"]).STRATS_SETTINGS})
    args.dataset_path = str(frozen_pickle)
    args.logger = SimpleNamespace(write=lambda *a: None)
    set_all_seeds(meta["seed"])
    ds = Dataset(args)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ds.inference_schema, destination)
    return str(destination)


def strats_predictions(arm, cohort, samples, checkpoint, schema_path=None, device=None, **_):
    from kineret.strats.dataset import Dataset
    from kineret.strats.strats import Strats
    from kineret.strats.preprocess import _build_input_table
    schema_path = Path(schema_path or Path(checkpoint).parent/"inference_schema.pkl")
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing {schema_path}; recover from the ORIGINAL processed pickle via the notebook.")
    schema = joblib.load(schema_path)
    args = SimpleNamespace(**schema["args"])
    args.device = _device(device)
    if args.outcome_names != cohort.outcome_names:
        raise ValueError("STraTS outcome layout mismatch.")
    raw = _raw(cohort); raw["minute"] = raw.hours*60
    ctx = cohort.context_for_samples(samples, ARM_SPEC[arm][1]).reindex(columns=schema["static_varis"])
    static = ctx.rename_axis("ts_id").reset_index().melt(id_vars="ts_id", var_name="variable", value_name="value")
    static["minute"] = 0.
    data, _, _ = _build_input_table(raw, samples, static, cohort.leakage_blocklist(), 0.)
    data = data[~data.variable.isin(schema["static_varis"]) & data.variable.isin(schema["var_to_ind"])].copy()
    # Fixed seed, frozen cap and scale; inference never estimates normalization.
    data = data.sample(frac=1, random_state=args.seed).groupby("ts_id", sort=False).head(args.max_obs)
    data = data.merge(schema["means_stds"].reset_index(), on="variable", how="left")
    data["value"] = (data.value-data["mean"])/data["std"]
    data["minute"] = data.minute/schema["max_minute"]*2-1
    # At the original K, preserve the exact stochastic observation cap used by
    # the fitted model. This also permits reliable legacy preprocessing recovery.
    selected = schema.get("selected_eval_observations")
    if selected is not None:
        known = set(selected.ts_id)
        available = set(samples.sample_id) & known
        data = pd.concat([data[~data.ts_id.isin(available)],
                          selected[selected.ts_id.isin(available)]], ignore_index=True)
    ds = Dataset.__new__(Dataset); ds.args = args
    ds.N = len(samples)
    ds.demo = (ctx.reindex(samples.sample_id).fillna(0).to_numpy()-schema["static_mean"])/schema["static_std"]
    ds.values, ds.times, ds.varis = [[[] for _ in range(ds.N)] for _ in range(3)]
    index = dict(zip(samples.sample_id, range(ds.N)))
    for r in data.itertuples():
        i = index[r.ts_id]
        ds.values[i].append(r.value); ds.times[i].append(r.minute); ds.varis[i].append(schema["var_to_ind"][r.variable])
    # The legacy batch interface expects targets, although eval forward ignores them.
    ds.y = np.zeros((ds.N, len(cohort.outcome_names)), np.float32)
    ds.los_target_norm = ds.los_mask = np.zeros(ds.N, np.float32)
    ds.time_target_norm = ds.time_mask = np.zeros_like(ds.y)
    model = Strats(args).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True)); model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, ds.N, args.eval_batch_size):
            batch = ds.get_batch_strats(list(range(start, min(ds.N, start+args.eval_batch_size))))
            batch = {k: v.to(args.device) for k, v in batch.items()
                     if k in ("values", "times", "varis", "obs_mask", "demo")}
            results.append(model(**batch).cpu().numpy())
    result = np.concatenate(results)
    n = len(cohort.outcome_names)
    probs = result[:, :n]
    means = np.array([args.time_mean[o] for o in cohort.outcome_names])
    stds = np.array([args.time_std[o] for o in cohort.outcome_names])
    return prediction_frame(cohort, samples, probs, result[:, n:2*n]*stds+means,
                            result[:, 2*n]*args.los_std+args.los_mean)


def predict_arm(arm, cohort, checkpoint, k=4, split="test", **kwargs):
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"No frozen checkpoint for {arm}: {checkpoint}. No automatic retraining.")
    with evaluation_window(k):
        samples = cohort.samples(split)
        if samples.empty:
            raise ValueError(f"No eligible {split} admissions at K={k}.")
        small = subset(cohort, samples.PatientId)
        model = ARM_SPEC[arm][0]
        if model.startswith("intervene"):
            result = encoder_predictions(arm, small, samples, checkpoint, k, **kwargs)
        elif model == "strats":
            result = strats_predictions(arm, small, samples, checkpoint, **kwargs)
        else:
            result = logreg_predictions(arm, small, samples, checkpoint, **kwargs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
