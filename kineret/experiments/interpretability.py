"""Trained embedding geometry and exploratory named-concept attribution."""
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .cohorts import evaluation_window, subset, people, fingerprint
from .inference import load_encoder, encoder_dataset
from .workflow import write_json, _stamp
from kineret.config import paths


def embedding_geometry(model, directory, seed=19):
    """One timestamp, eval mode, original-space distances plus a descriptive UMAP."""
    import matplotlib.pyplot as plt
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("Install the experiments extra: pip install -e '.[experiments]'") from exc
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    tok = model.embedder.tokenizer
    device = next(model.parameters()).device
    rows = []
    for token, tid in sorted(tok.token2id.items(), key=lambda item: item[1]):
        if not token.endswith(("_START", "_END")) or token.startswith("["):
            continue
        value, boundary = token.rsplit("_", 1)
        candidates = [c for c in tok.concept2id if value == c or value.startswith(c+"_")]
        if not candidates or value not in tok.value2id:
            continue
        concept = max(candidates, key=len)
        parents = tok.tokenid2parent_raw_ids[tid].detach().cpu().tolist()
        parents = [p for p in parents if p != tok.pad_token_id]
        raw_names = {v: k for k, v in tok.rawconcept2id.items()}
        rows.append(dict(token=token, tid=tid, value=value, boundary=boundary, concept=concept,
                         parents=parents, raw="|".join(sorted(raw_names[p] for p in parents))))
    if len(rows) < 4:
        raise ValueError("Too few named boundary tokens for embedding geometry.")
    width = max(len(r["parents"]) for r in rows)
    parents = [r["parents"]+[tok.pad_token_id]*(width-len(r["parents"])) for r in rows]
    def ids(values):
        return torch.tensor([values], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        vectors, _ = model.embedder(
            ids(parents), ids([tok.concept2id[r["concept"]] for r in rows]),
            ids([tok.value2id[r["value"]] for r in rows]), ids([r["tid"] for r in rows]),
            torch.zeros((1, len(rows)), device=device),
            torch.zeros((1, model.cfg["ctx_dim"]), device=device))
    vectors = vectors[0].cpu().numpy()
    projection = UMAP(n_neighbors=min(15, len(rows)-1), min_dist=.15, metric="cosine", random_state=seed).fit_transform(vectors)
    frame = pd.DataFrame(rows).drop(columns="parents")
    frame["x"], frame["y"] = projection.T
    frame.to_csv(directory/"embedding_umap.csv", index=False)
    np.save(directory/"trained_embeddings.npy", vectors)
    unit = vectors/np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    distances = {k: [] for k in ("matching_boundaries", "same_concept", "same_raw", "different_raw")}
    # Bounded pair sample for large vocabularies; fixed sampling seed is not a training sweep.
    rng = np.random.default_rng(seed)
    for i in range(len(rows)):
        partners = np.arange(i+1, len(rows))
        if len(partners) > 500:
            partners = rng.choice(partners, 500, replace=False)
        for j in partners:
            a, b = rows[i], rows[j]
            group = ("matching_boundaries" if a["value"] == b["value"] else
                     "same_concept" if a["concept"] == b["concept"] else
                     "same_raw" if a["raw"] == b["raw"] else "different_raw")
            distances[group].append(float(1-unit[i] @ unit[j]))
    pd.DataFrame([dict(relation=k, n_pairs=len(v), median=np.median(v) if v else np.nan)
                  for k, v in distances.items()]).to_csv(directory/"cosine_distances.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, pair in frame.groupby("value"):
        if set(pair.boundary) == {"START", "END"}:
            ax.plot(pair.x, pair.y, color="gray", alpha=.15, lw=.5, zorder=0)
    group = np.select([frame.concept.str.contains("GLUCOSE"), frame.concept.str.contains("CREATININE"),
                       frame.concept.str.contains("PATTERN")], ["glucose", "creatinine", "QA patterns"], default="other")
    for label in np.unique(group):
        for boundary, marker in (("START", "o"), ("END", "^")):
            m = (group == label) & (frame.boundary == boundary)
            ax.scatter(frame.loc[m, "x"], frame.loc[m, "y"], s=15, marker=marker,
                       alpha=.35 if label == "other" else .85, label=f"{label} {boundary}")
    labeled = frame[(frame.boundary == "START") &
                    frame.concept.str.contains("GLUCOSE|CREATININE")].head(24)
    for row in labeled.itertuples():
        ax.annotate(row.value, (row.x, row.y), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.legend(fontsize=7); ax.set_title("Trained input embeddings · timestamp held fixed")
    fig.tight_layout(); fig.savefig(directory/"embedding_umap.png", dpi=180)
    fig.savefig(directory/"embedding_umap.pdf"); plt.close(fig)
    return frame


def _sample(cohort, split, n, seed):
    samples = cohort.samples(split).copy()
    samples["person"] = people(cohort, samples.PatientId)
    # One admission per person prevents repeated admissions dominating the exploratory panel.
    return samples.sample(frac=1, random_state=seed).drop_duplicates("person").head(n)


def _batch(model, batch):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def _predict(model, batch):
    return model.predict(**{k: batch[k] for k in ("parent_raw_ids", "concept_ids", "value_ids",
                                                 "position_ids", "abs_ts", "context_vec")})


def _capture_bounded(model, loader, layer, tokens_per_admission=256):
    """Capture one layer at a time; cap storage per admission, including long sequences."""
    xs, ys = [], []
    for batch in loader:
        batch = _batch(model, batch)
        valid = batch["position_ids"][0] != model.embedder.padding_idx
        stash = {}
        def hook(module, inputs, output):
            positions = torch.where(valid)[0]
            if len(positions) > tokens_per_admission:
                positions = positions[torch.linspace(0, len(positions)-1, tokens_per_admission, device=positions.device).long()]
            stash["x"] = inputs[0][0, positions].detach().cpu()
            stash["y"] = output[0, positions].detach().cpu()
        handle = model.blocks[layer].mlp.register_forward_hook(hook)
        try:
            with torch.no_grad():
                _predict(model, batch)
        finally:
            handle.remove()
        xs.append(stash["x"]); ys.append(stash["y"])
    return torch.cat(xs), torch.cat(ys)


def sparse_xai(arm, cohort, checkpoint, directory, device=None, n_fit=300, n_validation=64,
               n_explain=300, tokens_per_admission=256, n_epochs=20, expansion=16, seed=2023):
    """Fit on training people, validate reconstruction on validation people, explain test people.

    Scores are local gradient×activation sensitivities through sparse features.
    Named tokens provide the clinical vocabulary, not a causal explanation.
    """
    from kineret.intervene.dataset import collate_emr
    from kineret.intervene.xai.transcoder.train import train_transcoders, save_transcoders, load_transcoders
    from kineret.intervene.xai.attribute import feature_to_logit_attribution, TranscoderHookManager
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    design = dict(cohort=fingerprint(cohort), checkpoint=_stamp(checkpoint), arm=arm, k=4,
                  n_fit=n_fit, n_validation=n_validation, n_explain=n_explain,
                  tokens_per_admission=tokens_per_admission, n_epochs=n_epochs, expansion=expansion, seed=seed,
                  inputs={key: _stamp(getattr(paths, key))
                          for key in ("ABSTRACT_FILE", "QA_FILE", "TAK_REPO_PATH")})
    manifest = directory/"design.json"
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError("XAI design changed: use a new output directory to avoid stale transcoders.")
    if (directory/"completed.json").exists():
        return pd.read_csv(directory/"named_concept_attribution.csv")
    write_json(manifest, design)
    model, tok, scaler = load_encoder(arm, cohort, checkpoint, 4, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.use_checkpoint = False
    torch.manual_seed(seed)
    with evaluation_window(4):
        def dataset(split, n):
            samples = _sample(cohort, split, n, seed)
            samples.to_csv(directory/f"{split}_sample.csv", index=False)
            small = subset(cohort, samples.PatientId)
            return encoder_dataset(arm, small, samples, tok, scaler, 4)
        bundle = directory/"transcoders.pt"
        if bundle.exists():
            transcoders = load_transcoders(bundle)
        else:
            fit_ds = dataset("train", n_fit)
            loader = DataLoader(fit_ds, batch_size=1, collate_fn=collate_emr, shuffle=False)
            transcoders = {}
            for layer in range(len(model.blocks)):
                layer_path = directory/f"transcoder_{layer}.pt"
                if layer_path.exists():
                    transcoders.update(load_transcoders(layer_path)); continue
                x, y = _capture_bounded(model, loader, layer, tokens_per_admission)
                fitted, history = train_transcoders({layer: (x, y)}, d_model=model.cfg["embed_dim"],
                    n_features=expansion*model.cfg["embed_dim"], n_epochs=n_epochs,
                    device=str(next(model.parameters()).device))
                save_transcoders(fitted, layer_path); transcoders.update(fitted)
                write_json(directory/f"training_layer_{layer}.json", history)
                del x, y; gc.collect()
            save_transcoders(transcoders, bundle)
            del loader, fit_ds; gc.collect()
        validation_ds = dataset("val", n_validation)
        loader = DataLoader(validation_ds, batch_size=1, collate_fn=collate_emr)
        diagnostics = []
        for layer, tc in transcoders.items():
            x, y = _capture_bounded(model, loader, layer, tokens_per_admission)
            tc = tc.cpu().eval()
            sse, active, count = 0., 0., 0
            with torch.no_grad():
                for start in range(0, len(x), 1024):
                    features = tc.encode(x[start:start+1024])
                    recon = tc.decode(features)
                    sse += float(((recon-y[start:start+1024])**2).sum())
                    active += float((features != 0).sum()); count += features.numel()
            denom = float(((y-y.mean(dim=0))**2).sum())
            diagnostics.append(dict(layer=layer, r2=1-sse/denom if denom else np.nan,
                                     active_fraction=active/count, validation_tokens=len(x)))
            del x, y
        pd.DataFrame(diagnostics).to_csv(directory/"reconstruction_validation.csv", index=False)
        del loader, validation_ds; gc.collect()
        explain_ds = dataset("test", n_explain)
        loader = DataLoader(explain_ds, batch_size=1, collate_fn=collate_emr)
        risk_names = [model.outcome_names[i] for i in model.task_heads.risk_idx.cpu().tolist()]
        rows, checks = [], []
        for i, batch in enumerate(loader):
            batch = _batch(model, batch)
            with torch.no_grad():
                original = _predict(model, batch)[0]
                with TranscoderHookManager(model, transcoders).attach() as manager:
                    hooked = _predict(model, batch)[0]
                difference = float((original-hooked).abs().max())
            if difference > 1e-6:
                raise ValueError("XAI hooks changed deployed risk logits.")
            checks.append(dict(sample_id=explain_ds.patient_ids[i], max_logit_difference=difference))
            for j, outcome in enumerate(risk_names):
                contributions, mask = feature_to_logit_attribution(model, transcoders, batch, j)
                scores = sum(v.sum(dim=-1) for v in contributions.values()).cpu().numpy()
                ids = batch["position_ids"][0].cpu().numpy()
                per_concept = {}
                for token_id, score, valid in zip(ids, scores, mask.cpu().numpy()):
                    token = tok.id2token[int(token_id)]
                    if not valid or token.startswith("["):
                        continue
                    name = token.rsplit("_", 1)[0] if token.endswith(("_START", "_END")) else token
                    per_concept[name] = per_concept.get(name, 0.) + float(score)
                for name, score in per_concept.items():
                    rows.append(dict(sample_id=explain_ds.patient_ids[i], outcome=outcome,
                                     named_concept=name, contribution=score))
        details = pd.DataFrame(rows)
        details.to_csv(directory/"attribution_by_admission.csv", index=False)
        pd.DataFrame(checks).to_csv(directory/"forward_invariance.csv", index=False)
        rng = np.random.default_rng(seed)
        summaries = []
        for outcome, group in details.groupby("outcome"):
            matrix = group.pivot(index="sample_id", columns="named_concept", values="contribution").reindex(explain_ds.patient_ids).fillna(0.)
            means = matrix.mean()
            top = means.abs().nlargest(30).index
            values = matrix[top].to_numpy()
            draws = np.array([values[rng.integers(len(values), size=len(values))].mean(axis=0) for _ in range(2000)])
            for col in means.index:
                ci = np.quantile(draws[:, list(top).index(col)], [.025, .975]) if col in top else [np.nan, np.nan]
                summaries.append(dict(outcome=outcome, named_concept=col, mean_contribution=means[col],
                    lo=ci[0], hi=ci[1], n_people=len(matrix), exploratory_selected_ci=col in top))
        summary = pd.DataFrame(summaries)
        summary.to_csv(directory/"named_concept_attribution.csv", index=False)
        review = summary.loc[summary.exploratory_selected_ci].copy()
        review["clinical_coherence_rating"] = ""
        review["clinical_rationale_or_disagreement"] = ""
        review.to_csv(directory/"clinical_review_template.csv", index=False)
        _plot_attributions(summary, directory)
        write_json(directory/"completed.json", design)
    return summary


def _plot_attributions(summary, directory):
    import matplotlib.pyplot as plt
    outcomes = list(summary.outcome.unique())
    fig, axes = plt.subplots((len(outcomes)+1)//2, 2, figsize=(13, 3.8*((len(outcomes)+1)//2)), squeeze=False)
    for ax, outcome in zip(axes.flat, outcomes):
        group = summary[summary.outcome == outcome]
        group = group.loc[group.mean_contribution.abs().nlargest(12).index].sort_values("mean_contribution")
        ax.barh(group.named_concept, group.mean_contribution,
                color=np.where(group.mean_contribution >= 0, "#b44", "#4682b4"))
        ax.set_title(outcome, fontsize=9); ax.tick_params(axis="y", labelsize=6)
        ax.axvline(0, color="black", lw=.5)
    for ax in list(axes.flat)[len(outcomes):]: ax.axis("off")
    fig.suptitle("Named-concept local sensitivity · sparse-transcoder attribution")
    fig.tight_layout(); fig.savefig(Path(directory)/"named_concept_attribution.png", dpi=180)
    fig.savefig(Path(directory)/"named_concept_attribution.pdf"); plt.close(fig)
