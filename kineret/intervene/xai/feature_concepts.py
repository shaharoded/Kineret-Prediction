"""
feature_concepts.py
===================

Auto-label transcoder features by the tokens / token families that activate
them most strongly across a dataset.

The pipeline:

1. Run the frozen encoder over a dataloader with the transcoder swap-in
   *passively attached* (we read features but leave the MLP output alone).
2. For every non-pad token row we keep ``(features_per_layer, concept_id,
   parent_family_id)``.
3. Per (layer, feature) we rank tokens by their activation magnitude and
   summarise the top-K -- both as the dominant concept names and as the
   dominant *parent raw concept* (token family).

This is the lightweight, dataset-level cousin of grad-times-activation: it
tells you what each feature *represents* in the corpus, regardless of any
single prediction.
"""

from typing import Dict, List

import torch
from tqdm.auto import tqdm

from kineret.intervene.xai.attribute import TranscoderHookManager


@torch.no_grad()
def collect_feature_activations(model, transcoders, dataloader, device,
                                max_batches: int = 20):
    """
    Purpose: Capture transcoder feature activations and the token identity
             behind every row.
    Method:  Attach the transcoder hooks in sniff mode (no MLP swap); on each
             batch grab the stashed ``mgr.features``, flatten over (B, T),
             and align with ``concept_ids`` + first non-pad parent raw id.

    Args:
        model        (InterveneEncoder)
        transcoders  (Dict[int, JumpReLUTranscoder])
        dataloader   (DataLoader): standard EMR batches.
        device       (torch.device)
        max_batches  (int): how many batches to scan.

    Returns:
        Dict[layer_idx -> Tensor[N, F]] : feature activations per layer.
        Tensor[N]                       : concept_id for each row.
        Tensor[N]                       : parent_family_id for each row
                                          (== first non-pad parent_raw_id).
    """
    model.eval()
    pad_idx = model.embedder.padding_idx
    mgr = TranscoderHookManager(model, transcoders, enabled=False).attach()

    per_layer: Dict[int, List[torch.Tensor]] = {i: [] for i in transcoders}
    concepts: List[torch.Tensor] = []
    parents:  List[torch.Tensor] = []

    try:
        for bi, batch in enumerate(tqdm(dataloader,
                                        total=min(max_batches, len(dataloader)),
                                        desc="collect-features", leave=False)):
            if bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            model.encode(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )
            pad_mask = (batch["position_ids"] != pad_idx)            # [B, T]
            keep = pad_mask.reshape(-1)                              # [B*T]

            # First non-pad parent id per token; parent_raw_ids: [B, T, P].
            pr = batch["parent_raw_ids"]
            first_parent = pr[..., 0]                                # [B, T]
            # If the first slot is PAD (shouldn't be for valid tokens) fall back to
            # the concept id so the row stays interpretable.
            first_parent = torch.where(first_parent != pad_idx, first_parent,
                                       batch["concept_ids"])

            concepts.append(batch["concept_ids"].reshape(-1)[keep].cpu())
            parents.append(first_parent.reshape(-1)[keep].cpu())

            for i, f in mgr.features.items():
                flat = f.reshape(-1, f.shape[-1])[keep].float().cpu()
                per_layer[i].append(flat)
    finally:
        mgr.detach()

    features = {i: torch.cat(chunks, dim=0) for i, chunks in per_layer.items()}
    return features, torch.cat(concepts, dim=0), torch.cat(parents, dim=0)


def summarise_feature(features: torch.Tensor, concept_ids: torch.Tensor,
                      parent_ids: torch.Tensor, feat_idx: int,
                      id2concept: dict, id2parent: dict,
                      top_k_tokens: int = 8, top_k_families: int = 5):
    """
    Purpose: Build a human-readable summary for one feature.
    Method:  Sort tokens by activation strength; collect (a) top-K concept
             names with their activation values and (b) the most over-
             represented parent families among the active tokens.

    Args:
        features         (Tensor [N, F])
        concept_ids      (Tensor [N])
        parent_ids       (Tensor [N])
        feat_idx         (int)
        id2concept       (dict[int -> str]): from ``tokenizer.id2token``.
        id2parent        (dict[int -> str]): inverted ``tokenizer.rawconcept2id``.
        top_k_tokens     (int)
        top_k_families   (int)

    Returns:
        dict with keys:
            'n_active'           : how many of N tokens fire this feature.
            'fire_rate'          : fraction of tokens that fire it.
            'top_tokens'         : list of (concept_name, activation) length top_k_tokens.
            'top_families'       : list of (family_name, share_of_active) length top_k_families.
            'mean_active_act'    : mean activation over firing tokens.
    """
    f = features[:, feat_idx]
    active = f > 0
    n_active = int(active.sum().item())
    if n_active == 0:
        return {"n_active": 0, "fire_rate": 0.0,
                "top_tokens": [], "top_families": [],
                "mean_active_act": 0.0}

    # ---- Top-K firing tokens by activation magnitude ----
    vals, idx = torch.topk(f, k=min(top_k_tokens, n_active))
    top_tokens = [
        (id2concept.get(int(concept_ids[i].item()), f"id={int(concept_ids[i].item())}"),
         float(v.item()))
        for v, i in zip(vals, idx)
    ]

    # ---- Family share among ALL firing tokens (not just top-K) ----
    fam_active = parent_ids[active]
    uniq, counts = torch.unique(fam_active, return_counts=True)
    share = counts.float() / n_active
    order = torch.argsort(share, descending=True)
    top_families = [
        (id2parent.get(int(uniq[j].item()), f"id={int(uniq[j].item())}"),
         float(share[j].item()))
        for j in order[:top_k_families]
    ]

    return {
        "n_active":         n_active,
        "fire_rate":        n_active / features.shape[0],
        "top_tokens":       top_tokens,
        "top_families":     top_families,
        "mean_active_act":  float(f[active].mean().item()),
    }


def feature_concept_table(features_per_layer, concept_ids, parent_ids,
                          feats_of_interest, id2concept, id2parent,
                          top_k_tokens: int = 8, top_k_families: int = 3):
    """
    Purpose: Build a tidy summary dataframe for a list of (layer, feature)
             pairs -- convenient as the input to a notebook display cell.

    Args:
        features_per_layer (Dict[int, Tensor[N, F]])
        concept_ids        (Tensor[N])
        parent_ids         (Tensor[N])
        feats_of_interest  (Iterable[(layer_idx, feat_idx)])
        id2concept, id2parent (dict[int -> str])

    Returns:
        list[dict] -- each row has: layer, feature, fire_rate, n_active,
                      top_tokens (string), top_families (string).
    """
    rows = []
    for layer_idx, feat_idx in feats_of_interest:
        s = summarise_feature(features_per_layer[layer_idx], concept_ids,
                              parent_ids, feat_idx, id2concept, id2parent,
                              top_k_tokens=top_k_tokens,
                              top_k_families=top_k_families)
        rows.append({
            "layer":       layer_idx,
            "feature":     feat_idx,
            "fire_rate":   s["fire_rate"],
            "n_active":    s["n_active"],
            "top_tokens":   ", ".join(f"{n} ({a:.2f})" for n, a in s["top_tokens"]),
            "top_families": ", ".join(f"{n} ({p*100:.0f}%)" for n, p in s["top_families"]),
        })
    return rows
