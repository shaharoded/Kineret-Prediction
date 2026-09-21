"""
INTERVenE-Enc, adapted for the Kineret benchmark.

Ported from the `intervene_enc` package with three substantive changes, all
documented in the repository README:

* Phase-3 supervises against cohort-supplied labels rather than outcomes
  re-read from the token stream, so INTERVenE, ss-STraTS and LogReg are fit
  against byte-identical targets;
* the Phase-3 label window is closed at the top, matching `(K*24, N*24]`;
* the config is re-pointable per sweep cell, and per abstraction arm.

Import `kineret.intervene.train` for the driver; the rest of the surface is
unchanged from upstream.
"""

from kineret.intervene.dataset import (
    DataProcessor, EMRDataset, EMRTokenizer, collate_emr, get_dataloader,
)
from kineret.intervene.embedder import EMREmbedding, train_embedder
from kineret.intervene.inference import predict
from kineret.intervene.transformer import (
    InterveneEncoder, PerOutcomeAttnPool, TaskHeads,
    finetune_transformer, pretrain_transformer,
)

__all__ = [
    "EMRDataset",
    "DataProcessor",
    "EMRTokenizer",
    "collate_emr",
    "get_dataloader",
    "EMREmbedding",
    "train_embedder",
    "InterveneEncoder",
    "TaskHeads",
    "PerOutcomeAttnPool",
    "pretrain_transformer",
    "finetune_transformer",
    "predict",
]
