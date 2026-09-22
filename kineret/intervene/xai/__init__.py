"""
xai
===

Explainability layer for InterveneEncoder, built on top of the
:mod:`transcoder` package.

Public API
----------
* :class:`TranscoderHookManager` -- attach/detach swap-in hooks that replace
  each ``MLP`` output with its transcoder reconstruction.
* :func:`feature_activations`    -- collect per-token sparse feature firings
  alongside the encoder's normal outputs.
* :func:`feature_to_logit_attribution` -- decompose risk logits per outcome
  into contributions of individual transcoder features.
"""

from kineret.intervene.xai.attribute import (
    TranscoderHookManager,
    feature_activations,
    feature_to_logit_attribution,
)
from kineret.intervene.xai.feature_concepts import (
    collect_feature_activations,
    summarise_feature,
    feature_concept_table,
)

__all__ = [
    "TranscoderHookManager",
    "feature_activations",
    "feature_to_logit_attribution",
    "collect_feature_activations",
    "summarise_feature",
    "feature_concept_table",
]
