"""
transcoder
==========

External (non-invasive) explainability layer for InterveneEncoder.

A *transcoder* is a sparse cross-layer dictionary learner that approximates the
input -> output mapping of an MLP through a sparse latent. Unlike a sparse
autoencoder fitted to residual activations, the transcoder side-steps the
SwiGLU nonlinearity entirely (it learns the I/O map, not an internal
decomposition), which makes it the natural choice for gated MLPs like the one
in `intervene_enc.transformer.MLP`.

Public API
----------
* :class:`JumpReLUTranscoder`  -- the sparse bottleneck module.
* :func:`capture_mlp_io`        -- hook-based activation collector.
* :func:`train_transcoders`     -- fit one transcoder per encoder layer.
* :func:`load_transcoders`      -- restore a trained bundle.
"""

from kineret.intervene.xai.transcoder.transcoder import JumpReLUTranscoder
from kineret.intervene.xai.transcoder.hooks import capture_mlp_io, MLPIOBuffer
from kineret.intervene.xai.transcoder.train import train_transcoders, load_transcoders, save_transcoders

__all__ = [
    "JumpReLUTranscoder",
    "capture_mlp_io",
    "MLPIOBuffer",
    "train_transcoders",
    "load_transcoders",
    "save_transcoders",
]
