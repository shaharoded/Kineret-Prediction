"""
attribute.py
============

Explainability via transcoder features, with the forward pass kept
**bit-exact to the deployed model**.

Why no swap-in
--------------
An earlier version of this file supported replacing each block's MLP output
with its transcoder reconstruction (the "swap").  In our setting the head
is sharp enough that even a per-layer R² ≈ 0.97 reconstruction decorrelates
the risk logits under a single-block swap (Pearson ρ ≈ 0.02), so any
attribution computed against a swapped model would be explaining a model
that doesn't behave like the deployed one.

Instead we use a **gradient-decoupled** hook:

    y_used = output + (y_hat - y_hat.detach())

* Forward value equals ``output`` exactly (because ``y_hat - y_hat.detach()``
  is numerically zero), so the model is bit-exact equal to the deployed model.
* Gradient w.r.t. the transcoder features flows through ``y_hat``, so
  ``∂risk / ∂f`` exists and equals ``∂risk / ∂y · ∂y_hat / ∂f`` -- the local
  sensitivity of the *real* risk logit to each transcoder feature.

Attribution is then the standard "gradient × activation" decomposition:

    contrib_t^{(k)} = sum_L  f_L(p, t) · (∂ risk_k(p) / ∂ f_L(p, t))

which is a Taylor expansion of the *deployed model's* risk logit into
contributions from each (layer, token, feature) triple.  Nothing about this
claim depends on the transcoder's reconstruction quality at the swap site;
the reconstruction R² (§4 of the demo notebook) is used only to argue that
the features form a meaningful basis for the MLP output.
"""

from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn as nn


class TranscoderHookManager:
    """
    Manage forward hooks that expose per-layer transcoder features for
    attribution **without altering the model's forward output**.

    Parameters
    ----------
    model
        InterveneEncoder (frozen).
    transcoders
        Dict[layer_idx -> JumpReLUTranscoder].
    gradient_decoupled
        When True (default) and ``enabled=True``, the forward output is left
        bit-exact equal to the real MLP output but gradient flows through the
        transcoder features so attribution is well-defined.  When False, the
        manager behaves as a pure sniffer (no gradient path through the
        features; ``model`` runs exactly as if the manager weren't attached).
    enabled
        Master switch.  When False, the hook records features but otherwise
        does nothing -- equivalent to ``gradient_decoupled=False``.

    Notes
    -----
    The hook *records*:
      * ``features[L]``     : transcoder features f_L (leaf when
                              ``attach(track_grad=True)``).
      * ``recon[L]``        : transcoder's MLP-output reconstruction y_hat_L.
      * ``true_output[L]``  : the real MLP output at that site (detached).
    """

    def __init__(self, model, transcoders: Dict[int, nn.Module],
                 gradient_decoupled: bool = True,
                 enabled: bool = True):
        self.model = model
        self.transcoders = transcoders
        self.gradient_decoupled = gradient_decoupled
        self.enabled = enabled
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        # Per-layer storage filled on every forward.  Tensors live on the
        # encoder's device so attribution can run autograd through them.
        self.features:    Dict[int, torch.Tensor] = {}
        self.recon:       Dict[int, torch.Tensor] = {}
        self.true_output: Dict[int, torch.Tensor] = {}
        self._track_grad = False

    # ------------------------------------------------------------ context ## #
    def attach(self, track_grad: bool = False):
        """
        Attach hooks.  When ``track_grad=True`` the stashed feature tensors
        are detached and re-marked as autograd leaves so
        :func:`feature_to_logit_attribution` can backprop into them.
        """
        self._track_grad = track_grad
        for i, tc in self.transcoders.items():
            tc.to(next(self.model.parameters()).device).eval()
            h = self.model.blocks[i].mlp.register_forward_hook(self._make_hook(i))
            self._handles.append(h)
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear(); self.recon.clear(); self.true_output.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    # ---------------------------------------------------------- internals ## #
    def _make_hook(self, layer_idx: int):
        tc = self.transcoders[layer_idx]

        def _hook(_module, inputs, output):
            x_in = inputs[0]                                     # [B, T, D]
            f = tc.encode(x_in)                                  # [B, T, F]
            if self._track_grad:
                # Detach upstream so autograd treats f as a leaf, then re-decode
                # so the gradient w.r.t. f reflects only the path through this
                # transcoder.
                f = f.detach().requires_grad_(True)
            y_hat = tc.decode(f)                                 # [B, T, D]
            self.features[layer_idx]    = f
            self.recon[layer_idx]       = y_hat
            self.true_output[layer_idx] = output.detach()
            if self.enabled and self.gradient_decoupled:
                # Forward value = output (real MLP); gradient flows through
                # y_hat to f.  Numerically (y_hat - y_hat.detach()) == 0, so
                # the model is bit-exact equal to the deployed model.
                return output + (y_hat - y_hat.detach())
            return output
        return _hook


@contextmanager
def feature_activations(model, transcoders, gradient_decoupled: bool = True):
    """
    Purpose: Convenience context manager -- yield a manager that records
             feature activations for every block on each forward pass.
    Method:  Wraps :class:`TranscoderHookManager` with sensible defaults.

    Args:
        model               (InterveneEncoder)
        transcoders         (Dict[int, JumpReLUTranscoder])
        gradient_decoupled  (bool): forward stays bit-exact to the real model;
                                    gradient flows through the transcoder
                                    features (default True).

    Yields:
        TranscoderHookManager
    """
    mgr = TranscoderHookManager(model, transcoders,
                                gradient_decoupled=gradient_decoupled,
                                enabled=True).attach()
    try:
        yield mgr
    finally:
        mgr.detach()


def feature_to_logit_attribution(model, transcoders, batch, outcome_idx: int,
                                 patient_idx: int = 0):
    """
    Purpose: For one (patient, outcome), compute grad × activation of the
             risk logit w.r.t. each transcoder feature at every (layer, token).
    Method:  Attach gradient-decoupled hooks with ``track_grad=True`` so the
             feature tensors are leaves; run :meth:`InterveneEncoder.predict`
             (forward is bit-exact equal to the real model); pick the desired
             risk logit; backprop and multiply gradient by the feature value.

    Args:
        model        (InterveneEncoder): must have ``task_heads`` attached.
        transcoders  (Dict[int, JumpReLUTranscoder])
        batch        (dict): standard EMR batch already on the right device.
        outcome_idx  (int): index into ``task_heads.risk_idx``.
        patient_idx  (int): row of the batch to explain.

    Returns:
        (attribution, pad_mask)

        * ``attribution`` -- Dict[int, Tensor[T, F]] per-layer signed
          contribution of each (token, feature) pair to the chosen risk logit.
          The sum across (L, t, F) equals (to first order) the deviation of
          the risk logit from its zero-feature baseline.
        * ``pad_mask`` -- Tensor[T] bool; True at valid (non-padded) positions.
    """
    if model.task_heads is None:
        raise RuntimeError("attach_task_heads() before attribution.")

    # Gradient checkpointing in the encoder uses use_reentrant=True, which
    # disconnects the autograd graph when none of the checkpoint inputs require
    # grad -- exactly our situation (all model params are frozen for
    # attribution). The features tensor we stash inside the hook then never
    # propagates a gradient to risk_logits.  Disable checkpointing for the
    # duration of the attribution pass and restore it on the way out.
    prev_ckpt = getattr(model, "use_checkpoint", False)
    model.use_checkpoint = False

    mgr = TranscoderHookManager(
        model, transcoders, gradient_decoupled=True, enabled=True,
    ).attach(track_grad=True)
    try:
        risk_logits, _, _, pad_mask = model.predict(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )
        target = risk_logits[patient_idx, outcome_idx]
        # Backprop into the feature leaves only.
        leaves = list(mgr.features.values())
        grads = torch.autograd.grad(target, leaves, retain_graph=False,
                                    allow_unused=False)

        attribution = {}
        for (layer_idx, f), g in zip(mgr.features.items(), grads):
            attribution[layer_idx] = (f[patient_idx] * g[patient_idx]).detach()
        return attribution, pad_mask[patient_idx].detach()
    finally:
        mgr.detach()
        model.use_checkpoint = prev_ckpt
