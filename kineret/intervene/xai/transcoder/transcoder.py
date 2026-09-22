"""
transcoder.py
=============

A single-layer sparse transcoder for a SwiGLU MLP.

The module learns an approximation::

    mlp_out_hat  =  W_dec . f(x_in) + b_dec
    f(x_in)      =  JumpReLU( W_enc . x_in + b_enc ; theta )

where :func:`JumpReLU` is a hard-thresholded ReLU (zero below a learnable
per-feature threshold ``theta``).  Training optimises MSE against the real
MLP output plus an L0 surrogate that pushes ``theta`` upward so most features
stay silent at any given token.

The choice of JumpReLU (rather than L1 / TopK) follows the latest interp
literature: it gives clean L0 control without the well-known L1 shrinkage
bias, and a single threshold per feature is easy to plot in a demo.
"""

import math
import torch
import torch.nn as nn


class _StepWithSTE(torch.autograd.Function):
    """
    Heaviside step with a straight-through gradient.

    Forward:  1{ x > 0 }
    Backward: rectangular kernel of width ``bandwidth`` centred at 0 -- this is
              the standard JumpReLU STE from Rajamanoharan et al. 2024.
    """

    @staticmethod
    def forward(ctx, x, bandwidth: float):
        ctx.save_for_backward(x)
        ctx.bandwidth = bandwidth
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        bw = ctx.bandwidth
        # Rectangular kernel: 1/bw inside [-bw/2, +bw/2], 0 outside.
        grad = (x.abs() < (bw / 2.0)).to(grad_out.dtype) / bw
        return grad_out * grad, None


def jump_relu(pre_act: torch.Tensor, theta: torch.Tensor, bandwidth: float = 1e-3) -> torch.Tensor:
    """
    Purpose: JumpReLU activation -- zero below per-feature threshold, identity above.
    Method:  ``(pre_act - theta) * 1{pre_act > theta}`` with a rectangular STE
             on the indicator so ``theta`` receives gradient.

    Args:
        pre_act   (Tensor): [..., F] pre-activations.
        theta     (Tensor): [F] non-negative per-feature thresholds.
        bandwidth (float):  STE kernel width (small positive).

    Returns:
        Tensor [..., F] post-activation features.
    """
    gate = _StepWithSTE.apply(pre_act - theta, bandwidth)
    return pre_act * gate


class JumpReLUTranscoder(nn.Module):
    """
    Sparse transcoder bottleneck for a single MLP layer.

    Parameters
    ----------
    d_model      : encoder hidden dim (matches ``cfg['embed_dim']``).
    n_features   : dictionary size (typically 8x-32x d_model).
    bandwidth    : JumpReLU STE bandwidth.
    init_theta   : initial value for all per-feature thresholds.
    """

    def __init__(self, d_model: int, n_features: int,
                 bandwidth: float = 1e-3, init_theta: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.bandwidth = bandwidth

        # Encoder / decoder: tied init (decoder = encoder^T) is a common
        # warm-start and tends to converge faster than independent inits.
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # log_theta keeps theta strictly positive via exp(); init so
        # exp(log_theta) == init_theta.
        self.log_theta = nn.Parameter(torch.full((n_features,), math.log(init_theta)))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            # Unit-norm decoder columns -- standard SAE/transcoder convention so
            # feature magnitudes live in `f` rather than smuggled into `W_dec`.
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    # ----------------------------------------------------------------- API #
    @property
    def theta(self) -> torch.Tensor:
        """Per-feature positive thresholds (n_features,)."""
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose: Project input into the sparse feature space.
        Method:  Linear -> JumpReLU.

        Args:
            x (Tensor): [..., d_model] MLP input (post-ln, pre-MLP).

        Returns:
            Tensor [..., n_features] sparse, non-negative features.
        """
        pre = x @ self.W_enc + self.b_enc
        return jump_relu(pre, self.theta, bandwidth=self.bandwidth)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Purpose: Reconstruct the MLP output from sparse features.
        Method:  Linear with unit-norm decoder columns (norm re-applied on the
                 fly so the constraint is robust to optimiser drift).
        """
        # Re-normalise on the fly -- cheap and keeps the magnitude-in-`f`
        # convention even if the optimiser has nudged W_dec.
        w = self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return f @ w + self.b_dec

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [..., d_model]

        Returns:
            f      : [..., n_features] sparse features.
            x_hat  : [..., d_model]    reconstructed MLP output.
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        return f, x_hat

    # ---------------------------------------------------------- diagnostics #
    @torch.no_grad()
    def l0(self, x: torch.Tensor) -> torch.Tensor:
        """Mean number of active features per token (a scalar)."""
        f = self.encode(x)
        return (f > 0).float().sum(dim=-1).mean()
