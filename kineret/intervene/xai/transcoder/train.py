"""
train.py
========

Fit one :class:`JumpReLUTranscoder` per encoder layer on cached MLP I/O pairs.

Loss
----
    L = MSE(W_dec . f(x) + b_dec , y_true)  +  lambda_l0(t) * L0(f)

where ``L0(f) = sum_i 1{f_i > 0}`` (Heaviside, STE'd through JumpReLU's
threshold gate).  No L1 -- we want sparsity *without* magnitude shrinkage on
the active features.

Schedulers
----------
* LR: linear warmup -> cosine decay to ``lr_min`` over the full training run.
* lambda_l0: linear ramp from 0 -> target across the first ``lambda_warmup_frac``
  of total steps.  Letting MSE settle before the L0 penalty kicks in avoids
  the cold-start "all features die" failure mode.

I/O caching
-----------
``train_one`` and ``train_transcoders`` accept either dense tensors or paths
to per-layer cache files written by
:func:`xai.transcoder.hooks.collect_activations_cached`.  When given paths,
the dataset is built on a memory-mapped view of the file, so peak RAM is
bounded by the active mini-batch instead of the full layer.
"""

import gc
import math
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from kineret.intervene.xai.transcoder.transcoder import JumpReLUTranscoder, _StepWithSTE
from kineret.intervene.xai.transcoder.hooks import load_layer_cache


# ---------------------------------------------------------------- losses --- #
def _l0_surrogate(f: torch.Tensor) -> torch.Tensor:
    """
    Differentiable L0 = sum over features of Heaviside(f).

    JumpReLU produces exact zeros below threshold, so Heaviside is a clean
    indicator.  We re-wrap with the STE so theta still gets gradient from L0.
    """
    return _StepWithSTE.apply(f, 1e-3).sum(dim=-1).mean()


# ---------------------------------------------------------------- data ---- #
class _MMapPairDataset(Dataset):
    """
    Purpose: Dataset over an (X, Y) cache loaded with ``torch.load(mmap=True)``.
    Method:  Indexes directly into the memory-mapped tensors and casts the
             requested rows to float32 lazily.  Avoids materialising the full
             layer in RAM.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape[0] == Y.shape[0], "X and Y must align on dim 0"
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Cast at fetch time -- stored dtype may be bf16/fp16; training math
        # stays fp32 for stable gradients.
        return self.X[idx].float(), self.Y[idx].float()


def _build_dataset(X_or_path, Y_or_none) -> Dataset:
    """
    Accepts either (X_tensor, Y_tensor) or (path, None).  Returns a Dataset
    suitable for DataLoader iteration.  Memory-maps if given a path.
    """
    if isinstance(X_or_path, (str, Path)):
        assert Y_or_none is None, "When passing a cache path, second arg must be None."
        X, Y = load_layer_cache(X_or_path, mmap=True)
        return _MMapPairDataset(X, Y)
    # Dense tensor path -- keep prior fp32 behaviour.
    return TensorDataset(X_or_path, Y_or_none)


# ---------------------------------------------------------- schedulers ---- #
def _make_lr_lambda(total_steps: int, warmup_frac: float, min_factor: float):
    """
    Linear warmup over ``warmup_frac`` of steps, then cosine decay from 1.0
    down to ``min_factor`` over the rest.  Returned callable maps step -> mult.
    """
    warm = max(1, int(total_steps * warmup_frac))

    def _lr(step):
        if step < warm:
            return step / warm
        # cosine from 1.0 -> min_factor
        progress = (step - warm) / max(1, total_steps - warm)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_factor + (1.0 - min_factor) * cos
    return _lr


def _lambda_l0_at(step: int, total_steps: int, target: float,
                  warmup_frac: float) -> float:
    """Linear ramp 0 -> target across the first ``warmup_frac`` of steps."""
    warm = max(1, int(total_steps * warmup_frac))
    if step >= warm:
        return target
    return target * (step / warm)


# ---------------------------------------------------------------- train --- #
def train_one(transcoder: JumpReLUTranscoder,
              X: Union[torch.Tensor, str, Path],
              Y: Union[torch.Tensor, None] = None,
              n_epochs: int = 20, batch_size: int = 4096,
              lr: float = 3e-4, lr_min_factor: float = 0.1,
              lr_warmup_frac: float = 0.05,
              lambda_l0: float = 3e-4,
              lambda_warmup_frac: float = 0.2,
              device: str = "cpu", desc: str = "transcoder",
              num_workers: int = 0):
    """
    Purpose: Train a single transcoder on (X, Y) activation pairs.
    Method:  AdamW on MSE + lambda_l0(t) * L0, with cosine LR (linear warmup)
             and linear lambda_l0 warmup so MSE can settle before sparsity
             pressure kicks in.

    Args:
        transcoder         (JumpReLUTranscoder)
        X                  : Tensor [N, d_model] OR path to a cache file from
                             :func:`collect_activations_cached`.
        Y                  : Tensor [N, d_model] OR ``None`` if X is a path.
        n_epochs           (int)
        batch_size         (int)
        lr                 (float): peak LR (after warmup).
        lr_min_factor      (float): cosine floor as a fraction of ``lr``.
        lr_warmup_frac     (float): warmup fraction of total steps.
        lambda_l0          (float): target sparsity weight.
        lambda_warmup_frac (float): fraction of steps to ramp lambda_l0.
        device             (str)
        desc               (str): tqdm label.
        num_workers        (int): DataLoader workers; 0 is safest with mmap.

    Returns:
        dict with epoch-level history: mse, l0, total, lr, lambda_l0.
    """
    transcoder = transcoder.to(device)
    ds = _build_dataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False,
                    num_workers=num_workers, pin_memory=(device != "cpu"))
    opt = torch.optim.AdamW(transcoder.parameters(), lr=lr)

    total_steps = max(1, n_epochs * len(dl))
    lr_lambda = _make_lr_lambda(total_steps, lr_warmup_frac, lr_min_factor)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    hist = {"mse": [], "l0": [], "total": [], "lr": [], "lambda_l0": []}
    global_step = 0
    for ep in range(n_epochs):
        tot_mse = tot_l0 = tot = 0.0
        n_steps = 0
        pbar = tqdm(dl, desc=f"{desc} ep{ep+1}/{n_epochs}", leave=False,
                    mininterval=1.0)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            f, y_hat = transcoder(xb)
            mse = nn.functional.mse_loss(y_hat, yb)
            l0  = _l0_surrogate(f)
            lam = _lambda_l0_at(global_step, total_steps, lambda_l0,
                                lambda_warmup_frac)
            loss = mse + lam * l0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_mse += mse.item(); tot_l0 += l0.item(); tot += loss.item()
            n_steps += 1; global_step += 1

        ep_mse = tot_mse / max(1, n_steps)
        ep_l0  = tot_l0  / max(1, n_steps)
        ep_lr  = opt.param_groups[0]["lr"]
        ep_lam = _lambda_l0_at(global_step, total_steps, lambda_l0,
                               lambda_warmup_frac)
        hist["mse"].append(ep_mse)
        hist["l0"].append(ep_l0)
        hist["total"].append(tot / max(1, n_steps))
        hist["lr"].append(ep_lr)
        hist["lambda_l0"].append(ep_lam)
        # Per-epoch text used to spam stdout (~hundreds of lines per layer).
        # The inner per-step tqdm bar already shows live progress and final
        # metrics land in `hist` -- no extra printout per epoch.
    return hist


def train_transcoders(activations: dict, d_model: int, n_features: int,
                      n_epochs: int = 20, batch_size: int = 4096,
                      lr: float = 3e-4, lr_min_factor: float = 0.1,
                      lr_warmup_frac: float = 0.05,
                      lambda_l0: float = 3e-4,
                      lambda_warmup_frac: float = 0.2,
                      bandwidth: float = 1e-3, init_theta: float = 0.05,
                      device: str = "cpu", num_workers: int = 0):
    """
    Purpose: Fit one transcoder per captured layer.
    Method:  Iterate :func:`train_one` over ``activations``.

    Args:
        activations : Either
                      * Dict[int, (X, Y)]   from :func:`collect_activations`, or
                      * Dict[int, Path]     from :func:`collect_activations_cached`.
                      The two are handled transparently -- paths are mmap'd one
                      layer at a time and freed between layers, so peak RAM is
                      bounded by the active layer rather than the sum.

        (other args as in :func:`train_one`).

    Returns:
        Dict[int, JumpReLUTranscoder], Dict[int, history-dict].
    """
    models, histories = {}, {}
    items = list(activations.items())
    # Outer tqdm: one tick per transcoder.  Per-epoch progress stays inside
    # train_one's own bar; this replaces the noisy "=== Transcoder k/n ===" /
    # "-> done" prints with a single live bar.
    outer = tqdm(items, desc="transcoders", total=len(items), leave=True)
    for layer_idx, payload in outer:
        if isinstance(payload, (str, Path)):
            X_arg, Y_arg = payload, None
            X_peek, _ = load_layer_cache(payload, mmap=True)
            shape_str = f"N={X_peek.shape[0]},d={X_peek.shape[1]},mmap"
            del X_peek
        else:
            X_arg, Y_arg = payload
            shape_str = f"X={tuple(X_arg.shape)}"

        outer.set_description(f"transcoders [layer {layer_idx}  {shape_str}  "
                              f"n_feat={n_features}]")
        tc = JumpReLUTranscoder(d_model=d_model, n_features=n_features,
                                bandwidth=bandwidth, init_theta=init_theta)
        h = train_one(tc, X_arg, Y_arg,
                      n_epochs=n_epochs, batch_size=batch_size,
                      lr=lr, lr_min_factor=lr_min_factor,
                      lr_warmup_frac=lr_warmup_frac,
                      lambda_l0=lambda_l0,
                      lambda_warmup_frac=lambda_warmup_frac,
                      device=device, desc=f"layer{layer_idx}",
                      num_workers=num_workers)
        models[layer_idx] = tc.eval().cpu()  # park on CPU to free GPU between layers
        histories[layer_idx] = h
        # Surface the per-layer final metrics on the outer bar's postfix.
        outer.set_postfix(last_layer=layer_idx,
                          mse=f"{h['mse'][-1]:.3e}",
                          L0=f"{h['l0'][-1]:.1f}")

        # Aggressively reclaim memory between layers -- the prior layer's
        # mmap view + GPU buffers can otherwise stack up.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    outer.close()
    return models, histories


# ----------------------------------------------------- (de)serialisation --- #
def save_transcoders(models: dict, path):
    """Serialise a {layer_idx: JumpReLUTranscoder} bundle to one file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layers": sorted(models.keys()),
        "d_model": next(iter(models.values())).d_model,
        "n_features": next(iter(models.values())).n_features,
        "state": {i: m.state_dict() for i, m in models.items()},
    }
    torch.save(payload, path)


def load_transcoders(path, map_location: str = "cpu"):
    """Restore a bundle produced by :func:`save_transcoders`."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    out = {}
    for i in payload["layers"]:
        tc = JumpReLUTranscoder(d_model=payload["d_model"],
                                n_features=payload["n_features"])
        tc.load_state_dict(payload["state"][i])
        out[i] = tc.eval()
    return out
