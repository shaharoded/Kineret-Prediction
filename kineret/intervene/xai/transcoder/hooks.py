"""
hooks.py
========

Forward-hook helpers that capture every ``MLP`` block's (input, output) pairs
from a frozen :class:`InterveneEncoder` -- without editing the source module.

The encoder's AdaLN block applies the MLP as::

    x = x + gate_mlp * self.mlp(norm_x)

so the MLP input we want is ``norm_x`` (the modulated post-LN value) and the
MLP output we want is the raw ``self.mlp(norm_x)`` *before* the gate scaling
and residual addition.  Both are exactly what ``self.mlp.__call__`` sees and
returns, which means a single forward-pre + forward hook on each
``block.mlp`` captures the right tensors with zero source edits.
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tqdm.auto import tqdm


@dataclass
class MLPIOBuffer:
    """
    Holds captured MLP inputs / outputs for one layer.

    Tensors are appended in micro-batches and concatenated lazily via
    :meth:`stack`.  Stored on CPU to avoid OOM during long capture runs;
    dtype is configurable (bf16 by default for ~2x RAM/disk savings).
    """
    inputs:  List[torch.Tensor] = field(default_factory=list)
    outputs: List[torch.Tensor] = field(default_factory=list)
    store_dtype: torch.dtype = torch.float32

    def stack(self):
        """Return (X_in, Y_out) as flat [N_tokens, d_model] CPU tensors."""
        X = torch.cat(self.inputs,  dim=0)
        Y = torch.cat(self.outputs, dim=0)
        return X, Y


@contextmanager
def capture_mlp_io(model, layer_indices=None, pad_mask_getter=None,
                   store_dtype: torch.dtype = torch.float32):
    """
    Purpose: Attach forward-pre + forward hooks on selected ``block.mlp``
             modules and yield one :class:`MLPIOBuffer` per layer.
    Method:  Pre-hook stores ``input[0]`` flattened over batch & time; the
             forward hook stores the matching output.  Padded positions are
             dropped if ``pad_mask_getter`` is supplied.

    Args:
        model            (InterveneEncoder): the frozen encoder.
        layer_indices    (Iterable[int]|None): which blocks to hook
                          (default: every block).
        pad_mask_getter  (callable|None): function ``() -> Tensor[B,T] (bool,
                          True at valid positions)``.  Called inside each
                          hook so the caller can stash the current batch's
                          pad mask between forward() calls.

    Yields:
        Dict[int, MLPIOBuffer] keyed by layer index.
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.blocks)))

    buffers = {i: MLPIOBuffer(store_dtype=store_dtype) for i in layer_indices}
    handles = []

    def make_pre_hook(layer_idx):
        def _pre_hook(_module, inputs):
            # `inputs` is a tuple of positional args; MLP takes a single tensor.
            x_in = inputs[0].detach()
            _stash_for_layer[layer_idx] = x_in       # see post-hook
        return _pre_hook

    def make_post_hook(layer_idx):
        def _post_hook(_module, _inputs, output):
            y_out = output.detach()
            x_in  = _stash_for_layer.pop(layer_idx, None)
            if x_in is None:
                return
            # Flatten [B, T, D] -> [B*T, D] and drop PAD if a mask is available.
            x_flat = x_in.reshape(-1, x_in.shape[-1])
            y_flat = y_out.reshape(-1, y_out.shape[-1])
            if pad_mask_getter is not None:
                m = pad_mask_getter()
                if m is not None:
                    keep = m.reshape(-1).to(x_flat.device)
                    x_flat = x_flat[keep]
                    y_flat = y_flat[keep]
            buf = buffers[layer_idx]
            # Cast to the configured store dtype on CPU to save RAM/disk.
            buf.inputs.append(x_flat.to(dtype=buf.store_dtype, device='cpu'))
            buf.outputs.append(y_flat.to(dtype=buf.store_dtype, device='cpu'))
        return _post_hook

    _stash_for_layer: dict = {}

    try:
        for i in layer_indices:
            mlp = model.blocks[i].mlp
            handles.append(mlp.register_forward_pre_hook(make_pre_hook(i)))
            handles.append(mlp.register_forward_hook(make_post_hook(i)))
        yield buffers
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def collect_activations(model, dataloader, device, max_batches=None,
                        layer_indices=None):
    """
    Purpose: Run the frozen encoder over a dataloader and harvest
             (mlp_in, mlp_out) pairs from every requested layer.
    Method:  Wraps :func:`capture_mlp_io`; runs ``model.encode`` (no task
             heads needed) and filters out padded tokens via the returned
             pad mask.

    Args:
        model         (InterveneEncoder): frozen encoder (call .eval()).
        dataloader    (DataLoader)       : provides standard EMR batches.
        device        (torch.device)
        max_batches   (int|None)         : early-stop after N batches (smoke test).
        layer_indices (list[int]|None)

    Returns:
        Dict[int, (X_in, Y_out)] -- both CPU tensors of shape [N_tokens, d_model].
    """
    model.eval()
    pad_holder = {"mask": None}
    getter = lambda: pad_holder["mask"]

    pad_idx = model.embedder.padding_idx
    with capture_mlp_io(model, layer_indices=layer_indices,
                        pad_mask_getter=getter) as buffers:
        for bi, batch in enumerate(tqdm(dataloader, desc="capture", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            # Set the pad mask BEFORE encode so the hooks see it (the encoder
            # forward fires the MLP hooks during, not after, the call).
            pad_holder["mask"] = (batch["position_ids"] != pad_idx)
            model.encode(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )
        # Final flush.
        out = {i: buf.stack() for i, buf in buffers.items()}
    return out


@torch.no_grad()
def collect_activations_cached(model, dataloader, device, cache_dir,
                               max_batches=None, layer_indices=None,
                               store_dtype: torch.dtype = torch.bfloat16,
                               overwrite: bool = True):
    """
    Purpose: Same as :func:`collect_activations`, but spills each layer's
             (X, Y) tensors to disk as bf16 .pt files instead of returning
             them in memory.
    Method:  Captures into per-layer CPU lists (bf16), then on context exit
             concatenates ONE layer at a time, writes to
             ``cache_dir/layer_{i}.pt``, and frees before moving to the next
             layer.  Peak RAM is bounded by the largest single layer rather
             than the sum across all layers.

    Args:
        model         (InterveneEncoder): frozen encoder (call .eval()).
        dataloader    (DataLoader)
        device        (torch.device)
        cache_dir     (str|Path)          : directory to write layer_{i}.pt files.
        max_batches   (int|None)
        layer_indices (list[int]|None)
        store_dtype   (torch.dtype)       : on-disk dtype (default bf16).
        overwrite     (bool)              : if False, skip layers whose cache
                                            file already exists.

    Returns:
        Dict[int, Path] mapping layer index -> cache file path. Each file
        contains a dict {"X": Tensor, "Y": Tensor, "dtype": str, "n": int}.
    """
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    pad_holder = {"mask": None}
    getter = lambda: pad_holder["mask"]
    pad_idx = model.embedder.padding_idx

    if layer_indices is None:
        layer_indices = list(range(len(model.blocks)))

    # If everything is already cached and overwrite=False, short-circuit.
    paths = {i: cache_dir / f"layer_{i}.pt" for i in layer_indices}
    if not overwrite and all(p.exists() for p in paths.values()):
        print(f"[cache] all {len(paths)} layer caches present, skipping capture.")
        return paths

    with capture_mlp_io(model, layer_indices=layer_indices,
                        pad_mask_getter=getter,
                        store_dtype=store_dtype) as buffers:
        for bi, batch in enumerate(tqdm(dataloader, desc="capture", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_holder["mask"] = (batch["position_ids"] != pad_idx)
            model.encode(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )

        # Spill one layer at a time so peak RAM is bounded by a single layer
        # rather than the full {layer: (X,Y)} dict.
        for li in list(buffers.keys()):
            buf = buffers[li]
            X = torch.cat(buf.inputs,  dim=0)
            Y = torch.cat(buf.outputs, dim=0)
            n = X.shape[0]
            path = paths[li]
            torch.save({"X": X, "Y": Y, "dtype": str(store_dtype), "n": n}, path)
            mb = (X.numel() + Y.numel()) * X.element_size() / 1e6
            print(f"[cache] layer {li}: wrote {path.name}  shape={tuple(X.shape)}  "
                  f"dtype={store_dtype}  size~{mb:.1f} MB")
            # Free this layer before touching the next.
            buf.inputs.clear(); buf.outputs.clear()
            del X, Y, buf
            buffers[li] = None
            gc.collect()
    return paths


def load_layer_cache(path, map_location: str = "cpu", mmap: bool = True):
    """
    Purpose: Load a per-layer cache produced by
             :func:`collect_activations_cached`.
    Method:  ``torch.load`` with mmap=True so tensors are paged in lazily;
             keeps RAM usage proportional to the active mini-batch, not the
             whole layer.

    Args:
        path        (str|Path)
        map_location(str)
        mmap        (bool): mmap the file (default True).

    Returns:
        (X, Y) tensors in their stored dtype.
    """
    payload = torch.load(str(path), map_location=map_location,
                         mmap=mmap, weights_only=True)
    return payload["X"], payload["Y"]
