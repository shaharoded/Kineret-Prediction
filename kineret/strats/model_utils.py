"""This file contain common utility functions."""
import argparse
from datetime import datetime
import string
import os
import random
import json
from pytz import timezone
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer


def set_seed(seed: int) -> None:
    """
    Purpose: Seed python, numpy and torch.
    Method:  This used to be `transformers.set_seed`, which does exactly this.
             Importing it dragged in the whole transformers package -- and with
             it a TensorFlow import that spends several seconds failing to
             register cuFFT/cuDNN/cuBLAS against the CUDA runtime torch has
             already claimed, printing a wall of red that looks like a GPU
             fault and is not one. Nothing else here needed the dependency.

    Args:
        seed (int): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
from typing import Any, Union


def get_curr_time() -> str:
    """Get current date and time in PST as str."""
    return datetime.now().astimezone(
            timezone('US/Pacific')).strftime("%d/%m/%Y %H:%M:%S")


class Logger:
    """Class to write message to both output_dir/filename.txt and terminal."""
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool=True) -> None:
        "write the message"
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'):
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print (message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n')


def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    set_seed(seed)


def count_parameters(logger: Logger, model: nn.Module):
    """Print model parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: '+str(total))
    logger.write('# trainable parameters: '+str(trainable)+', '
                 +str(100*trainable/total)+'%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write('#params by dtype:')
    for k, v in dtypes.items():
        logger.write(str(k)+': '+str(v)+', '+str(100*v/total)+'%')


class TimeSeriesModel(nn.Module):
    """
    Shared head for the supervised sparse-time-series baseline.

    Kineret note: this is the **ss-STraTS** configuration only -- the
    supervised-only variant from Tipirneni & Reddy. The two-stage variant's
    self-supervised forecasting head is gone, along with the `--pretrain` /
    `--load_ckpt_path` plumbing: this cohort is small and is the only data
    available, so a forecasting stage has no unlabelled pool to pay for itself
    with, and carrying the code would only invite an unused branch to rot.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.demo_emb = nn.Sequential(nn.Linear(args.D, args.hid_dim * 2),
                                      nn.Tanh(),
                                      nn.Linear(args.hid_dim * 2, args.hid_dim))
        ts_demo_emb_size = args.hid_dim * 2
        # The prediction head emits one vector of length (2 * num_labels + 1),
        # so this model answers exactly the same three questions INTERVenE does:
        #
        #   [0 : K]        per-outcome risk logits      -- will it happen?
        #   [K : 2K]       per-outcome z-scored time    -- if so, when?
        #   [2K]           z-scored length of stay      -- how long is the stay?
        #
        # The time block was added so the prediction task is identical across
        # every model in the benchmark rather than INTERVenE alone being scored
        # on onset timing. It is trained on positives only -- a complication
        # that does not occur has no onset to predict.
        head_out_dim = 2 * args.num_labels + 1
        self.binary_head = nn.Linear(ts_demo_emb_size, head_out_dim)
        self.register_buffer('pos_class_weight',
                             torch.as_tensor(args.pos_class_weight).float())
        self.los_loss_weight = float(getattr(args, 'los_loss_weight', 1.0))
        # Matches INTERVenE's `phase3_time_lambda` so the two models weight
        # onset timing against risk the same way.
        self.time_loss_weight = float(getattr(args, 'time_loss_weight', 0.1))

    def binary_cls_final(self, logits, labels, los_target_norm=None, los_mask=None,
                         time_target_norm=None, time_mask=None):
        """
        Purpose: Turn the combined head into a loss (training) or a prediction
                 block (eval).
        Method:  The head vector splits three ways:

                   [0 : K]   risk logits -> multi-label BCE with pos_weight
                   [K : 2K]  z-scored onset time -> masked MSE, positives only
                   [2K]      z-scored length of stay -> masked MSE, discharged only

                 Both regressions are masked because both are conditional: a
                 complication that never happens has no onset, and a patient who
                 died has no length of stay. Dividing by the mask sum keeps the
                 gradient scale steady on batches where few samples qualify.

        Args:
            logits           (Tensor): [bsz, 2K+1] raw head output.
            labels           (Tensor|None): [bsz, K] binary targets; None at eval.
            los_target_norm  (Tensor|None): [bsz] z-scored length of stay.
            los_mask         (Tensor|None): [bsz] 1 where LoS is known.
            time_target_norm (Tensor|None): [bsz, K] z-scored onset hours.
            time_mask        (Tensor|None): [bsz, K] 1 where the outcome occurs.

        Returns:
            Tensor: scalar loss when `labels` is given, else a [bsz, 2K+1] block
                    of (probabilities, normalised times, normalised LoS) that the
                    evaluator denormalises.
        """
        K = self.args.num_labels
        binary_logits = logits[:, :K]
        time_pred_norm = logits[:, K:2 * K]
        los_pred_norm = logits[:, 2 * K]

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                binary_logits, labels, pos_weight=self.pos_class_weight)
            if time_target_norm is not None and time_mask is not None:
                sq_err = (time_pred_norm - time_target_norm) ** 2 * time_mask
                denom = torch.clamp(time_mask.sum(), min=1.0)
                loss = loss + self.time_loss_weight * sq_err.sum() / denom
            if los_target_norm is not None and los_mask is not None:
                # Masked MSE on z-scored LoS; only RELEASE-discharged patients
                # contribute.
                sq_err = (los_pred_norm - los_target_norm) ** 2 * los_mask
                denom = torch.clamp(los_mask.sum(), min=1.0)
                loss = loss + self.los_loss_weight * sq_err.sum() / denom
            return loss

        # Eval: probabilities, then the two normalised regressions.
        binary_probs = torch.sigmoid(binary_logits)
        return torch.cat([binary_probs, time_pred_norm,
                          los_pred_norm.unsqueeze(-1)], dim=-1)



class CycleIndex:
    """Class to generate batches of training ids,
    shuffled after each epoch."""
    def __init__(self, indices:Union[int,list], batch_size: int,
                 shuffle: bool=True) -> None:
        if type(indices)==int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, self.pointer + self.batch_size
        # If we have a full batch within this epoch, then get it.
        if end <= self.num_samples:
            if end==self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]
        # Otherwise, fill the batch with samples from next epoch.
        last_batch_indices_incomplete = self.indices[start:]
        remaining = self.batch_size - (self.num_samples-start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last_batch_indices_incomplete,
                               self.indices[:remaining]))