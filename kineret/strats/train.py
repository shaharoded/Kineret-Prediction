"""
Training driver for ss-STraTS.

A function-shaped port of `med-transformers-baseline/main.py`, reduced to the
supervised-only variant this study compares against. Four changes:

1. It is callable (`run(...)`) rather than argv-only, so the notebook can drive
   the whole grid in one process.
2. **Single stage.** The self-supervised forecasting stage is gone: this cohort
   is small and is the only data available, so there is no unlabelled pool for
   it to pay for itself with.
3. Training reads the cohort's AUGMENTED samples -- one per (patient, context
   window) -- while validation and test use the single evaluation window. Model
   selection therefore happens under the same conditions the results are
   reported under.
4. The held-out test split is scored ONCE, at the end, from the best-val
   checkpoint. The baseline re-scored test at every validation round, which is
   both slow and an invitation to peek. Intervals come from
   `kineret.evaluation.score_run`.
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from kineret.config import data_config as C
from kineret.config import paths
from kineret.evaluation import score_run
from kineret.strats import config as SC
from kineret.strats.dataset import Dataset
from kineret.strats.evaluator import Evaluator
from kineret.intervene.utils import save_checkpoint
from kineret.strats.model_utils import Logger, count_parameters, set_all_seeds
from kineret.strats.strats import Strats


def _build_args(use_qa, output_dir, device=None, seed=C.SEED, overrides=None):
    """
    Purpose: Assemble the Namespace the ported baseline code expects.
    Method:  Start from this model's own defaults in `strats/config.py`, then
             apply the run's QA arm and any caller overrides. Shared study
             decisions are not here -- they reached the pickle at preprocess
             time.

    Args:
        use_qa     (bool):      QA ablation arm.
        output_dir (str):       Run directory.
        device     (str|None):  Torch device string; auto-selects CUDA.
        seed       (int):       RNG seed.
        overrides  (dict|None): Hyperparameter overrides.

    Returns:
        argparse.Namespace: Fully populated args.
    """
    base = dict(SC.STRATS_SETTINGS)
    base.update(overrides or {})

    args = argparse.Namespace(
        dataset=SC.dataset_name(use_qa),
        train_frac=1.0, run="1o1",
        model_type="strats",
        max_obs=base["max_obs"],
        hid_dim=base["hid_dim"],
        num_layers=base["num_layers"],
        num_heads=base["num_heads"],
        dropout=base["dropout"],
        attention_dropout=base["attention_dropout"],
        output_dir=output_dir,
        seed=seed,
        max_epochs=base["max_epochs"],
        patience=base["patience"],
        lr=base["lr"],
        train_batch_size=base["train_batch_size"],
        gradient_accumulation_steps=base["gradient_accumulation_steps"],
        eval_batch_size=base["eval_batch_size"],
        print_train_loss_every=100,
        validate_after=-1,
        validate_every=None,
        los_loss_weight=base["los_loss_weight"],
        time_loss_weight=base["time_loss_weight"],
        # Intervals come from the shared scorer at the end of the run, so the
        # in-loop evaluator stays cheap.
        bootstrap_resamples=0,
        bootstrap_seed=SC.BOOTSTRAP_SEED,
        use_qa=use_qa,
        context_days=C.EVAL_CONTEXT_DAYS,
    )
    args.device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    return args


def _train_loop(args, dataset, model, evaluator, model_path_best):
    """
    Purpose: The shared optimisation loop for both stages.
    Method:  AdamW, gradient clipping at 0.3, validation every epoch, and
             early stopping on macro AUROC+AUPRC over the VALIDATION split,
             which uses the single evaluation window. The best checkpoint is the
             only thing that survives.

    Args:
        args            (Namespace): Run configuration.
        dataset         (Dataset):   The augmented sample dataset.
        model           (nn.Module): Strats.
        evaluator       (Evaluator): Matching evaluator.
        model_path_best (str):       Checkpoint destination.

    Returns:
        dict|None: Best validation result.
    """
    # torch's AdamW, not the one transformers used to re-export: that symbol
    # was removed from `transformers.optimization` in newer releases, and the
    # two are the same algorithm anyway (HF's wrapper defaulted to
    # correct_bias=True, which is torch's only behaviour). Depending on it made
    # this arm fail to import on any environment past the pinned version.
    from torch.optim import AdamW

    num_train = len(dataset.splits["train"])
    batches_per_epoch = max(num_train / args.train_batch_size, 1.0)
    args.max_steps = int(round(batches_per_epoch) * args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(np.ceil(batches_per_epoch))
    args.logger.write(f"\nBatches/epoch = {batches_per_epoch:.1f}, max_steps = {args.max_steps}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    cum_loss, num_steps, num_batches = 0.0, 0, 0
    wait, best_val_metric, best_val_res = args.patience, -np.inf, None

    model.train()
    for step in tqdm(range(args.max_steps), desc="ss-STraTS",
                      leave=False, dynamic_ncols=True, mininterval=1.0):
        batch = {k: v.to(args.device) for k, v in dataset.get_batch().items()}
        loss = model(**batch)

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        cum_loss += loss.item()
        num_steps += 1
        num_batches += 1
        if num_steps % args.print_train_loss_every == 0:
            args.logger.write(f"\nTrain-loss at step {num_steps}: {cum_loss / num_batches:.5f}")
            cum_loss, num_batches = 0.0, 0

        if num_steps >= args.validate_after and num_steps % args.validate_every == 0:
            val_res = evaluator.evaluate(model, dataset, "val", train_step=step)
            model.train(True)
            metric = val_res["auprc"] + val_res["auroc"]
            if metric > best_val_metric:
                best_val_metric, best_val_res = metric, val_res
                save_checkpoint(model.state_dict(), model_path_best)
                args.logger.write(f"\nSaved ckpt at {model_path_best}")
                wait = args.patience
            else:
                wait -= 1
                args.logger.write(f"Updating wait to {wait}")
                if wait == 0:
                    args.logger.write("Patience reached")
                    break

    # A very short run (or a split smaller than one validation interval) can
    # exit the loop without ever validating, leaving no checkpoint for the next
    # stage to load. Force one pass so `checkpoint_best.bin` always exists.
    if best_val_res is None:
        best_val_res = evaluator.evaluate(model, dataset, "val", train_step=None)
        save_checkpoint(model.state_dict(), model_path_best)
        args.logger.write("\nNo validation fired during training; saved final "
                          f"weights at {model_path_best}")

    args.logger.write(f"Final val res: {best_val_res}")
    return best_val_res


def run(context_days=None, use_qa=False, output_root=None, device=None,
        seed=C.SEED, overrides=None, bootstrap_resamples=SC.BOOTSTRAP_RESAMPLES,
        **_ignored):
    """
    Purpose: Train and score ss-STraTS for one QA arm.
    Method:  Supervised training on the augmented samples, model selection on
             the single-window validation split, then one held-out test pass
             from the best-val checkpoint, scored by the shared scorer.

    Args:
        context_days        (int|None): Accepted for call-site symmetry; the
                            evaluation window is a study-level decision and
                            comes from the pickle, not from here.
        use_qa              (bool):     QA ablation arm.
        output_root         (str|None): Defaults to `outputs/`.
        device              (str|None): Torch device string.
        seed                (int):      RNG seed.
        overrides           (dict|None): Hyperparameter overrides.
        bootstrap_resamples (int):      Resamples for the final intervals.

    Returns:
        dict: {'output_dir': str, 'scores': dict}
    """
    output_root = output_root or paths.OUTPUT_DIR
    run_dir = os.path.join(output_root, "strats", "qa" if use_qa else "noqa")
    os.makedirs(run_dir, exist_ok=True)

    args = _build_args(use_qa, run_dir, device, seed=seed, overrides=overrides)
    args.logger = Logger(run_dir, "log.txt")
    args.logger.write(f"\n[ss-STraTS] {vars(args)}")
    set_all_seeds(seed)

    dataset = Dataset(args)
    model = Strats(args).to(args.device)
    count_parameters(args.logger, model)

    ckpt_best = os.path.join(run_dir, "checkpoint_best.bin")
    evaluator = Evaluator(args)
    _train_loop(args, dataset, model, evaluator, ckpt_best)

    # --- single held-out test pass from the best-val checkpoint -----------
    if os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=args.device))
    test_res = evaluator.evaluate(model, dataset, "test", train_step=None)
    args.logger.write(f"Final test res: {test_res}")

    meta = {
        "model": "strats",
        "use_qa": use_qa,
        "seed": seed,
        "outcome_names": list(args.outcome_names),
        "context_days": C.EVAL_CONTEXT_DAYS,
        "train_context_days": list(C.TRAIN_CONTEXT_DAYS),
        "label_window_hours": [C.EVAL_CONTEXT_DAYS * 24.0, C.HORIZON_END_DAYS * 24.0],
        "n_test": int(len(dataset.splits["test"])),
        "n_train_samples": int(len(dataset.splits["train"])),
        "n_val": int(len(dataset.splits["val"])),
        "hyperparameters": {k: v for k, v in vars(args).items()
                            if isinstance(v, (int, float, str, bool)) and k != "device"},
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    scores = score_run(run_dir, n_resamples=bootstrap_resamples, seed=SC.BOOTSTRAP_SEED)
    return {"output_dir": run_dir, "scores": scores}
