"""
Three-phase training driver for INTERVenE-Enc on the Kineret cohort.

Replaces the notebook / `api.py` driver from the upstream repo with a single
callable that owns one cell of the K x QA sweep:

    Phase 1  train the embedder (token + time + patient context)
    Phase 2  MLM-pretrain the bidirectional encoder on the FULL trajectory
    Phase 3  attach TaskHeads and fine-tune on the [0, K*24] h seed, supervised
             against the shared cohort labels for (K*24, HORIZON*24]

Two properties are worth stating explicitly, because they are what make the
benchmark honest:

* **Splits and labels come from `kineret.cohort`.** Phase 3 is supervised
  against the same 0/1 matrix STraTS and LogReg see, not against outcomes
  re-read from the token stream.
* **Phases 1 and 2 only ever see train/val patients.** They pretrain on those
  patients' full trajectories, which is their own future, not held-out data.
  The test split is untouched until the final inference pass.

Every phase runs per cell rather than being shared across K. That is
deliberate: the context vector carries QA features aggregated over [0, K*24],
so an embedder trained at one K is not the same function at another.
"""

import json
import gc
import os

import numpy as np
import pandas as pd
import torch

from kineret.cohort import Cohort
from kineret.config import data_config as C
from kineret.config import paths
from kineret.evaluation import score_run, write_predictions
from kineret.io_utils import load_table, normalise_temporal

# Config must be configured BEFORE the package modules are imported, because
# they pull it in with `import *`. Importing the config module alone is safe.
from kineret.intervene.config import dataset_config as DCFG
from kineret.intervene.config import model_config as MCFG


def _build_label_frames(cohort: Cohort, samples: pd.DataFrame):
    """
    Purpose: Turn the cohort's label block into the two matrices EMRDataset wants.
    Method:  Indexed by SAMPLE, so a training patient's K=2 row and K=5 row carry
             different labels -- each sample's window is (k*24, HORIZON*24].
             Cohort targets come straight from `labels_for_samples`.
             RELEASE_EVENT is added on top, derived from length-of-stay:
             INTERVenE keeps it on the time head as the length-of-stay
             regression even though the risk head drops it. DEATH_EVENT is added
             the same way when the support filter cut it, since TERMINAL_OUTCOMES
             puts it in the head regardless.

    Args:
        cohort  (Cohort):       Shared cohort artefact.
        samples (pd.DataFrame): sample_id / PatientId / k / split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (labels 0/1, first-occurrence hours),
                                           both sample_id-indexed.
    """
    block = cohort.labels_for_samples(samples).set_index("sample_id")
    index = block.index
    hi = C.HORIZON_END_DAYS * 24.0
    lo_per_sample = block["k"].to_numpy() * 24.0

    labels = pd.DataFrame(index=index)
    times = pd.DataFrame(index=index)
    for name in cohort.outcome_names:
        labels[name] = block[name].fillna(0).astype("float32")
        times[name] = block[f"{name}__first_hour"].astype("float32")

    # RELEASE: positive when the discharge falls in THIS sample's label window.
    los = block["length_of_stay_hours"].to_numpy(dtype=float)
    in_window = (los > lo_per_sample) & (los <= hi)
    labels[DCFG.RELEASE_TOKEN] = in_window.astype("float32")
    times[DCFG.RELEASE_TOKEN] = np.where(in_window, los, np.inf).astype("float32")

    # DEATH may have been support-filtered out of the head; TERMINAL_OUTCOMES
    # puts it back, so give it a column either way.
    if DCFG.DEATH_TOKEN not in labels.columns:
        deaths = cohort.events[cohort.events["outcome"] == DCFG.DEATH_TOKEN]
        first = deaths.groupby("PatientId")["hours"].min()
        hours = block["PatientId"].map(first).to_numpy(dtype=float)
        occurs = np.isfinite(hours) & (hours > lo_per_sample) & (hours <= hi)
        labels[DCFG.DEATH_TOKEN] = occurs.astype("float32")
        times[DCFG.DEATH_TOKEN] = np.where(occurs, hours, np.inf).astype("float32")

    return labels, times


def _prepare_split(abstract_df, cohort, samples, use_qa, tokenizer, scaler,
                   truncate, tak_repo_path=None):
    """
    Purpose: Build one split's processed dataframes.
    Method:  Phase 1/2 are self-supervised over the FULL trajectory, one view per
             patient -- there is no context window there, so nothing to augment.

             Phase 3 needs the truncated view, and that is where the augmentation
             lives: `DataProcessor` runs once per distinct window in `samples`,
             each pass relabelling `PatientId` to the sample id before the passes
             are stacked. Downstream code groups by that column, so a
             (patient, K) sample is simply a patient as far as the dataset is
             concerned -- no change needed in `EMRDataset`.

    Args:
        abstract_df  (pd.DataFrame): Abstraction rows for these patients.
        cohort       (Cohort):       Shared cohort artefact.
        samples      (pd.DataFrame): This split's samples.
        use_qa       (bool):         QA ablation arm.
        tokenizer    (EMRTokenizer|None): None on the train split (fitted there).
        scaler       (StandardScaler|None): None on the train split.
        truncate     (bool):         Also build the per-sample truncated view.
        tak_repo_path (str|None):    TAK repository to resolve concepts against.

    Returns:
        dict: {'full': df, 'full_ctx': df, 'trunc': df|None, 'trunc_ctx': df|None}
    """
    from kineret.intervene.dataset import DataProcessor

    tak_repo_path = tak_repo_path or paths.TAK_REPO_PATH
    patient_ids = samples["PatientId"].drop_duplicates().to_numpy()
    df = abstract_df[abstract_df["PatientId"].isin(set(patient_ids.tolist()))].copy()

    # Phase 1/2 view: full trajectory, one row-group per patient. Its context is
    # taken at the evaluation window so the self-supervised stages condition on
    # the same static vector the supervised stage will use.
    full_ctx = cohort.context_for_k(C.EVAL_CONTEXT_DAYS, use_qa) \
                     .reindex(patient_ids).reset_index()
    full_proc = DataProcessor(df.copy(), full_ctx, tak_repo_path=tak_repo_path,
                              max_input_days=None, scaler=scaler,
                              checkpoint_path=MCFG.CHECKPOINT_PATH)
    full_df, full_ctx_df = full_proc.run()

    trunc_df = trunc_ctx_df = None
    if truncate:
        fitted = trunc_scaler_of(full_proc, scaler)
        token_blocks, ctx_blocks = [], []
        for k, group in samples.groupby("k", sort=True):
            sample_of = dict(zip(group["PatientId"], group["sample_id"]))
            ctx = cohort.context_for_k(int(k), use_qa) \
                        .reindex(group["PatientId"]).reset_index()
            # Only the admissions that reach this window. Passing the whole
            # split made the processor build placeholder context rows for
            # everyone else, cut them at k days, and then throw them away when
            # the sample-id map came back NaN -- thirteen times over the full
            # table, for nothing.
            window_ids = set(group["PatientId"])
            proc = DataProcessor(
                df[df["PatientId"].isin(window_ids)].copy(), ctx,
                tak_repo_path=tak_repo_path,
                max_input_days=int(k), scaler=fitted,
                checkpoint_path=MCFG.CHECKPOINT_PATH,
                # The cohort already decided who is in the study, against both
                # source tables. Letting the processor re-decide would hand
                # INTERVenE a different test set from the other models.
                enforce_min_duration=False)
            tokens, ctx_out = proc.run()
            tokens["PatientId"] = tokens["PatientId"].map(sample_of)
            ctx_out.index = ctx_out.index.map(sample_of)
            token_blocks.append(tokens[tokens["PatientId"].notna()])
            ctx_blocks.append(ctx_out[ctx_out.index.notna()])
        trunc_df = pd.concat(token_blocks, ignore_index=True)
        trunc_ctx_df = pd.concat(ctx_blocks)
        trunc_ctx_df.index.name = "PatientId"

    return {"full": full_df, "full_ctx": full_ctx_df,
            "trunc": trunc_df, "trunc_ctx": trunc_ctx_df,
            "processor": full_proc}


def trunc_scaler_of(processor, scaler):
    """
    Purpose: Reuse the scaler the untruncated pass just fitted.
    Method:  `DataProcessor` fits and dumps a scaler when handed None; loading
             the dumped copy keeps the truncated pass on identical statistics
             instead of refitting on a different row set.

    Args:
        processor (DataProcessor): The pass that may have fitted a scaler.
        scaler    (StandardScaler|None): Scaler that was passed in, if any.

    Returns:
        StandardScaler: The scaler to use for the truncated pass.
    """
    if scaler is not None:
        return scaler
    import joblib
    return joblib.load(os.path.join(MCFG.CHECKPOINT_PATH, "scaler.pkl"))


def clear_stale_checkpoints(model_config, tokenizer=None, verbose=True):
    """
    Purpose: Drop checkpoints that were written by a differently-shaped run.
    Method:  Every phase checkpoint stores the config it was trained under.
             Compare the shape-determining fields against this run's config and
             delete any phase whose checkpoint disagrees.

             Without this, resuming is a landmine: change `embed_dim` (or flip
             the QA arm, which changes `ctx_dim`; or switch abstraction arm,
             which changes the vocabulary) and re-run, and the phase functions
             happily load the old weights and then die on
             `assert cfg["embed_dim"] == embedder.output_dim` several minutes
             later. Deleting up front turns a confusing crash into one line of
             log and a clean retrain.

    Args:
        model_config (dict):             This run's MODEL_CONFIG.
        tokenizer    (EMRTokenizer|None): Current tokenizer, for the vocab check.
        verbose      (bool):             Print what was cleared.

    Returns:
        list[str]: Phase names that were cleared.
    """
    say = print if verbose else (lambda *a, **kw: None)
    expected = {k: model_config[k] for k in ("embed_dim", "time2vec_dim", "ctx_dim")
                if k in model_config}
    vocab_size = len(tokenizer.token2id) if tokenizer is not None else None

    cleared = []
    for phase, best_path in (("phase1", MCFG.PHASE1_CHECKPOINT),
                             ("phase2", MCFG.PHASE2_CHECKPOINT),
                             ("phase3", MCFG.PHASE3_CHECKPOINT)):
        phase_dir = os.path.dirname(best_path)
        if not os.path.isdir(phase_dir):
            continue
        stale = False
        for name in ("ckpt_last.pt", os.path.basename(best_path)):
            path = os.path.join(phase_dir, name)
            if not os.path.exists(path):
                continue
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                stale = True     # unreadable is as good as incompatible
                break
            cfg = ckpt.get("config") or {}
            if any(k in cfg and cfg[k] != v for k, v in expected.items()):
                stale = True
                break
            if vocab_size is not None:
                saved_vocab = ckpt.get("vocab_size") or cfg.get("vocab_size")
                if saved_vocab is not None and saved_vocab != vocab_size:
                    stale = True
                    break
        if stale:
            for name in os.listdir(phase_dir):
                if name.endswith(".pt"):
                    os.remove(os.path.join(phase_dir, name))
            cleared.append(phase)

    if cleared:
        say(f"[intervene/train] Cleared stale checkpoints for {cleared} -- they "
            f"were written under a different model shape or vocabulary.")
    return cleared


def run(context_days=None, use_qa=False, cohort: Cohort = None, output_root=None,
        seed=None, resume=True, bootstrap_resamples=2000,
        model_overrides=None, training_overrides=None, checkpoint_root=None,
        abstraction="kb", device=None, **_ignored):
    """
    Purpose: Train and score INTERVenE-Enc for one (K, QA-arm) cell.
    Method:  Configure the ported package for this cell, process the Mediator
             output per split, run Phases 1-3, then a single held-out inference
             pass scored by the shared scorer.

    Args:
        context_days        (int):  K -- context window in days.
        use_qa              (bool): QA ablation arm.
        cohort              (Cohort|None): Shared cohort; loaded when None.
        output_root         (str|None): Defaults to `outputs/`.
        seed                (int|None): RNG seed; defaults to the shared seed.
        resume              (bool): Resume each phase from `ckpt_last` if present.
        bootstrap_resamples (int):  Resamples for the final CIs.
        model_overrides     (dict|None): MODEL_CONFIG overrides.
        training_overrides  (dict|None): TRAINING_SETTINGS overrides.
        checkpoint_root     (str|None): Checkpoint root; defaults to
                            `checkpoints/`. The smoke test passes its own so a
                            toy-width run cannot collide with a real one.
        device              (str|None): Torch device string. INTERVenE's phase
                            functions auto-select CUDA; passing `cuda:N` here
                            pins which GPU they land on.
        abstraction         (str):  Which interval source to feed the encoder.
                            'kb'  -- the Mediator's knowledge-based abstractions.
                            'std' -- distribution-derived sigma bins over the same
                                     raw measurements, built by
                                     `kineret.abstraction`. The two arms differ in
                                     exactly one thing: where the intervals came
                                     from.

    Returns:
        dict: {'output_dir': str, 'scores': dict}
    """
    if abstraction not in C.ABSTRACTION_SOURCES:
        raise ValueError(f"abstraction must be one of {C.ABSTRACTION_SOURCES}, "
                         f"got {abstraction!r}")

    # The ported phase functions each resolve their own device as
    # `cuda if available else cpu`, so an explicit `cuda:N` is honoured by
    # making N the process default rather than by threading a device argument
    # through three training loops.
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        if ":" in str(device):
            torch.cuda.set_device(int(str(device).split(":")[1]))
    cohort = cohort or Cohort.load()
    seed = C.SEED if seed is None else seed

    # ── configure the ported package for this cell ───────────────────────
    DCFG.configure(C.EVAL_CONTEXT_DAYS, use_qa, outcome_names=cohort.outcome_names)
    ckpt_root = checkpoint_root or paths.CHECKPOINT_DIR
    if abstraction != "kb":
        # A std-arm tokenizer has a different vocabulary, so its Phase-1/2
        # checkpoints must never be resumed by the KB arm.
        ckpt_root = os.path.join(ckpt_root, abstraction)
    MCFG.configure_checkpoints(C.EVAL_CONTEXT_DAYS, use_qa, root=ckpt_root)
    # Drop whatever the PREVIOUS run left in these dicts before this run
    # configures them -- see `reset_to_defaults`. Without it the wiring check's
    # toy hyperparameters silently become the study's. Must precede
    # `configure_windows`, which writes the horizon keys into TRAINING_SETTINGS.
    MCFG.reset_to_defaults()
    MCFG.configure_windows(C.EVAL_CONTEXT_DAYS)
    if model_overrides:
        MCFG.MODEL_CONFIG.update(model_overrides)
    if training_overrides:
        MCFG.TRAINING_SETTINGS.update(training_overrides)

    # Imported after configure() so `import *` picks up the bound values.
    from kineret.intervene.dataset import (
        EMRDataset, EMRTokenizer, collate_emr, get_dataloader,
    )
    from kineret.intervene.embedder import EMREmbedding, train_embedder
    from kineret.intervene.inference import predict
    from kineret.intervene.transformer import (
        InterveneEncoder, finetune_transformer, pretrain_transformer,
    )

    output_root = output_root or paths.OUTPUT_DIR
    model_key = "intervene_enc" if abstraction == "kb" else f"intervene_enc_{abstraction}"
    run_dir = os.path.join(output_root, model_key, "qa" if use_qa else "noqa")
    os.makedirs(run_dir, exist_ok=True)

    samples = cohort.samples()
    train_s = samples[samples["split"] == "train"].reset_index(drop=True)
    val_s = samples[samples["split"] == "val"].reset_index(drop=True)
    test_s = samples[samples["split"] == "test"].reset_index(drop=True)
    print(f"[intervene/train] qa={use_qa} abstraction={abstraction}: "
          f"{len(train_s)} train samples (augmented over "
          f"K={list(C.TRAIN_CONTEXT_DAYS)}) / {len(val_s)} val / "
          f"{len(test_s)} test @ K={C.EVAL_CONTEXT_DAYS}.")

    # ── choose the interval source ───────────────────────────────────────
    if abstraction == "kb":
        abstract = normalise_temporal(load_table(paths.ABSTRACT_FILE))
        # Same canonical target events as every other arm. Without this the KB
        # arm reads the Mediator's own event rows while the raw arms read the
        # reconciled ones -- and after cross-file alignment those differ, so the
        # ladder would be comparing arms that saw different events.
        # kb_events=True keeps the NON-target `*_EVENT` abstractions: this is
        # the knowledge stream, and they are the knowledge.
        abstract = cohort.harmonise_events(
            abstract, patient_ids=cohort.patients["PatientId"],
            kb_events=True, verbose=True)
        tak_repo_path = paths.TAK_REPO_PATH
    else:
        from kineret.abstraction import ensure_std_temporal
        table_path, tak_repo_path = ensure_std_temporal(cohort)
        abstract = normalise_temporal(load_table(table_path))
    abstract = abstract[abstract["PatientId"].isin(set(cohort.patients["PatientId"]))]

    # ── train split: fits the scaler and the tokenizer ───────────────────
    train_parts = _prepare_split(abstract, cohort, train_s, use_qa,
                                 tokenizer=None, scaler=None, truncate=True,
                                 tak_repo_path=tak_repo_path)
    tokenizer = EMRTokenizer.from_processed_df(train_parts["full"])
    tokenizer.save(os.path.join(MCFG.CHECKPOINT_PATH, "tokenizer.pt"))
    scaler = trunc_scaler_of(train_parts["processor"], None)

    val_parts = _prepare_split(abstract, cohort, val_s, use_qa, tokenizer,
                               scaler, truncate=True, tak_repo_path=tak_repo_path)
    test_parts = _prepare_split(abstract, cohort, test_s, use_qa, tokenizer,
                                scaler, truncate=True, tak_repo_path=tak_repo_path)

    MCFG.MODEL_CONFIG["ctx_dim"] = int(train_parts["full_ctx"].shape[1])
    if resume:
        clear_stale_checkpoints(MCFG.MODEL_CONFIG, tokenizer=tokenizer)

    # ── datasets ─────────────────────────────────────────────────────────
    def _make(parts, sample_rows, truncated):
        """
        Build one EMRDataset.

        Truncated (Phase-3) datasets are keyed by SAMPLE and carry the cohort's
        labels for that sample's own window. Untruncated (Phase-1/2) datasets
        are keyed by patient over the full trajectory -- self-supervised, so no
        labels and no augmentation.
        """
        if not truncated:
            return EMRDataset(parts["full"], parts["full_ctx"], tokenizer=tokenizer)
        tokens = parts["trunc"]
        present = set(tokens["PatientId"].unique())
        rows = sample_rows[sample_rows["sample_id"].isin(present)]
        labels, times = _build_label_frames(cohort, rows)
        # `label_source_df` is deliberately omitted: labels are injected from
        # the cohort, so the token-derived fallback in `build_patient_labels`
        # is never reached and replicating the full trajectory per sample would
        # only cost memory.
        return EMRDataset(tokens, parts["trunc_ctx"], tokenizer=tokenizer,
                          patient_labels=labels, patient_gt_time=times)

    p12_train_ds = _make(train_parts, train_s, truncated=False)
    p12_val_ds = _make(val_parts, val_s, truncated=False)
    p3_train_ds = _make(train_parts, train_s, truncated=True)
    p3_val_ds = _make(val_parts, val_s, truncated=True)
    test_ds = _make(test_parts, test_s, truncated=True)

    # Everything the datasets needed is now inside them. What is still bound
    # here is setup scaffolding that stays resident for the whole run:
    #
    #   parts["processor"]  a DataProcessor, and `__init__` does `self.df =
    #                       df.copy()` -- a full second copy of that split's
    #                       abstraction stream, x3 splits. Dead once the scaler
    #                       has been read off it.
    #   abstract            the source table the three splits were cut from.
    #   parts[...]          the extra reference to each token frame; the
    #                       datasets hold their own.
    #
    # On the real cohort that is several GB held through Phase 1 and Phase 2,
    # on a box that was already at 28 of 30 GB.
    for _parts in (train_parts, val_parts, test_parts):
        _parts.clear()
    del train_parts, val_parts, test_parts, abstract
    gc.collect()

    bs = MCFG.TRAINING_SETTINGS["batch_size"]
    nw = MCFG.TRAINING_SETTINGS.get("dataloader_workers", 0)
    p12_train_dl = get_dataloader(p12_train_ds, bs, collate_emr,
                                  bucket_batching=True, num_workers=nw)
    p12_val_dl = get_dataloader(p12_val_ds, bs, collate_emr,
                                bucket_batching=True, num_workers=nw)
    p2_train_dl = get_dataloader(p12_train_ds, bs, collate_emr,
                                 oversample=True, bucket_batching=True,
                                 num_workers=nw)
    p3_train_dl = get_dataloader(p3_train_ds, bs, collate_emr,
                                 bucket_batching=True, num_workers=nw)
    p3_val_dl = get_dataloader(p3_val_ds, bs, collate_emr,
                               bucket_batching=True, num_workers=nw)

    # ── Phase 1: embedder ────────────────────────────────────────────────
    print("\n[intervene/train] === Phase 1: embedder ===")
    embedder = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=MCFG.MODEL_CONFIG["ctx_dim"],
        time2vec_dim=MCFG.MODEL_CONFIG["time2vec_dim"],
        embed_dim=MCFG.MODEL_CONFIG["embed_dim"],
        dropout=MCFG.MODEL_CONFIG["dropout"],
    )
    embedder, _, _ = train_embedder(
        embedder, p12_train_dl, p12_val_dl, resume=resume,
        checkpoint_path=MCFG.PHASE1_CHECKPOINT,
        training_settings=MCFG.TRAINING_SETTINGS,
    )

    # `train_embedder` returns the RESUMED embedder when a `ckpt_last` exists,
    # and that checkpoint carries its own widths. If they disagree with the
    # current MODEL_CONFIG the checkpoint is from a differently-shaped run --
    # a changed embed_dim, or a QA arm that changed ctx_dim. Retrain rather than
    # dying three lines later inside the encoder's dimension assert.
    if embedder.output_dim != MCFG.MODEL_CONFIG["embed_dim"]:
        print(f"[intervene/train] Phase-1 checkpoint has embed_dim="
              f"{embedder.output_dim} but this run wants "
              f"{MCFG.MODEL_CONFIG['embed_dim']} -- discarding it and "
              f"retraining Phase 1 from scratch.")
        embedder = EMREmbedding(
            tokenizer=tokenizer,
            ctx_dim=MCFG.MODEL_CONFIG["ctx_dim"],
            time2vec_dim=MCFG.MODEL_CONFIG["time2vec_dim"],
            embed_dim=MCFG.MODEL_CONFIG["embed_dim"],
            dropout=MCFG.MODEL_CONFIG["dropout"],
        )
        embedder, _, _ = train_embedder(
            embedder, p12_train_dl, p12_val_dl, resume=False,
            checkpoint_path=MCFG.PHASE1_CHECKPOINT,
            training_settings=MCFG.TRAINING_SETTINGS,
        )

    # ── Phase 2: MLM pretraining ─────────────────────────────────────────
    print("\n[intervene/train] === Phase 2: MLM pretraining ===")
    model = InterveneEncoder(MCFG.MODEL_CONFIG, embedder=embedder)
    model, _, _ = pretrain_transformer(
        model, p2_train_dl, p12_val_dl, resume=resume,
        checkpoint_path=MCFG.PHASE2_CHECKPOINT,
        training_settings=MCFG.TRAINING_SETTINGS,
    )

    # ── Phase 3: risk + time fine-tune ───────────────────────────────────
    print("\n[intervene/train] === Phase 3: risk + time fine-tune ===")
    # Tell build_patient_labels how the injected label matrix is laid out. It
    # goes on the TOKENIZER, not the model: `finetune_transformer` rebuilds the
    # model object when it resumes from `ckpt_last`, and an attribute set on the
    # old instance would vanish with it. The tokenizer rides through on the
    # embedder, so it survives the reload. Set on both for directness.
    tokenizer.injected_label_columns = list(p3_train_ds.label_columns)
    model.injected_label_columns = list(p3_train_ds.label_columns)
    model, _, _ = finetune_transformer(
        model, p3_train_dl, p3_val_dl, resume=resume,
        checkpoint_path=MCFG.PHASE3_CHECKPOINT,
        training_settings=MCFG.TRAINING_SETTINGS,
    )

    # ── held-out inference ───────────────────────────────────────────────
    print("\n[intervene/train] === Test inference ===")
    if os.path.exists(MCFG.PHASE3_CHECKPOINT):
        model, *_ = InterveneEncoder.load(MCFG.PHASE3_CHECKPOINT, embedder=embedder,
                                          attach_task_heads=True)
    model.eval()
    preds = predict(model, test_ds)

    # Test samples are one-per-patient at the evaluation window, so the run's
    # rows are patients. Reindex to the cohort's test set: if a degenerate
    # trajectory were dropped during processing the test set would quietly
    # differ from every other model's, so abstain (P=0.5) on any such row
    # rather than report a smaller cohort.
    test_sample_ids = list(test_s["sample_id"])
    test_pids = list(test_s["PatientId"])
    dropped = [sid for sid in test_sample_ids if sid not in set(test_ds.patient_ids)]
    if dropped:
        print(f"[intervene/train] WARNING: {len(dropped)} test sample(s) were "
              f"dropped during processing and will be scored as abstentions "
              f"(P=0.5) to keep the test set aligned with the other models: "
              f"{dropped[:10]}{' ...' if len(dropped) > 10 else ''}")
    preds = preds.reindex(test_sample_ids)

    outcomes = list(cohort.outcome_names)
    label_block = (cohort.labels_for_samples(test_s)
                         .set_index("sample_id").reindex(test_sample_ids))

    labels = label_block[outcomes].to_numpy(dtype=float)
    # A missing P_ column means the outcome never entered INTERVenE's risk head
    # (RELEASE is dropped by design). Emit 0.5 -- an explicit abstention -- so
    # the column stays present and comparable rather than silently vanishing.
    probs = np.column_stack([
        preds[f"P_{name}"].to_numpy(dtype=float) if f"P_{name}" in preds.columns
        else np.full(len(test_sample_ids), 0.5)
        for name in outcomes
    ])
    probs = np.where(np.isfinite(probs), probs, 0.5)
    missing = [n for n in outcomes if f"P_{n}" not in preds.columns]
    if missing:
        print(f"[intervene/train] No risk head for {missing} -- emitting 0.5.")

    los_true = label_block["length_of_stay_hours"].to_numpy(dtype=float)
    los_pred = (preds[f"T_{DCFG.RELEASE_TOKEN}"].to_numpy(dtype=float)
                if f"T_{DCFG.RELEASE_TOKEN}" in preds.columns else None)

    # Onset timing, in the shared schema.
    time_pred = np.column_stack([
        preds[f"T_{name}"].to_numpy(dtype=float) if f"T_{name}" in preds.columns
        else np.full(len(test_sample_ids), np.nan)
        for name in outcomes
    ])
    time_true = np.column_stack([
        label_block[f"{name}__first_hour"].to_numpy(dtype=float) for name in outcomes
    ])
    time_true = np.where(np.isfinite(time_true), time_true, np.nan)

    write_predictions(run_dir, test_pids, labels, probs, outcomes,
                      los_true=los_true, los_pred=los_pred,
                      time_true=time_true, time_pred=time_pred)
    preds.to_csv(os.path.join(run_dir, "intervene_raw_predictions.csv"))

    meta = {
        "model": model_key,
        "abstraction": abstraction,
        "context_days": C.EVAL_CONTEXT_DAYS,
        "train_context_days": list(C.TRAIN_CONTEXT_DAYS),
        "use_qa": use_qa,
        "seed": seed,
        "outcome_names": outcomes,
        "label_window_hours": [C.EVAL_CONTEXT_DAYS * 24.0, C.HORIZON_END_DAYS * 24.0],
        "n_train_samples": int(len(train_s)), "n_val": int(len(val_s)),
        "n_test": int(len(test_pids)),
        "model_config": {k: v for k, v in MCFG.MODEL_CONFIG.items()},
        "head_outcomes": list(model.outcome_names),
        "checkpoint_dir": MCFG.CHECKPOINT_PATH,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    scores = score_run(run_dir, n_resamples=bootstrap_resamples, seed=seed)
    return {"output_dir": run_dir, "scores": scores}
