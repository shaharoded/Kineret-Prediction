"""
schedulers.py
=============

Unified auxiliary loss weighting scheduler for Phase-1 (embedding) and Phase-2 (transformer) training.

LRScheduleController:
    - Wraps Phase-2 OneCycleLR construction and step updates.
    - Uses `phase2_scheduler["warmup_epochs"]` when provided, otherwise
        falls back to `phase2_scheduler["main_only_epochs"]` for LR warmup.
    - Exposes `update`, `state_dict`, and `load_state_dict` for training/resume.

LambdaScheduleController:
  - Accepts a phase-specific schedule config dict.
  - Defines auxiliary tasks and their curriculum via an `order` list-of-lists:
      - Each inner list is a stage of aux tasks that activate together.
      - Stage 0 activates after a main-loss-only warmup period (main_only_epochs).
      - Subsequent stages are unlocked dynamically when the total weighted validation
        loss plateaus — but only after the current stage's ramp has completed.
  - Frozen-fraction calibration: lambda_max = fraction_cap * tr_main / tr_aux
    (computed from TRAINING losses once, then fixed).
  - Linear ramp from 0 to lambda_max over ramp_epochs (ramp_epochs=0 means immediate).
  - Warmup tracking: reports the epoch after which early stopping may begin.

Config expected shape (phase-specific dict):
    {
        "aux_fraction_caps":  {name: fraction, ...},  # required; every aux name must be present
        "order":              [[name, ...], [name, ...], ...],
        "ramp_epochs":        {name: int, ...},
        "main_only_epochs":    int,                    # epochs of main-loss-only training before aux activates
                                                       # (Phase-1 main = outcome BCE; Phase-2 main = MLM CE).
        # Multi-stage only (len(order) > 1):
        "plateau_min_delta":  float,
        "plateau_patience":   int | [int, ...],       # one per stage transition
    }

update() call convention:
    controller.update(
        epoch     = epoch,
        vl_total  = vl_loss,      # total weighted validation loss  → plateau detection
        tr_main   = tr_bce,       # training BCE                    → calibration denominator
        **{name: tr_raw_loss},    # training raw aux losses by name → calibration numerator
    )
"""

import torch


def linear_schedule(epoch: int, start_epoch: int, end_epoch: int, max_val: float) -> float:
    """
    Linearly ramp from 0 to `max_val` over [start_epoch, end_epoch].
    If end_epoch <= start_epoch, immediately returns max_val at start_epoch (no ramp).
    """
    if epoch < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return max_val
    progress = min(max((epoch - start_epoch) / float(end_epoch - start_epoch), 0.0), 1.0)
    return max_val * progress


class LRScheduleController:
    """Controller wrapper for Phase-2 LR scheduling.

    Encapsulates a OneCycleLR instance and exposes a minimal interface used by
    training loops and checkpointing (`update`, `state_dict`, `load_state_dict`).
    """

    def __init__(self, optimizer, training_settings, train_dl):
        """Configure the phase-2 learning-rate scheduler controller.

        Builds and owns a `torch.optim.lr_scheduler.OneCycleLR` instance used
        during phase-2 training. The warmup portion is derived from
        `training_settings["phase2_scheduler"]["warmup_epochs"]` or `training_settings["phase2_scheduler"]["main_only_epochs"]`
        and converted to OneCycle's
        `pct_start` fraction.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance whose param
                    groups are already configured for transformer training.
            training_settings (dict): Training hyperparameters containing at
                    least:
                    - `phase2_n_epochs`
                    - `phase2_scheduler`
            train_dl (DataLoader): Training dataloader used to infer
                    `steps_per_epoch` via `len(train_dl)`.

        Notes:
            - Expected optimizer param-group order is
                `[decay, no_decay, embedder]`.
            - `max_lr` is mapped to those groups as
                `[base_lr, base_lr, base_lr * 0.1]`.
            - Auxiliary-loss lambda scheduling is independent and handled by
                `LambdaScheduleController`.
        """
        total_epochs = training_settings["phase2_n_epochs"]
        lr_warmup_epochs = training_settings["phase2_scheduler"].get("warmup_epochs", 0) or training_settings["phase2_scheduler"].get("main_only_epochs", 0)
        pct = max(1e-6, min(0.9, lr_warmup_epochs / float(total_epochs)))

        base_lr = training_settings["phase2_learning_rate"]
        # Keep max_lr aligned with optimizer param groups: decay, no_decay, embedder(0.1x)
        max_lrs = [base_lr, base_lr, base_lr * 0.1]

        self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=total_epochs,
            steps_per_epoch=len(train_dl),
            pct_start=pct,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=10,
            final_div_factor=10,
        )

    def step(self):
        """Advance LR scheduler by one optimizer step."""
        self._scheduler.step()

    def get_last_lr(self):
        return self._scheduler.get_last_lr()

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)


class LambdaScheduleController:
    """
    Unified scheduler for auxiliary loss weighting.

    Phase behaviour is driven purely by the `order` list:
      - Single stage  → Phase-1 style: all aux tasks activate after main_only_epochs, no plateau gating.
                        Main loss = outcome BCE.
      - Multi-stage   → Phase-2 style: stage 0 activates after main_only_epochs; later stages unlock
                        on plateau detection of vl_total — but only after the current stage's ramp ends.
                        Main loss = MLM cross-entropy.

    Usage (once per epoch, after both train and val epochs):
        msgs = controller.update(epoch, vl_total=vl_loss, tr_main=tr_bce, **tr_aux_losses)
        lambdas = controller.get_lambdas(epoch)   # call per batch during training
    """

    def __init__(self, schedule_config: dict, start_epoch: int = 0):
        """
        Parameters
        ----------
        schedule_config : dict
            Phase-specific scheduler config (see module docstring for expected keys).
        start_epoch : int
            Current training epoch (for checkpoint resume).
        """
        self._cfg = schedule_config
        self.start_epoch = int(start_epoch)
        self._min_aux_loss = 1e-8
        self._max_lambda_clamp = 10.0
        # Per-aux λ_max override. A trivially-solved / tiny-magnitude aux (e.g.
        # t_local, raw MSE ~5e-4) wants λ ≫ 10 to reach its fraction_cap share of
        # the main loss; the global 10 clamp pins it invisible. Allow specific
        # auxes a higher ceiling via schedule_config["max_lambda"] = {name: cap}.
        self._max_lambda_overrides = dict(schedule_config.get("max_lambda", {}))

        caps = schedule_config["aux_fraction_caps"]
        ramp_cfg = schedule_config.get("ramp_epochs", {})
        order = schedule_config.get("order", [])
        self._order = order

        main_only = max(1, int(schedule_config.get("main_only_epochs", 1)))

        # Register all auxiliaries.
        # Stage 0 starts after main_only_epochs; later stages start as None (unlocked later).
        # Raises KeyError immediately if any aux name is missing from aux_fraction_caps.
        self._auxiliaries = {}
        for stage_idx, stage_auxi in enumerate(order):
            if stage_idx == 0:
                s_epoch = self.start_epoch + main_only
            else:
                s_epoch = None
            for name in stage_auxi:
                if name not in caps:
                    raise KeyError(
                        f"aux_fraction_caps is missing an entry for '{name}'. "
                        f"Add it explicitly — no silent defaults."
                    )
                self._register_aux(
                    name=name,
                    start_epoch=s_epoch,
                    ramp_epochs=max(0, int(ramp_cfg.get(name, 0))),
                    fraction=caps[name],
                )

        # Multi-stage: plateau-based curriculum
        n_stages = len(order)
        self._has_dynamic = n_stages > 1

        if self._has_dynamic:
            patience_cfg = schedule_config.get("plateau_patience", 3)
            if isinstance(patience_cfg, int):
                patience_cfg = [patience_cfg] * (n_stages - 1)

            self.plateau_min_delta = float(schedule_config.get("plateau_min_delta", 1e-4))
            self._plateau_patience = patience_cfg

            self._current_stage = 0
            self._stage_best = float("inf")
            self._stage_bad_epochs = 0

        # Warmup completion gate used by training loops to start early-stop counting.
        # Single-stage schedules have a known warmup completion at init time;
        # multi-stage schedules resolve this only after the final stage unlocks.
        if self._has_dynamic:
            self._warmup_complete_epoch = None
        else:
            if len(order) == 0:
                self._warmup_complete_epoch = self.start_epoch
            else:
                # +1 accounts for the calibration delay. update() runs at end-of-epoch,
                # so at epoch == start_epoch, get_lambdas() still returns 0 (lambda_max
                # not yet set). The aux only becomes active from start_epoch+1 onwards.
                # Without this offset the LAST bce-only epoch's val_loss locks in as
                # 'best' before the aux ever contributes to training — early-stop then
                # counts down and kills the run before the aux can compete.
                # (Multi-stage Phase-2 gets this for free via unlock_epoch = epoch + 1.)
                self._warmup_complete_epoch = max(self._ramp_end(n) for n in order[0]) + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_aux(self, name: str, start_epoch, ramp_epochs: int, fraction: float):
        self._auxiliaries[name] = {
            "name": name,
            "start_epoch": start_epoch,
            "ramp_epochs": max(0, int(ramp_epochs)),
            "fraction": float(fraction),
            "lambda_max": None,
            "anchor_main_loss": None,
            "anchor_aux_loss": None,
        }

    def _ramp_end(self, name: str) -> int:
        """Epoch at which the named aux task reaches its full lambda_max.

        ramp_epochs=0 → immediate (lambda at full value from start_epoch).
        ramp_epochs=N → linear ramp; reaches max at start_epoch + N.
        """
        spec = self._auxiliaries[name]
        s = spec["start_epoch"]
        if s is None:
            return float("inf")
        return s + spec["ramp_epochs"]

    @staticmethod
    def _check_plateau(metric_val, best_val, bad_epochs, min_delta, patience):
        """Returns (new_best, new_bad_epochs, is_plateau). min_delta is relative (e.g. 1e-4 = 0.01%)."""
        if metric_val < best_val * (1.0 - min_delta):
            return metric_val, 0, False
        bad_epochs += 1
        return best_val, bad_epochs, bad_epochs >= patience

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_dynamic(self) -> bool:
        """True when the scheduler has more than one stage (plateau-gated curriculum)."""
        return self._has_dynamic

    def update(self, epoch: int, vl_total: float, tr_main: float, **tr_aux_losses) -> list:
        """
        Calibrate auxiliaries and advance dynamic stage transitions.

        Parameters
        ----------
        epoch : int
            Current epoch.
        vl_total : float
            Total weighted validation loss (BCE + lambda*aux + ...). Used for plateau detection.
        tr_main : float
            Training BCE loss. Used as the calibration denominator.
        **tr_aux_losses : float
            Training raw aux losses, keyed by plain aux name (e.g. mlm=0.4, dt=0.1).
            Used as the calibration numerator.

        Returns
        -------
        list[str]
            Log messages for calibration events and stage transitions.
        """
        messages = []

        # Step 1: Check stage transitions first — so newly unlocked aux can calibrate
        # in the same call. Plateau check is skipped while current stage is still ramping.
        if self._has_dynamic:
            messages.extend(self._check_stage_transitions(epoch, vl_total))

        # Step 2: Calibrate each auxiliary once using training losses.
        # Pre-calibrate one epoch before start_epoch so the first active epoch already
        # has a known lambda_max (get_lambdas still returns 0 when epoch < start_epoch).
        for name, spec in self._auxiliaries.items():
            if spec["start_epoch"] is None or epoch < spec["start_epoch"] - 1:
                continue  # Not yet time to calibrate
            if spec["lambda_max"] is not None:
                continue  # Already calibrated
            if name not in tr_aux_losses:
                continue  # Loss not provided this call
            tr_aux = float(tr_aux_losses[name])
            if tr_aux > self._min_aux_loss:
                lam = (spec["fraction"] * tr_main) / max(tr_aux, self._min_aux_loss)
                _clamp = self._max_lambda_overrides.get(name, self._max_lambda_clamp)
                spec["lambda_max"] = min(lam, _clamp)
                spec["anchor_main_loss"] = tr_main
                spec["anchor_aux_loss"] = tr_aux
                messages.append(
                    f"[Scheduler]: {name} calibrated at epoch {epoch}, "
                    f"λ_max={spec['lambda_max']:.4f} "
                    f"(tr_main={tr_main:.4f}, tr_aux={tr_aux:.4f})"
                )

        return messages

    def _check_stage_transitions(self, epoch: int, vl_total: float) -> list:
        """Check whether the next stage should be unlocked based on plateau detection."""
        messages = []

        if self._current_stage >= len(self._order) - 1:
            return messages

        transition_idx = self._current_stage
        next_stage_idx = self._current_stage + 1
        next_stage_auxi = self._order[next_stage_idx]

        # Resume edge case: next stage already unlocked externally
        if all(self._auxiliaries[n]["start_epoch"] is not None for n in next_stage_auxi):
            self._current_stage = next_stage_idx
            return messages

        # Don't start plateau check until the current stage's ramp has completed.
        # This prevents the growing lambda during ramp from triggering false plateaus.
        current_stage_auxi = self._order[self._current_stage]
        stage_ramp_end = max(self._ramp_end(n) for n in current_stage_auxi)
        if epoch < stage_ramp_end:
            return messages

        self._stage_best, self._stage_bad_epochs, plateau = self._check_plateau(
            vl_total, self._stage_best, self._stage_bad_epochs,
            self.plateau_min_delta, self._plateau_patience[transition_idx],
        )

        if plateau:
            unlock_epoch = epoch + 1
            for name in next_stage_auxi:
                self._auxiliaries[name]["start_epoch"] = unlock_epoch

            messages.append(
                f"[Scheduler][Dynamic]: Stage {next_stage_idx} "
                f"({', '.join(next_stage_auxi)}) unlocked at epoch {unlock_epoch}"
            )

            if next_stage_idx == len(self._order) - 1:
                max_ramp = max(self._auxiliaries[n]["ramp_epochs"] for n in next_stage_auxi)
                self._warmup_complete_epoch = unlock_epoch + max_ramp
                messages.append(
                    f"[Scheduler]: Warmup completes at epoch {self._warmup_complete_epoch}"
                )

            self._current_stage = next_stage_idx
            self._stage_best = float("inf")
            self._stage_bad_epochs = 0

        return messages

    def get_lambdas(self, epoch: int) -> dict:
        """
        Return current lambda values for all registered auxiliaries.

        Returns 0.0 for aux tasks not yet active (before main_only_epochs or before stage unlock)
        or not yet calibrated.
        """
        lambdas = {}
        for name, spec in self._auxiliaries.items():
            if spec["start_epoch"] is None or spec["lambda_max"] is None:
                lambdas[name] = 0.0
            else:
                start = spec["start_epoch"]
                end = self._ramp_end(name)
                lambdas[name] = linear_schedule(epoch, start, end, spec["lambda_max"])
        return lambdas

    def current_warmup_end_epoch(self):
        """
        Return the epoch after which early-stopping may begin counting.

        Returns
        -------
        int | float | None
            - Multi-stage: float('inf') until last stage unlocked, then concrete epoch.
            - Single-stage: concrete epoch (end of stage-0 ramp).
        """
        if self._warmup_complete_epoch is None:
            return float("inf")
        return self._warmup_complete_epoch

    def state_dict(self) -> dict:
        """
        Return serialisable scheduler state for checkpoint saving.
        Covers all mutable state so that resume is exact.
        """
        aux_state = {
            name: {
                "start_epoch": spec["start_epoch"],
                "lambda_max": spec["lambda_max"],
                "anchor_main_loss": spec["anchor_main_loss"],
                "anchor_aux_loss": spec["anchor_aux_loss"],
            }
            for name, spec in self._auxiliaries.items()
        }
        state = {"auxiliaries": aux_state, "warmup_complete_epoch": self._warmup_complete_epoch}
        if self._has_dynamic:
            state.update({
                "current_stage": self._current_stage,
                "stage_best": self._stage_best,
                "stage_bad_epochs": self._stage_bad_epochs,
            })
        return state

    def load_state_dict(self, state: dict):
        """Restore scheduler state from a checkpoint dict (produced by state_dict())."""
        for name, saved in state["auxiliaries"].items():
            if name in self._auxiliaries:
                self._auxiliaries[name].update(saved)
        self._warmup_complete_epoch = state.get("warmup_complete_epoch")
        if self._has_dynamic:
            self._current_stage = state.get("current_stage", 0)
            self._stage_best = state.get("stage_best", float("inf"))
            self._stage_bad_epochs = state.get("stage_bad_epochs", 0)

    def status_line(self, epoch: int) -> str:
        """Human-readable status line for logging."""
        parts = []
        lambdas = self.get_lambdas(epoch)
        for name in sorted(self._auxiliaries.keys()):
            spec = self._auxiliaries[name]
            lam = lambdas[name]
            if spec["lambda_max"] is None:
                parts.append(f"{name}:λ={lam:.4f}(pending)")
            else:
                parts.append(f"{name}:λ={lam:.4f}/λ_max={spec['lambda_max']:.4f}")
        return f"[Scheduler] epoch={epoch} | {' | '.join(parts)}"
