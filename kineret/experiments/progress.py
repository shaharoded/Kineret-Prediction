"""Compact live training status; full model output stays in a per-model log."""
from contextlib import contextmanager, ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
import importlib
import sys
import time
from unittest.mock import patch

from tqdm.std import tqdm as TextTqdm


class LiveStatus:
    def __init__(self, title, log, interval=5):
        self.title, self.log, self.interval = title, log, interval
        self.bars, self.last = {}, 0.
        self.console = sys.stdout
        self.handle = None
        try:
            from IPython import get_ipython
            if getattr(get_ipython(), "kernel", None) is not None:
                from IPython.display import display
                self.handle = display(title, display_id=True)
        except ImportError:
            pass

    def render(self, force=False):
        now = time.monotonic()
        if not force and now-self.last < self.interval:
            return
        self.last = now
        lines = [self.title]
        for bar in self.bars.values():
            total = bar.total
            label = bar.desc or "Progress"
            if label.startswith("Phase-"):
                start = getattr(bar.iterable, "start", 1)
                label += f" epoch {min(start+bar.n, start+total-1)}/{start+total-1}" if total else ""
            progress = f"{bar.n}/{total}" if total is not None else str(bar.n)
            elapsed = max(0., time.time()-getattr(bar, "start_t", time.time()))
            rate = bar.n/elapsed if elapsed and bar.n else 0.
            eta = f" | ETA {(total-bar.n)/rate/60:.1f} min" if rate and total else ""
            lines.append(f"{label}: {progress} | elapsed {elapsed/60:.1f} min{eta}")
        message = "\n".join(lines)
        self.log.write(message+"\n")
        self.log.flush()
        if self.handle is not None:
            self.handle.update(message)
        else:
            self.console.write("\r"+" | ".join(lines)+"    ")
            self.console.flush()


@contextmanager
def compact_training(title, log_path, interval=5):
    """Sequential training only; restore streams/progress hooks even on interrupt."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        status = LiveStatus(title, log, interval)

        class CompactTqdm(TextTqdm):
            def __init__(self, *args, **kwargs):
                kwargs.update(file=log, mininterval=interval, leave=False)
                super().__init__(*args, **kwargs)

            def display(self, *args, **kwargs):
                status.bars[id(self)] = self
                status.render()

            def close(self):
                if not hasattr(self, "disable") or self.disable:
                    return
                status.bars.pop(id(self), None)
                super().close()
                status.bars.pop(id(self), None)
                status.render()

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(log))
            stack.enter_context(redirect_stderr(log))
            modules = [importlib.import_module(name) for name in (
                "kineret.strats.train", "kineret.strats.evaluator",
                "kineret.intervene.embedder", "kineret.intervene.transformer")]
            utils = importlib.import_module("kineret.intervene.utils")
            original_tee = utils._active_tee
            for module in modules:
                stack.enter_context(patch.object(module, "tqdm", CompactTqdm))
                if hasattr(module, "_QUIET_EPOCHS"):
                    stack.enter_context(patch.object(module, "_QUIET_EPOCHS", True))
            status.render(force=True)
            try:
                yield
            except BaseException:
                status.title = title+" | interrupted or failed; see "+str(log_path)
                raise
            else:
                status.title = title+" | complete"
            finally:
                status.bars.clear()
                status.render(force=True)
                if utils._active_tee is not original_tee:
                    utils._active_tee.close()
                    utils._active_tee = original_tee
        status.console.write("\n")
