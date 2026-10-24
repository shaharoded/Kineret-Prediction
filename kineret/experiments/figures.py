"""Small, exported figures from the extension tables; no model execution."""
from pathlib import Path
import pandas as pd
import numpy as np


def export_figures(directory):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory/f"{name}.png", dpi=180)
        fig.savefig(directory/f"{name}.pdf")
        plt.close(fig)
    metrics = directory/"metrics.csv"
    if metrics.exists() and metrics.stat().st_size > 2:
        frame = pd.read_csv(metrics)
        data = frame[(frame.outcome == "weighted") & (frame.metric == "auprc")]
        if len(data):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(data.arm, data.estimate, color="#4c72b0")
            ax.errorbar(data.estimate, np.arange(len(data)),
                        xerr=np.maximum(0, np.vstack([data.estimate-data.lo, data.hi-data.estimate])),
                        fmt="none", color="black", capsize=3)
            ax.set_xlabel("Support-weighted AUPRC · person-clustered 95% CI")
            save(fig, "ladder_auprc")
    path = directory/"reliability.csv"
    if path.exists():
        frame = pd.read_csv(path)
        outcomes = list(frame.outcome.unique())
        fig, axes = plt.subplots((len(outcomes)+3)//4, 4, figsize=(14, 3.3*((len(outcomes)+3)//4)), squeeze=False)
        for ax, outcome in zip(axes.flat, outcomes):
            ax.plot([0, 1], [0, 1], "k--", lw=.7)
            focus = frame[(frame.outcome == outcome) & frame.arm.isin(["intervene_kb", "intervene_kb_qa"])]
            for (arm, variant), group in focus.groupby(["arm", "variant"]):
                ax.plot(group.mean_pred, group.mean_obs, marker="o", ms=3, label=f"{arm}/{variant}")
            ax.set_title(outcome, fontsize=8); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        for ax in list(axes.flat)[len(outcomes):]: ax.axis("off")
        axes.flat[0].legend(fontsize=5); save(fig, "reliability")


def window_figure(summary, directory):
    import matplotlib.pyplot as plt
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory/"window_metrics.csv", index=False)
    data = summary[(summary.outcome == "weighted") & (summary.metric == "auprc")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, group in data.groupby("arm"):
        group = group.sort_values("k")
        ax.plot(group.k, group.estimate, "o-", label=arm)
        ax.fill_between(group.k, group.lo, group.hi, alpha=.1)
    ax.set_xlabel("Observed days K (horizon end stays day 14)")
    ax.set_ylabel("Support-weighted AUPRC"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(directory/"window_sensitivity.png", dpi=180)
    fig.savefig(directory/"window_sensitivity.pdf"); plt.close(fig)
