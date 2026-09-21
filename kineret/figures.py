"""
Publication figures for the study.

Every figure here is built to be dropped into the paper: print resolution, a
vector PDF beside each PNG, and the numbers behind it written out as CSV.

**Colour is assigned by the job it does, not by taste.**

* The ladder is an *ordered* set of arms and the point of the figure is the
  confidence intervals, so it is a dot-and-interval plot: one series, one hue,
  ordering carried by position. Colouring seven ordered rungs with seven hues
  would burn the only free channel on information the y-axis already shows.
* The per-target breakdown compares *input representations*, which is genuine
  categorical identity and the thing the study varies, so it takes four
  categorical slots validated on the all-pairs colour-vision test. The two
  INTERVenE arms get different hues on purpose: sigma-bins vs KB is the central
  claim, and giving them one hue would leave the comparison the figure exists to
  make encoded by nothing. The QA axis rides on texture instead, so it never
  competes with the representation axis for the same channel.
* The QA ablation is a *polarity* question (does it help or hurt), so it takes
  the diverging pair with a neutral zero.
* The support heatmap is *magnitude*, so it takes a single-hue light-to-dark
  ramp.

Palette values are the validated defaults; the categorical trio and the
diverging pair both pass the colour-vision and contrast checks. Three slots sit
below 3:1 against the surface, which obliges visible labels -- so every figure
carries direct value labels and writes its table alongside.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kineret.benchmark import (
    ARM_ORDER, arm_label, load_results, per_outcome_table, save_figure,
    save_table,
)
from kineret.config import data_config as C

# --- Palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK_SOFT = "#52514e"     # secondary text
GRID = "#e6e5e1"         # recessive grid

SERIES = "#2a78d6"       # categorical slot 1 -- the single-series hue

# Colour follows the INPUT REPRESENTATION, which is the thing the study varies.
# The two INTERVenE arms are NOT one family for this purpose: sigma-bins vs KB
# is the central claim, so giving them one hue would leave the comparison the
# figure exists to make encoded by nothing at all. Four slots, validated
# all-pairs (grouped bars put every bar next to every other).
FAMILY = {
    "LogReg": "#2a78d6",             # slot 1, blue
    "ss-STraTS": "#eb6834",          # slot 2, orange
    "INTERVenE-std": "#1baf7a",      # slot 3, aqua
    "INTERVenE-KB": "#4a3aa7",       # slot 7, violet
}
POS, NEG = "#2a78d6", "#e34948"   # diverging poles: cool / warm
NEUTRAL = "#f0efec"               # diverging midpoint
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Arm -> input representation (the colour key). QA arms share their base arm's
# hue and are separated by texture, so the QA axis never competes with the
# representation axis for the same channel.
ARM_FAMILY = {
    "logreg": "LogReg", "logreg_qa": "LogReg",
    "strats": "ss-STraTS", "strats_qa": "ss-STraTS",
    "intervene_std": "INTERVenE-std",
    "intervene_kb": "INTERVenE-KB",
    "intervene_kb_qa": "INTERVenE-KB",
}

# Arms whose band is shaded in the ladder figure, to group the interval model's
# rungs without spending a colour on the grouping.
INTERVENE_ARMS = {"intervene_std", "intervene_kb", "intervene_kb_qa"}

METRIC_TITLES = {
    "auroc": "AUROC", "auprc": "AUPRC", "best_f1": "Best F1",
    "f1_0_5": "F1 @ 0.5", "minrp": "min(Precision, Recall)",
}


def use_paper_style():
    """
    Purpose: Set the rcParams every figure in the paper shares.
    Method:  Serif text at print sizes, a recessive grid behind the marks, no
             top/right spines, and text in ink tokens rather than the series
             colour. Called by each figure so a notebook cell can be re-run in
             isolation.
    """
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 110,
    })


def _style_axis(ax, xgrid=True):
    """Purpose: Recessive grid behind the marks, on one axis only."""
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
    else:
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


# ---------------------------------------------------------------------------
# 1 · Target support -- the cell that picks the evaluation window
# ---------------------------------------------------------------------------

def plot_target_support(support: pd.DataFrame, threshold=None, eval_k=None,
                        save=True, name="fig1_target_support"):
    """
    Purpose: Show what each candidate context window leaves to predict, so the
             evaluation window is chosen from evidence rather than by default.
    Method:  Left panel is a magnitude heatmap -- train prevalence per (target,
             K) on a single-hue light-to-dark ramp, cells below the support
             threshold hatched so the head layout is readable at a glance.
             Right panel is how many targets survive at each K, which is the
             number the decision actually turns on.

    Args:
        support   (pd.DataFrame): Output of `benchmark.target_support`.
        threshold (float|None):   Support floor. Defaults to config.
        eval_k    (int|None):     Window to mark as chosen. Defaults to config.
        save      (bool):         Write PNG + PDF + CSV.
        name      (str):          Basename for the artefacts.

    Returns:
        matplotlib.figure.Figure
    """
    use_paper_style()
    threshold = C.OUTCOME_SUPPORT_THRESHOLD if threshold is None else threshold
    eval_k = C.EVAL_CONTEXT_DAYS if eval_k is None else eval_k

    grid = support.pivot(index="target", columns="k", values="prevalence_train")
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    ks = list(grid.columns)

    fig, axes = plt.subplots(
        1, 2, figsize=(12.6, 0.34 * len(grid) + 2.4),
        gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.34},
        # Constrained layout, because this figure has a colorbar: tight_layout
        # cannot place one (it is an Axes it does not own) and warns that the
        # result may be wrong -- which, on the figure the K decision is read
        # from, is not a warning to leave standing.
        constrained_layout=True)

    # --- magnitude heatmap ------------------------------------------------
    ax = axes[0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("kineret_blue", SEQ)
    values = grid.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=max(np.nanmax(values), threshold * 2))
    ax.set_xticks(range(len(ks)), [f"K={k}" for k in ks])
    # A candidate with no prediction head is named in muted ink and flagged, so
    # the filter's decision is visible rather than left to be inferred from a
    # small number.
    has_head = (support.drop_duplicates("target").set_index("target")["has_head"]
                if "has_head" in support.columns else None)
    tick_labels = []
    for target in grid.index:
        short = target.replace("_EVENT", "")
        if has_head is not None and not bool(has_head.get(target, True)):
            short += "  (no head)"
        tick_labels.append(short)
    ax.set_yticks(range(len(grid)), tick_labels)
    ax.set_title("Train prevalence in the label window $(K, %d]$ days"
                 % C.HORIZON_END_DAYS, pad=10)
    if has_head is not None:
        for i, target in enumerate(grid.index):
            if not bool(has_head.get(target, True)):
                ax.get_yticklabels()[i].set_color(INK_SOFT)

    # Direct labels: the contrast relief rule, and the numbers are the point.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            below = v < threshold
            if below:
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                    edgecolor="#b9b8b2", linewidth=0.0))
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7.5,
                    color=INK if v < 0.45 * np.nanmax(values) else SURFACE)
    ax.axvline(ks.index(eval_k) + 0.5, color=INK, linewidth=1.2, linestyle=":")
    ax.axvline(ks.index(eval_k) - 0.5, color=INK, linewidth=1.2, linestyle=":")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, length=0, colors=INK_SOFT)
    bar.set_label("prevalence", fontsize=8, color=INK_SOFT)

    # --- how many targets survive ----------------------------------------
    ax = axes[1]
    kept = (support.assign(ok=support["prevalence_train"] >= threshold)
                   .groupby("k")["ok"].sum())
    n_candidates = support["target"].nunique()
    colors = [SERIES if k != eval_k else "#184f95" for k in kept.index]
    ax.barh([str(k) for k in kept.index], kept.to_numpy(), height=0.55,
            color=colors)
    for y, (k, n) in enumerate(kept.items()):
        ax.text(n + max(kept.max() * 0.02, 0.08), y, str(int(n)),
                va="center", fontsize=9, color=INK)
    ax.set_xlabel(f"clear {threshold:.0%} support"
                  f"\n(of {n_candidates} candidates)")
    ax.set_ylabel("K (context days)")
    ax.set_title("Head size at each window", pad=10)
    ax.set_xlim(0, kept.max() * 1.22 if kept.max() else 1)
    ax.invert_yaxis()
    _style_axis(ax)

    fig.suptitle(
        f"Choosing the evaluation window   ·   hatched = below the "
        f"{threshold:.0%} floor   ·   dotted = chosen (K={eval_k})",
        fontsize=10, color=INK_SOFT)

    if save:
        save_figure(fig, name)
        save_table(support, name)
    return fig


# ---------------------------------------------------------------------------
# 2 · The ladder -- dot-and-interval, the headline figure
# ---------------------------------------------------------------------------

def plot_ladder(average="weighted", metrics=("auroc", "auprc", "best_f1"),
                results=None, save=True, name="fig2_ladder"):
    """
    Purpose: The study's headline -- every arm's score with its 95 % interval,
             in ladder order.
    Method:  A dot-and-interval plot, one panel per metric. One series, so one
             hue: the ordering is the y-axis's job, and spending seven hues on
             it would double-encode position. Family membership is carried by a
             faint background band rather than by hue, so the ladder reads as a
             progression instead of a set of unrelated bars.

             Intervals are the reason this is not a bar chart. Adjacent rungs
             whose intervals overlap have not been separated by the data, and
             the figure should make that impossible to miss.

    Args:
        average (str):             'weighted' or 'macro'.
        metrics (tuple):           Metrics, one panel each.
        results (pd.DataFrame|None): Output of `load_results`; loaded when None.
        save    (bool):            Write PNG + PDF + CSV.
        name    (str):             Basename for the artefacts.

    Returns:
        matplotlib.figure.Figure
    """
    use_paper_style()
    results = load_results(average=average) if results is None else results
    if results.empty:
        raise ValueError("No finished runs to plot. Run the ladder first.")

    order = [a for a in ARM_ORDER if a in set(results["arm"])]
    results = results.set_index("arm").loc[order].reset_index()
    y = np.arange(len(results))[::-1]        # rung 1 at the top

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics),
                             0.52 * len(results) + 2.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        # Faint family bands: grouping without spending a colour channel.
        for yi, arm in zip(y, results["arm"]):
            if arm in INTERVENE_ARMS:
                ax.axhspan(yi - 0.5, yi + 0.5, color="#f4f3f0", zorder=0)

        mean = results[metric].to_numpy(dtype=float)
        lo = results[f"{metric}_lo"].to_numpy(dtype=float)
        hi = results[f"{metric}_hi"].to_numpy(dtype=float)
        lo = np.where(np.isfinite(lo), lo, mean)
        hi = np.where(np.isfinite(hi), hi, mean)

        ax.hlines(y, lo, hi, color=SERIES, linewidth=2.0, alpha=0.55,
                  zorder=2)
        ax.plot(mean, y, "o", markersize=7.5, color=SERIES,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)

        # Direct labels -- required by the contrast relief rule, and the
        # numbers are what a reader wants anyway.
        span = np.nanmax(hi) - np.nanmin(lo) if np.isfinite(hi).any() else 1.0
        for yi, m, h in zip(y, mean, hi):
            if np.isfinite(m):
                ax.text(h + span * 0.045, yi, f"{m:.3f}", va="center",
                        fontsize=8.5, color=INK)

        if metric == "auroc":
            ax.axvline(0.5, color=INK_SOFT, linewidth=1.0, linestyle="--",
                       zorder=1)
            # Annotate at the BOTTOM of the panel: at the top it collided with
            # the panel title.
            ax.annotate("chance", xy=(0.5, y.min() - 0.42), fontsize=7.5,
                        color=INK_SOFT, ha="center", va="top",
                        annotation_clip=False)

        ax.set_title(METRIC_TITLES.get(metric, metric.upper()), pad=12)
        ax.set_xlim(np.nanmin(lo) - span * 0.10, np.nanmax(hi) + span * 0.22)
        _style_axis(ax)

    axes[0].set_yticks(y, [arm_label(a) for a in results["arm"]])
    fig.suptitle(
        f"Ablation ladder   ·   support-weighted across "
        f"{results['n_test'].iloc[0]} held-out patients at K={C.EVAL_CONTEXT_DAYS}"
        f"   ·   point estimate and 95 % bootstrap interval",
        y=1.02, fontsize=10, color=INK_SOFT)
    fig.tight_layout()

    if save:
        save_figure(fig, name)
        save_table(results, name)
    return fig


# ---------------------------------------------------------------------------
# 3 · QA ablation -- polarity
# ---------------------------------------------------------------------------

def plot_qa_delta(qa_delta: pd.DataFrame, save=True, name="fig3_qa_delta"):
    """
    Purpose: Does the treatment-quality signal help, and for which model?
    Method:  A polarity question, so the diverging pair with a neutral zero:
             cool for improvement, warm for harm, gray at no change. Bars are
             grouped by metric so the three models are compared like for like.

             The arms being differenced share patients, labels, events and seed,
             so the delta is attributable to the QA block rather than to
             run-to-run variance -- but a delta narrower than the intervals in
             the ladder figure is still noise.

    Args:
        qa_delta (pd.DataFrame): Output of `benchmark.qa_delta_table`.
        save     (bool):         Write PNG + PDF + CSV.
        name     (str):          Basename for the artefacts.

    Returns:
        matplotlib.figure.Figure
    """
    use_paper_style()
    if qa_delta.empty:
        raise ValueError("No paired QA arms to compare.")

    metrics = list(dict.fromkeys(qa_delta["Metric"]))
    models = list(dict.fromkeys(qa_delta["Model"]))
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(models) * len(metrics) + 1.8))

    labels, values = [], []
    for metric in metrics:
        for model in models:
            row = qa_delta[(qa_delta["Metric"] == metric) & (qa_delta["Model"] == model)]
            if row.empty:
                continue
            pretty = METRIC_TITLES.get(metric.lower(), metric)
            labels.append(f"{model}  ·  {pretty}")
            values.append(float(row["delta"].iloc[0]))

    y = np.arange(len(labels))[::-1]
    colors = [POS if v >= 0 else NEG for v in values]
    ax.barh(y, values, height=0.55, color=colors)
    ax.axvline(0, color=INK_SOFT, linewidth=1.2)

    span = max(abs(min(values)), abs(max(values))) or 1.0
    for yi, v in zip(y, values):
        offset = span * 0.03 * (1 if v >= 0 else -1)
        ax.text(v + offset, yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)

    ax.set_yticks(y, labels)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("change from adding the QA compliance block")
    ax.set_title("Does the treatment-quality signal help?", pad=12)
    _style_axis(ax)

    # Identity is never colour-alone: name the two directions.
    handles = [mpl.patches.Patch(color=POS, label="QA helps"),
               mpl.patches.Patch(color=NEG, label="QA hurts")]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()

    if save:
        save_figure(fig, name)
        save_table(qa_delta, name)
    return fig


# ---------------------------------------------------------------------------
# 4 · Per-target breakdown -- categorical identity
# ---------------------------------------------------------------------------

def plot_per_target(metric="auprc", arms=None, table=None, save=True,
                    name=None):
    """
    Purpose: Where does any advantage actually come from?
    Method:  Grouped horizontal bars, one hue per MODEL FAMILY -- that is real
             categorical identity, and the three families use the colour slots
             that pass the all-pairs colour-vision test. QA arms within a family
             are hatched rather than given a fourth hue, so the family stays the
             thing colour encodes.

             Targets are ordered by positive support and the count is printed on
             the axis, because a strong number on a single-digit-support target
             is noise, not a result.

    Args:
        metric (str):              Metric to plot.
        arms   (list|None):        Arms to include.
        table  (pd.DataFrame|None): Output of `per_outcome_table`; loaded when None.
        save   (bool):             Write PNG + PDF + CSV.
        name   (str|None):         Basename; defaults to the metric.

    Returns:
        matplotlib.figure.Figure
    """
    use_paper_style()
    name = name or f"fig4_per_target_{metric}"
    table = per_outcome_table(metric=metric, arms=arms) if table is None else table
    if table.empty:
        raise ValueError("No per-target metrics found. Run the ladder first.")

    arm_cols = [c for c in table.columns if c not in ("n_pos", "ci_reliable")]
    label_to_arm = {arm_label(a): a for a in ARM_ORDER}

    n_targets, n_arms = len(table), len(arm_cols)
    height = 0.78 / max(n_arms, 1)
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * n_targets * max(n_arms / 3, 1) + 2.0))

    y_base = np.arange(n_targets)[::-1]
    for i, col in enumerate(arm_cols):
        arm = label_to_arm.get(col, col)
        family = ARM_FAMILY.get(arm, "LogReg")
        is_qa = arm.endswith("_qa")
        offset = (i - (n_arms - 1) / 2) * height
        ax.barh(y_base + offset, table[col].to_numpy(dtype=float),
                height=height * 0.86, color=FAMILY[family],
                hatch="///" if is_qa else None,
                edgecolor=SURFACE, linewidth=0.6, label=col)

    ticks = [f"{t.replace('_EVENT', '')}  (n={int(n)})"
             for t, n in zip(table.index, table["n_pos"])]
    ax.set_yticks(y_base, ticks)
    ax.set_xlabel(METRIC_TITLES.get(metric, metric.upper()))
    ax.set_title(f"Per-target {METRIC_TITLES.get(metric, metric.upper())} "
                 f"at K={C.EVAL_CONTEXT_DAYS}", pad=12)
    # Legend below the axes: inside the plot it sat on top of the last group's
    # bars, and with seven entries there is no free corner.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.9 / n_targets),
              ncol=2, frameon=False)
    _style_axis(ax)
    fig.tight_layout()

    if save:
        save_figure(fig, name)
        save_table(table.reset_index(), name)
    return fig


# ---------------------------------------------------------------------------
# 5 · The other two questions -- onset and length of stay
# ---------------------------------------------------------------------------

def plot_error_heads(average="weighted", results=None, save=True,
                     name="fig5_onset_and_los"):
    """
    Purpose: Report the two regression questions every arm also answers.
    Method:  Dot-and-interval, same form as the ladder, but LOWER IS BETTER --
             so the axis is labelled that way and the panels sit side by side
             rather than being mixed in with the risk metrics, where a reader
             scanning for "further right is better" would misread them.

    Args:
        average (str):               'weighted' or 'macro'.
        results (pd.DataFrame|None): Output of `load_results`.
        save    (bool):              Write PNG + PDF + CSV.
        name    (str):               Basename for the artefacts.

    Returns:
        matplotlib.figure.Figure
    """
    use_paper_style()
    results = load_results(average=average) if results is None else results
    if results.empty:
        raise ValueError("No finished runs to plot.")

    order = [a for a in ARM_ORDER if a in set(results["arm"])]
    results = results.set_index("arm").loc[order].reset_index()
    y = np.arange(len(results))[::-1]

    panels = [("onset_mae_h", "Onset MAE (hours)\npositives only"),
              ("los_mae_h", "Length-of-stay MAE (hours)\ndischarged patients")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.52 * len(results) + 2.2),
                             sharey=True)

    for ax, (key, title) in zip(axes, panels):
        for yi, arm in zip(y, results["arm"]):
            if arm in INTERVENE_ARMS:
                ax.axhspan(yi - 0.5, yi + 0.5, color="#f4f3f0", zorder=0)

        mean = results[key].to_numpy(dtype=float)
        lo = results[f"{key}_lo"].to_numpy(dtype=float)
        hi = results[f"{key}_hi"].to_numpy(dtype=float)
        lo = np.where(np.isfinite(lo), lo, mean)
        hi = np.where(np.isfinite(hi), hi, mean)

        ax.hlines(y, lo, hi, color=SERIES, linewidth=2.0, alpha=0.55, zorder=2)
        ax.plot(mean, y, "o", markersize=7.5, color=SERIES,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)

        finite = np.isfinite(hi)
        span = (np.nanmax(hi[finite]) - np.nanmin(lo[finite])) if finite.any() else 1.0
        for yi, m, h in zip(y, mean, hi):
            if np.isfinite(m):
                ax.text(h + span * 0.045, yi, f"{m:.0f}", va="center",
                        fontsize=8.5, color=INK)
        ax.set_title(title, pad=12)
        ax.set_xlabel("hours  (lower is better)")
        if finite.any():
            ax.set_xlim(max(np.nanmin(lo[finite]) - span * 0.10, 0),
                        np.nanmax(hi[finite]) + span * 0.24)
        _style_axis(ax)

    axes[0].set_yticks(y, [arm_label(a) for a in results["arm"]])
    fig.suptitle("Onset timing and length of stay   ·   lower is better",
                 y=1.02, fontsize=10, color=INK_SOFT)
    fig.tight_layout()

    if save:
        save_figure(fig, name)
        save_table(results[["arm", "label", "onset_mae_h", "onset_mae_h_lo",
                            "onset_mae_h_hi", "los_mae_h", "los_mae_h_lo",
                            "los_mae_h_hi"]], name)
    return fig
