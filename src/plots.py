"""Publication-quality figures for the replication.

Every function takes already-computed results, writes one PNG into the
configured figure directory and returns the path, so the plotting layer has
no opinions about how the numbers were produced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)

#: Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE: Sequence[str] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
)

STYLE: Dict[str, Any] = {
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
}


def _save(fig: Figure, directory: str | Path, name: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("wrote figure %s", path)
    return path


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ---------------------------------------------------------------------------
# Step 1 - panel figures
# ---------------------------------------------------------------------------
def plot_coverage(coverage: pd.DataFrame, directory: str | Path,
                  name: str = "fig01_coverage_matrix") -> Path:
    """Heatmap of usable observations by country and decade."""
    with plt.rc_context(STYLE):
        height = max(4.0, 0.22 * coverage.shape[0] + 1.5)
        fig, ax = plt.subplots(figsize=(11, height))
        data = coverage.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=10)
        ax.set_yticks(range(coverage.shape[0]))
        ax.set_yticklabels(coverage.index)
        ax.set_xticks(range(coverage.shape[1]))
        ax.set_xticklabels([str(c) for c in coverage.columns], rotation=90)
        ax.set_title("Usable country-years by decade")
        ax.set_xlabel("Decade")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="years with complete data")
    return _save(fig, directory, name)


def plot_country_returns(summary: pd.DataFrame, directory: str | Path,
                         name: str = "fig02_country_real_returns") -> Path:
    """Geometric mean real returns by country and asset class.

    Split by provenance tier.  Pooling them hides the empirical cross-section
    behind a few 25-year synthetic histories whose sample means are extreme
    for reasons that have nothing to do with the economics.
    """
    wide = summary[summary["series"].isin(["dom_eq", "bond", "bill"])].pivot(
        index="iso", columns="series", values="geometric_mean")
    meta = summary.drop_duplicates("iso").set_index("iso")[["tier", "n_years"]]
    wide = wide.join(meta)

    panels = [("A", "Tier A - fully empirical (JST / JKKST)"),
              ("B", "Tier B - calibrated proxy")]
    present = [(tier, title) for tier, title in panels
               if (wide["tier"] == tier).any()]

    with plt.rc_context(STYLE):
        widths = [max((wide["tier"] == t).sum(), 1) for t, _ in present]
        fig, axes = plt.subplots(
            1, len(present), figsize=(12.5, 6.4),
            gridspec_kw={"width_ratios": widths}, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (tier, title) in zip(axes, present):
            block = wide[wide["tier"] == tier].sort_values("dom_eq")
            x = np.arange(len(block))
            width = 0.27
            for i, col in enumerate(["dom_eq", "bond", "bill"]):
                label = {"dom_eq": "Equity", "bond": "Bonds",
                         "bill": "Bills"}[col]
                ax.bar(x + (i - 1) * width, block[col] * 100, width,
                       label=label if ax is axes[0] else None,
                       color=_colour(i))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{iso} ({int(n)}y)" for iso, n in
                 zip(block.index, block["n_years"])],
                rotation=90, fontsize=8, ha="center")
            ax.set_title(title, fontsize=10)
            ax.set_xlim(-0.7, len(block) - 0.3)
        axes[0].set_ylabel("Geometric mean real return (% p.a.)")
        axes[0].legend(loc="upper left")
        fig.suptitle("Long-run real returns by country "
                     "(years of usable data in brackets)", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 2 - bootstrap diagnostics
# ---------------------------------------------------------------------------
def plot_bootstrap_diagnostics(diagnostics: Mapping[str, pd.DataFrame],
                               directory: str | Path,
                               name: str = "fig03_bootstrap_diagnostics") -> Path:
    """Mean preservation, persistence and cross-asset correlation fidelity."""
    moments = diagnostics["moments"]
    autocorr = diagnostics["autocorrelation"]
    gap = diagnostics["correlation_gap"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

        ax = axes[0]
        x = np.arange(len(moments))
        ax.bar(x - 0.2, moments["panel_pooled_mean"] * 100, 0.4,
               label="panel", color=_colour(0))
        ax.bar(x + 0.2, moments["bootstrap_mean"] * 100, 0.4,
               label="bootstrap", color=_colour(1))
        ax.set_xticks(x)
        ax.set_xticklabels(moments["series"], rotation=45, ha="right")
        ax.set_ylabel("mean (% p.a.)")
        ax.set_title("Sample mean preservation")
        ax.legend()

        ax = axes[1]
        for i, series in enumerate(autocorr["series"].unique()):
            sub = autocorr[autocorr["series"] == series]
            ax.plot(sub["lag"], sub["panel"], "--", color=_colour(i), alpha=0.7)
            ax.plot(sub["lag"], sub["bootstrap"], "-o", color=_colour(i),
                    markersize=3, label=series)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("lag (years)")
        ax.set_ylabel("autocorrelation")
        ax.set_title("Persistence: panel (dashed) vs bootstrap (solid)")
        ax.legend(fontsize=7, ncol=2)

        ax = axes[2]
        im = ax.imshow(gap.to_numpy(), cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        ax.set_xticks(range(gap.shape[1]))
        ax.set_xticklabels(gap.columns, rotation=45, ha="right")
        ax.set_yticks(range(gap.shape[0]))
        ax.set_yticklabels(gap.index)
        ax.set_title("Correlation gap (bootstrap - panel)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, directory, name)


def plot_block_sensitivity(sensitivity: pd.DataFrame, directory: str | Path,
                           name: str = "fig04_block_length_sensitivity") -> Path:
    """Dispersion of 68-year annualised outcomes against block length."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        x = sensitivity["mean_block_years"]
        ax.fill_between(x, sensitivity["dom_eq_p5_annualised"] * 100,
                        sensitivity["dom_eq_p95_annualised"] * 100,
                        alpha=0.25, color=_colour(0), label="5th-95th percentile")
        ax.plot(x, sensitivity["dom_eq_mean_annualised"] * 100, "-o",
                color=_colour(0), label="mean")
        ax2 = ax.twinx()
        ax2.plot(x, sensitivity["within_path_ar1"], "-s", color=_colour(1),
                 label="within-path AR(1)")
        ax2.set_ylabel("within-path AR(1)", color=_colour(1))
        ax2.grid(False)
        ax.set_xlabel("expected block length (years)")
        ax.set_ylabel("68-year annualised real equity return (%)")
        ax.set_title("Block-length sensitivity")
        ax.legend(loc="lower left")
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 3 - lifecycle figures
# ---------------------------------------------------------------------------
def plot_glide_paths(glide: pd.DataFrame, labels: Mapping[str, str],
                     directory: str | Path,
                     name: str = "fig05_glide_paths") -> Path:
    """Equity share by age for each candidate portfolio."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for i, column in enumerate(glide.columns):
            ax.plot(glide.index, glide[column] * 100, color=_colour(i),
                    linewidth=2, label=labels.get(column, column))
        ax.set_xlabel("Age")
        ax.set_ylabel("Equity share (%)")
        ax.set_ylim(-5, 105)
        ax.set_title("Strategy equity share over the lifecycle")
        ax.legend(fontsize=8)
    return _save(fig, directory, name)


def plot_terminal_wealth_cdf(outcomes: Mapping[str, Any], labels: Mapping[str, str],
                             directory: str | Path,
                             name: str = "fig06_terminal_wealth_cdf",
                             clip_percentile: float = 99.0) -> Path:
    """Empirical CDF of the real bequest at age 93, by strategy."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        upper = max(np.percentile(o.bequest, clip_percentile)
                    for o in outcomes.values())
        for i, (key, outcome) in enumerate(outcomes.items()):
            values = np.sort(outcome.bequest)
            grid = np.arange(1, values.size + 1) / values.size
            for ax in axes:
                ax.plot(values, grid, color=_colour(i), linewidth=1.8,
                        label=labels.get(key, key))
        axes[0].set_xlim(0, upper)
        axes[0].set_title("Bequest CDF (linear scale)")
        axes[1].set_xscale("symlog", linthresh=1.0)
        axes[1].set_title("Bequest CDF (log scale)")
        for ax in axes:
            ax.set_xlabel("Real bequest (multiples of initial annual income)")
            ax.set_ylabel("Cumulative probability")
            ax.set_ylim(0, 1)
        axes[0].legend(fontsize=8, loc="lower right")
    return _save(fig, directory, name)


def plot_retirement_consumption(outcomes: Mapping[str, Any],
                                labels: Mapping[str, str],
                                retirement_slice: slice,
                                directory: str | Path,
                                name: str = "fig07_retirement_consumption") -> Path:
    """Distribution of average real retirement consumption, by strategy."""
    keys = list(outcomes)
    data = [outcomes[k].consumption[:, retirement_slice].mean(axis=1)
            for k in keys]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

        ax = axes[0]
        parts = ax.violinplot(data, showextrema=False, widths=0.85)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(_colour(i))
            body.set_alpha(0.55)
        medians = [np.median(d) for d in data]
        ax.scatter(range(1, len(keys) + 1), medians, color="black", zorder=3,
                   s=18, label="median")
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels([labels.get(k, k) for k in keys], rotation=25,
                           ha="right", fontsize=8)
        ax.set_ylabel("Mean real retirement consumption")
        ax.set_ylim(0, float(np.percentile(np.concatenate(data), 99)))
        ax.set_title("Retirement consumption distribution")
        ax.legend(fontsize=8)

        ax = axes[1]
        for i, key in enumerate(keys):
            values = np.sort(data[i])
            grid = np.arange(1, values.size + 1) / values.size
            ax.plot(values, grid, color=_colour(i), linewidth=1.8,
                    label=labels.get(key, key))
        ax.set_xlim(0, float(np.percentile(np.concatenate(data), 99)))
        ax.set_xlabel("Mean real retirement consumption")
        ax.set_ylabel("Cumulative probability")
        ax.set_title("Retirement consumption CDF")
        ax.legend(fontsize=8, loc="lower right")
    return _save(fig, directory, name)


def plot_shortfall_curves(outcomes: Mapping[str, Any], labels: Mapping[str, str],
                          retirement_slice: slice, directory: str | Path,
                          reference: str = "balanced_all_equity",
                          name: str = "fig08_shortfall_probabilities") -> Path:
    """Shortfall probability against a grid of consumption targets.

    The left panel is the probability that mean real retirement consumption
    falls below a target; the right panel is the glide-path minus all-equity
    difference, which is the comparison the paper's headline rests on.
    """
    keys = list(outcomes)
    data = {k: outcomes[k].consumption[:, retirement_slice].mean(axis=1)
            for k in keys}
    hi = float(np.percentile(data[reference], 95)) if reference in data else 3.0
    grid = np.linspace(0.0, hi, 160)
    curves = {k: np.array([(v < g).mean() for g in grid]) for k, v in data.items()}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for i, key in enumerate(keys):
            axes[0].plot(grid, curves[key] * 100, color=_colour(i),
                         linewidth=1.8, label=labels.get(key, key))
        axes[0].set_xlabel("Real retirement consumption target")
        axes[0].set_ylabel("P(consumption < target)  (%)")
        axes[0].set_title("Shortfall probability")
        axes[0].legend(fontsize=8)

        base = curves.get(reference)
        if base is not None:
            for i, key in enumerate(keys):
                if key == reference:
                    continue
                axes[1].plot(grid, (curves[key] - base) * 100, color=_colour(i),
                             linewidth=1.8, label=labels.get(key, key))
            axes[1].axhline(0, color="black", linewidth=1.0)
        axes[1].set_xlabel("Real retirement consumption target")
        axes[1].set_ylabel(f"Shortfall gap vs {labels.get(reference, reference)} (pp)")
        axes[1].set_title("Excess shortfall probability")
        axes[1].legend(fontsize=8)
    return _save(fig, directory, name)


def plot_cec_by_risk_aversion(table: pd.DataFrame, directory: str | Path,
                              name: str = "fig09_cec_by_risk_aversion") -> Path:
    """Grouped bars of certainty equivalent consumption by risk aversion."""
    cec_cols = [c for c in table.columns if c.startswith("cec_crra_gamma")]
    ez_cols = [c for c in table.columns if c.startswith("cec_ez_")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
        for ax, cols, title in (
            (axes[0], cec_cols, "CRRA certainty equivalent consumption"),
            (axes[1], ez_cols, "Epstein-Zin certainty equivalent consumption"),
        ):
            x = np.arange(len(table))
            width = 0.8 / max(len(cols), 1)
            for i, col in enumerate(cols):
                offset = (i - (len(cols) - 1) / 2) * width
                pretty = (col.replace("cec_crra_gamma", "γ=")
                          .replace("cec_ez_gamma", "γ=")
                          .replace("_psi", ", ψ="))
                ax.bar(x + offset, table[col], width, label=pretty,
                       color=_colour(i))
            ax.set_xticks(x)
            ax.set_xticklabels(table["label"], rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("CEC (multiples of initial annual income)")
            ax.set_title(title)
            ax.legend(fontsize=8, ncol=2)
    return _save(fig, directory, name)


def plot_wealth_fan(outcomes: Mapping[str, Any], labels: Mapping[str, str],
                    ages: np.ndarray, directory: str | Path,
                    keys: Sequence[str] = ("balanced_all_equity",
                                           "target_date_fund", "sixty_forty"),
                    name: str = "fig10_wealth_fan") -> Path:
    """Percentile fan of the real wealth trajectory for selected strategies."""
    present = [k for k in keys if k in outcomes]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(present), figsize=(4.7 * len(present), 4.6),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, present):
            wealth = outcomes[key].wealth[:, :len(ages)]
            for lo, hi, alpha in ((5, 95, 0.15), (25, 75, 0.25)):
                ax.fill_between(ages, np.percentile(wealth, lo, axis=0),
                                np.percentile(wealth, hi, axis=0),
                                color=_colour(0), alpha=alpha)
            ax.plot(ages, np.percentile(wealth, 50, axis=0), color=_colour(1),
                    linewidth=2, label="median")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("Age")
            ax.set_title(labels.get(key, key), fontsize=10)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Real financial wealth\n(multiples of initial income)")
        fig.suptitle("Wealth trajectories: median with 25-75 and 5-95 percentile bands",
                     fontsize=11)
    return _save(fig, directory, name)


def plot_ruin_probability(table: pd.DataFrame, directory: str | Path,
                          name: str = "fig11_ruin_probability") -> Path:
    """Probability of exhausting financial wealth before age 93."""
    ordered = table.sort_values("prob_ruin")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        colours = [_colour(i) for i in range(len(ordered))]
        ax.barh(ordered["label"], ordered["prob_ruin"] * 100, color=colours)
        for y, value in enumerate(ordered["prob_ruin"] * 100):
            ax.text(value + 0.6, y, f"{value:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Probability of wealth depletion before age 93 (%)")
        ax.set_title("Ruin probability under the 4% real withdrawal rule")
    return _save(fig, directory, name)
