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


#: Marker shapes carry series identity alongside colour. Two entries of the
#: Okabe-Ito palette (the green and the pink) sit in the 6-8 CVD separation
#: band, which is only legal with a second encoding; every multi-series panel
#: below therefore varies the marker as well as the hue.
MARKERS: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X", "*")


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


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
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_yticks(range(coverage.shape[0]))
        ax.set_yticklabels(coverage.index)
        ax.set_xticks(range(coverage.shape[1]))
        ax.set_xticklabels([str(c) for c in coverage.columns], rotation=90)
        ax.set_title("Share of each decade with a complete return record")
        ax.set_xlabel("Decade")
        ax.grid(False)
        bar = fig.colorbar(im, ax=ax, shrink=0.7,
                           label="share of the decade with complete data")
        bar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        bar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        fig.tight_layout()
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

    # Only Tier A should ever be present. The other two stay in the list as a
    # tripwire: if a generated block returns to the panel, it gets its own
    # labelled sub-plot here rather than being averaged in silently.
    panels = [("A", "Observed returns (JST / JKKST)"),
              ("B", "Rates observed, equity simulated"),
              ("C", "Simulated returns")]
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


# ---------------------------------------------------------------------------
# Step 5 - sensitivity analysis
# ---------------------------------------------------------------------------
def plot_allocation_frontier(domestic: pd.DataFrame, equity: pd.DataFrame,
                             gammas: Sequence[float], directory: str | Path,
                             name: str = "fig12_allocation_frontier") -> Path:
    """Certainty equivalent against the two allocation dials.

    Left: how much of an all-equity portfolio should sit in the home market.
    Right: how much of the portfolio should be in equities at all.  Both
    curves are normalised to their own maximum so that three very different
    risk-aversion levels can share one axis.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        panels = (
            (axes[0], domestic, "domestic_share",
             "Domestic share of the equity sleeve"),
            (axes[1], equity, "equity_share", "Equity share of the portfolio"),
        )
        for ax, frame, column, xlabel in panels:
            for i, gamma in enumerate(gammas):
                col = f"cec_crra_gamma{float(gamma):g}"
                if col not in frame.columns:
                    continue
                block = frame.sort_values(column)
                values = block[col].to_numpy()
                ax.plot(block[column] * 100, values / values.max(),
                        "-o", markersize=3.5, color=_colour(i),
                        label=f"γ = {float(gamma):g}")
                best = int(np.argmax(values))
                ax.scatter([block[column].iloc[best] * 100], [1.0],
                           color=_colour(i), s=70, zorder=4,
                           edgecolor="white", linewidth=1.2)
            ax.set_xlabel(f"{xlabel} (%)")
            ax.set_ylabel("CEC relative to its own maximum")
            ax.legend(fontsize=8)
        axes[0].set_title("Home bias is costly")
        axes[1].set_title("More equity is better, at every risk aversion")
        fig.suptitle("Certainty equivalent consumption along the allocation "
                     "dials (dots mark each curve's optimum)", fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return _save(fig, directory, name)


def plot_risk_aversion_sweep(frame: pd.DataFrame, directory: str | Path,
                             challenger: str = "balanced_all_equity",
                             name: str = "fig13_risk_aversion_sweep") -> Path:
    """CEC by risk aversion, and the challenger's advantage over each rival."""
    wide = frame.pivot_table(index="risk_aversion", columns="label",
                             values="cec").sort_index()
    labels = {row["strategy"]: row["label"]
              for _, row in frame.drop_duplicates("strategy").iterrows()}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for i, column in enumerate(wide.columns):
            axes[0].plot(wide.index, wide[column], "-o", markersize=3,
                         color=_colour(i), label=column)
        axes[0].set_xlabel("CRRA risk aversion γ")
        axes[0].set_ylabel("Certainty equivalent consumption")
        axes[0].set_title("CEC falls with risk aversion, but the order holds")
        axes[0].legend(fontsize=8)

        base_label = labels.get(challenger, challenger)
        if base_label in wide.columns:
            for i, column in enumerate(wide.columns):
                if column == base_label:
                    continue
                axes[1].plot(wide.index,
                             (wide[base_label] / wide[column] - 1.0) * 100,
                             "-o", markersize=3, color=_colour(i), label=column)
            axes[1].axhline(0, color="black", linewidth=1.0)
        axes[1].set_xlabel("CRRA risk aversion γ")
        axes[1].set_ylabel(f"CEC advantage of\n{base_label} (%)")
        axes[1].set_title("Advantage stays positive across the whole grid")
        axes[1].legend(fontsize=8)
    return _save(fig, directory, name)


def plot_withdrawal_sensitivity(frame: pd.DataFrame, swr: pd.DataFrame,
                                target_ruin: float, directory: str | Path,
                                name: str = "fig14_withdrawal_sensitivity"
                                ) -> Path:
    """Ruin probability against withdrawal rate, plus the implied safe rate."""
    ruin = frame.pivot_table(index="withdrawal_rate", columns="label",
                             values="prob_ruin").sort_index()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                                 gridspec_kw={"width_ratios": [1.0, 1.15]})
        for i, column in enumerate(ruin.columns):
            axes[0].plot(ruin.index * 100, ruin[column] * 100, "-o",
                         markersize=3, color=_colour(i), label=column)
        axes[0].axhline(target_ruin * 100, color="black", linestyle="--",
                        linewidth=1.0, label=f"{target_ruin:.0%} ruin target")
        axes[0].axvline(4.0, color="grey", linestyle=":", linewidth=1.0)
        axes[0].text(4.05, 92, 'the "4% rule"', fontsize=8, color="grey")
        axes[0].set_xlabel("Real withdrawal rate (% of wealth at retirement)")
        axes[0].set_ylabel("Probability of ruin before age 93 (%)")
        axes[0].set_title("Ruin probability by withdrawal rate")
        axes[0].legend(fontsize=8)

        column = [c for c in swr.columns if c.startswith("safe_withdrawal_rate")][0]
        ordered = swr.sort_values(column)
        axes[1].barh(ordered["label"], ordered[column] * 100,
                     color=[_colour(i) for i in range(len(ordered))])
        axes[1].axvline(4.0, color="grey", linestyle=":", linewidth=1.2)
        axes[1].text(4.05, -0.4, '4% rule', fontsize=8, color="grey")
        for y, value in enumerate(ordered[column] * 100):
            axes[1].text(value + 0.05, y, f"{value:.2f}%", va="center",
                         fontsize=8)
        axes[1].set_xlabel(f"Withdrawal rate giving a {target_ruin:.0%} "
                           "ruin probability (%)")
        axes[1].set_title("Safe withdrawal rate by strategy")
        axes[1].tick_params(axis="y", labelsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_planning_sweeps(frames: Mapping[str, tuple], directory: str | Path,
                         metric: str = "cec_crra_gamma5",
                         name: str = "fig15_planning_sweeps") -> Path:
    """A small-multiple grid of CEC against each planning parameter."""
    items = [(title, frame, column) for title, (frame, column) in frames.items()
             if column in frame.columns and metric in frame.columns]
    if not items:  # pragma: no cover - defensive
        raise ValueError("no plottable planning sweeps supplied")
    ncols = min(3, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows),
                                 squeeze=False)
        flat = axes.ravel()
        for ax, (title, frame, column) in zip(flat, items):
            wide = frame.pivot_table(index=column, columns="label",
                                     values=metric).sort_index()
            for i, series in enumerate(wide.columns):
                ax.plot(wide.index, wide[series], "-o", markersize=3,
                        color=_colour(i), label=series)
            ax.set_xlabel(title)
            ax.set_ylabel("CEC")
        for ax in flat[len(items):]:
            ax.axis("off")
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", fontsize=8,
                   ncol=min(3, len(labels)))
        fig.suptitle("Certainty equivalent consumption across planning "
                     "parameters (γ = 5)", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    return _save(fig, directory, name)


def plot_tornado(tornado: pd.DataFrame, directory: str | Path,
                 incumbent: str = "target_date_fund",
                 name: str = "fig16_tornado") -> Path:
    """Range of the all-equity advantage across every swept assumption."""
    block = tornado[tornado["incumbent"] == incumbent].sort_values("range_pp")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.0, 0.45 * len(block) + 2.4))
        y = np.arange(len(block))
        lo = block["min_advantage_pct"].to_numpy()
        hi = block["max_advantage_pct"].to_numpy()
        median = block["median_advantage_pct"].to_numpy()
        ax.barh(y, hi - lo, left=lo, height=0.6, color=_colour(0), alpha=0.55)
        ax.scatter(median, y, color=_colour(1), zorder=4, s=28, label="median")
        ax.axvline(0, color="black", linewidth=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(block["dimension"], fontsize=9)
        ax.set_xlabel(f"CEC advantage of the 50/50 all-equity portfolio "
                      f"over the {incumbent.replace('_', ' ')} (%)")
        ax.set_title("Every bar sits entirely to the right of zero:\n"
                     "no tested assumption reverses the ranking", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 6 - retirement spending rules
# ---------------------------------------------------------------------------
def plot_spending_rate_curves(frame: pd.DataFrame, best: pd.DataFrame,
                              directory: str | Path, metric: str = "cec_gamma5",
                              name: str = "fig17_spending_rate_curves") -> Path:
    """CEC against the spending rate for each rule, and the ranking at each optimum."""
    rated = frame.dropna(subset=["rate"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                                 gridspec_kw={"width_ratios": [1.0, 1.15]})

        ax = axes[0]
        for i, (variant, block) in enumerate(rated.groupby("variant")):
            block = block.sort_values("rate")
            ax.plot(block["rate"] * 100, block[metric], "-o", markersize=3,
                    color=_colour(i), label=variant)
            peak = block.loc[block[metric].idxmax()]
            ax.scatter([peak["rate"] * 100], [peak[metric]], color=_colour(i),
                       s=70, zorder=4, edgecolor="white", linewidth=1.2)
        ax.set_xlabel("Spending rate (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        ax.set_title("Each rule has an interior optimum\n"
                     "(dots mark it)", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[1]
        ordered = best.sort_values(metric)
        colours = [_colour(2) if "Constant real" in v else _colour(0)
                   for v in ordered["variant"]]
        ax.barh(ordered["variant"], ordered[metric], color=colours)
        ax.set_xlim(min(ordered[metric]) * 0.95, max(ordered[metric]) * 1.02)
        for y, value in enumerate(ordered[metric]):
            ax.text(value + 0.002, y, f"{value:.3f}", va="center", fontsize=8)
        ax.set_xlabel("CEC at each rule's own optimal rate")
        ax.set_title("Ranking at each rule's optimum", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_spending_paths(paths: pd.DataFrame, best: pd.DataFrame,
                        directory: str | Path, metric: str = "cec_gamma5",
                        name: str = "fig18_spending_paths") -> Path:
    """Spending shape by age under each rule, and the volatility/level trade-off.

    Median and 10th-percentile paths are drawn in separate panels rather than
    as percentile bands: eight overlapping bands obscure exactly the thing
    the figure is for. Colours are keyed to the rule name so they mean the
    same thing in every panel.
    """
    variants = sorted(paths["variant"].unique())
    colour_of = {variant: _colour(i) for i, variant in enumerate(variants)}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

        for ax, column, title in (
            (axes[0], "p50", "Median real spending path"),
            (axes[1], "p10", "10th-percentile path (the bad outcomes)"),
        ):
            for variant in variants:
                block = paths[paths["variant"] == variant].sort_values("age")
                ax.plot(block["age"], block[column], color=colour_of[variant],
                        linewidth=2, label=variant)
            ax.set_xlabel("Age")
            ax.set_ylabel("Real retirement consumption")
            ax.set_title(title, fontsize=10)

        ax = axes[2]
        for _, row in best.iterrows():
            variant = str(row["variant"])
            ax.scatter(row["consumption_volatility"], row[metric],
                       s=40 + 3.0 * float(row["median_bequest"]),
                       color=colour_of.get(variant, _colour(0)), alpha=0.8,
                       edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xlabel("Consumption volatility (sd of log real spending)")
        ax.set_ylabel("Certainty equivalent consumption")
        ax.set_title("Smoother is not better: the flattest rule\n"
                     "ranks last (marker size = median bequest)", fontsize=10)

        handles = [plt.Line2D([], [], color=colour_of[v], linewidth=2, label=v)
                   for v in variants]
        fig.legend(handles=handles, loc="lower center", fontsize=8,
                   ncol=min(3, len(variants)))
        fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    return _save(fig, directory, name)


def plot_spending_bequest_pivot(frame: pd.DataFrame, directory: str | Path,
                                name: str = "fig19_spending_bequest_pivot"
                                ) -> Path:
    """How the spending-rule ranking turns on the strength of the bequest motive."""
    wide = frame.pivot_table(index="bequest_weight", columns="variant",
                             values="cec").sort_index()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for i, column in enumerate(wide.columns):
            ax.plot(wide.index, wide[column], "-o", markersize=4,
                    color=_colour(i), linewidth=2, label=column)
        ax.set_xlabel("Weight on the bequest in the utility aggregator")
        ax.set_ylabel("Certainty equivalent consumption")
        ax.set_title("The ranking of spending rules turns on the bequest motive")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 7 - optimal glide path
# ---------------------------------------------------------------------------
def plot_optimal_glide(schedules: pd.DataFrame, industry: pd.DataFrame,
                       deviation: pd.DataFrame, directory: str | Path,
                       name: str = "fig20_optimal_glide_path") -> Path:
    """Solved schedules, the domestic split, and what each age is actually worth.

    The third panel is the important one. A solved schedule plotted alone
    looks structured, but most of its deviations from a flat line sit on a
    part of the surface where the objective barely moves. Plotting the
    certainty-equivalent cost of forcing each age to 100% equity separates
    the one real feature from the search noise around it.
    """
    free = schedules[schedules["kind"] == "free_form"]
    para = schedules[schedules["kind"] == "parametric"]
    gammas = sorted(schedules["risk_aversion"].unique())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

        ax = axes[0]
        for i, gamma in enumerate(gammas):
            block = free[free["risk_aversion"] == gamma].sort_values("age")
            ax.plot(block["age"], block["equity_share"] * 100, "-o",
                    markersize=3, color=_colour(i), linewidth=1.8,
                    label=f"free-form, γ = {gamma:g}")
            pblock = para[para["risk_aversion"] == gamma].sort_values("age")
            if len(pblock):
                ax.plot(pblock["age"], pblock["equity_share"] * 100, "--",
                        color=_colour(i), linewidth=1.4, alpha=0.75,
                        label=f"parametric, γ = {gamma:g}")
        if "target_date_fund" in industry.columns:
            ax.plot(industry.index, industry["target_date_fund"] * 100,
                    color="black", linewidth=2.4, linestyle=":",
                    label="industry target-date fund")
        ax.axhline(100, color="grey", linewidth=0.8)
        ax.set_xlabel("Age")
        ax.set_ylabel("Equity share (%)")
        ax.set_ylim(-5, 108)
        ax.set_title("The solved path barely glides", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")

        ax = axes[1]
        for i, gamma in enumerate(gammas):
            block = free[free["risk_aversion"] == gamma].sort_values("age")
            ax.plot(block["age"], block["domestic_share_of_equity"] * 100,
                    "-o", markersize=3, color=_colour(i), linewidth=1.8,
                    label=f"γ = {gamma:g}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Domestic share of the equity sleeve (%)")
        ax.set_ylim(-5, 105)
        ax.set_title("Home bias stays low at every age", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[2]
        for i, gamma in enumerate(gammas):
            block = deviation[deviation["risk_aversion"] == gamma] \
                .sort_values("age")
            if not len(block):
                continue
            ax.plot(block["age"], block["cost_of_forcing_bp"], "-o",
                    markersize=3, color=_colour(i), linewidth=1.6,
                    label=f"γ = {gamma:g}")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axhspan(-1, 1, color="grey", alpha=0.18)
        ax.text(26, 1.4, "±1bp: indistinguishable from flat", fontsize=7.5,
                color="grey")
        ax.set_xlabel("Age")
        ax.set_ylabel("Cost of forcing this age to 100% equity (bp of CEC)")
        ax.set_title("Only the retirement date is worth anything", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_glide_comparison(comparison: pd.DataFrame, trace: pd.DataFrame,
                          directory: str | Path,
                          name: str = "fig21_glide_comparison") -> Path:
    """What the solved schedules buy, and how the search converged."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                 gridspec_kw={"width_ratios": [1.25, 1.0]})

        ax = axes[0]
        gammas = sorted(comparison["risk_aversion"].unique())
        names = list(comparison[comparison["risk_aversion"] == gammas[0]]
                     .sort_values("cec")["strategy"])
        y = np.arange(len(names))
        width = 0.8 / max(len(gammas), 1)
        for i, gamma in enumerate(gammas):
            block = (comparison[comparison["risk_aversion"] == gamma]
                     .set_index("strategy").reindex(names))
            offset = (i - (len(gammas) - 1) / 2) * width
            ax.barh(y + offset, block["cec"], width, color=_colour(i),
                    label=f"γ = {gamma:g}")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Certainty equivalent consumption")
        ax.set_title("Solved schedules against fixed benchmarks")
        ax.legend(fontsize=8)

        ax = axes[1]
        for i, gamma in enumerate(sorted(trace["gamma"].unique())):
            block = trace[trace["gamma"] == gamma].sort_values("sweep")
            ax.plot(block["sweep"], block["cec"], "-o", color=_colour(i),
                    label=f"γ = {gamma:g}")
        ax.set_xlabel("Coordinate-ascent sweep")
        ax.set_ylabel("Certainty equivalent consumption")
        ax.set_title("Convergence")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_retirement_anchor(anchor: pd.DataFrame, retire_age: int,
                           directory: str | Path,
                           name: str = "fig22_retirement_anchor") -> Path:
    """Does the dip at retirement survive a spending rule with no anchor?"""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for i, rule in enumerate(sorted(anchor["rule"].unique())):
            block = anchor[anchor["rule"] == rule].sort_values("age")
            ax.plot(block["age"], block["equity_share"] * 100, "-o",
                    markersize=3.5, color=_colour(i), linewidth=1.8,
                    label=rule)
        ax.axvline(retire_age, color="black", linestyle="--", linewidth=1.2)
        ax.text(retire_age + 0.4, 4, "retirement", fontsize=8)
        ax.set_xlabel("Age")
        ax.set_ylabel("Optimal equity share (%)")
        ax.set_ylim(-5, 108)
        ax.set_title("The dip at retirement belongs to the withdrawal rule,\\n"
                     "not to the investment problem")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 8 - currency hedging
# ---------------------------------------------------------------------------
def plot_hedging(frame: pd.DataFrame, break_even: pd.DataFrame,
                 directory: str | Path, metric: str = "cec_gamma5",
                 strategy: str = "balanced_all_equity",
                 name: str = "fig23_currency_hedging") -> Path:
    """Value of hedging by ratio and cost, its break-even, and the mechanism."""
    block = frame[frame["strategy"] == strategy]
    unhedged = block[block["hedge_ratio"] == 0.0]
    baseline = float(unhedged[metric].iloc[0]) if len(unhedged) else np.nan

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

        ax = axes[0]
        costs = sorted(block.loc[block["hedge_ratio"] > 0,
                                 "hedge_cost"].unique())
        show = [c for c in costs if c in (0.0, 0.002, 0.005, 0.01, 0.02)] or costs
        for i, cost in enumerate(show):
            chunk = block[(block["hedge_cost"] == cost)
                          & (block["hedge_ratio"] > 0)].sort_values("hedge_ratio")
            ratios = np.concatenate([[0.0], chunk["hedge_ratio"].to_numpy()])
            values = np.concatenate([[baseline], chunk[metric].to_numpy()])
            ax.plot(ratios * 100, (values / baseline - 1.0) * 100, "-o",
                    markersize=4, color=_colour(i),
                    label=f"cost {cost * 1e4:.0f}bp/yr")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Share of the international sleeve hedged (%)")
        ax.set_ylabel("CEC gain over unhedged (%)")
        # Titled from the data: whether any ratio pays at zero cost is the
        # whole question the panel answers.
        free = block[(block["hedge_cost"] == 0.0) & (block["hedge_ratio"] > 0)]
        pays = bool((free[metric].to_numpy() > baseline).any())
        ax.set_title("Hedging is worth little, and only in small doses" if pays
                     else "Every hedge ratio loses, even when hedging is free",
                     fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[1]
        if pays:
            ordered = break_even.sort_values("hedge_ratio")
            ticks = (ordered["hedge_ratio"] * 100).to_numpy()
            heights = (ordered["break_even_annual_cost"] * 1e4).to_numpy()
            positions = np.arange(len(ordered))
            ax.bar(positions, np.nan_to_num(heights), width=0.55,
                   color=_colour(0))
            top = np.nanmax(heights) if np.isfinite(heights).any() else 1.0
            for x, height in zip(positions, heights):
                if np.isfinite(height):
                    ax.text(x, height + top * 0.03, f"{height:.0f}bp",
                            ha="center", fontsize=9)
                else:
                    ax.text(x, top * 0.04, "never\nworth it", ha="center",
                            fontsize=9, color=_colour(1))
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
            ax.set_ylim(0, top * 1.22)
            ax.set_xlabel("Share of the international sleeve hedged")
            ax.set_ylabel("Break-even annual hedging cost (bp)")
            ax.set_title("What you could afford to pay", fontsize=10)
        else:
            # No ratio pays, so a break-even chart is an empty box. Show where
            # the loss is actually incurred instead: the bottom of the
            # distribution, which is what the certainty equivalent weighs.
            tail = (block[block["hedge_cost"] == 0.0]
                    .drop_duplicates("hedge_ratio").sort_values("hedge_ratio"))
            x = tail["hedge_ratio"].to_numpy() * 100
            ax.plot(x, tail["p5_retirement_consumption"], "-o", markersize=5,
                    color=_colour(0))
            ax.axhline(float(tail["p5_retirement_consumption"].iloc[0]),
                       color="black", linewidth=1.1, linestyle="--")
            ax.annotate("unhedged", (x[-1], tail["p5_retirement_consumption"]
                                     .iloc[0]), textcoords="offset points",
                        xytext=(-4, 5), ha="right", fontsize=8)
            ax.set_xlabel("Share of the international sleeve hedged (%)")
            ax.set_ylabel("5th-percentile retirement consumption")
            ax.set_title("The loss lands in the left tail,\nwhich is what the "
                         "certainty equivalent weighs", fontsize=10)

        ax = axes[2]
        moments = (block[block["hedge_cost"] == 0.0]
                   .drop_duplicates("hedge_ratio").sort_values("hedge_ratio"))
        ax.plot(moments["hedge_ratio"] * 100, moments["intl_sd"] * 100, "-o",
                color=_colour(0), linewidth=2,
                label="volatility of the international sleeve")
        ax2 = ax.twinx()
        ax2.plot(moments["hedge_ratio"] * 100,
                 moments["corr_intl_domestic_equity"], "-s", color=_colour(1),
                 linewidth=2, label="correlation with domestic equity")
        ax2.set_ylabel("Correlation with domestic equity", color=_colour(1))
        ax2.grid(False)
        ax.set_xlabel("Share of the international sleeve hedged (%)")
        ax.set_ylabel("Real return volatility (%)", color=_colour(0))
        ax.set_title("Why: hedging cuts standalone risk but\n"
                     "raises correlation with the home market", fontsize=10)
        handles = (ax.get_legend_handles_labels()[0]
                   + ax2.get_legend_handles_labels()[0])
        labels = (ax.get_legend_handles_labels()[1]
                  + ax2.get_legend_handles_labels()[1])
        ax.legend(handles, labels, fontsize=7.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 9 - endogenous retirement timing
# ---------------------------------------------------------------------------
def plot_retirement_timing(summary: pd.DataFrame, ages: Mapping[str, np.ndarray],
                           lottery: pd.DataFrame, directory: str | Path,
                           metric: str = "cec_gamma5",
                           baseline: str = "Fixed age 63 (baseline)",
                           name: str = "fig24_retirement_timing") -> Path:
    """When people retire, what it is worth, and the decade that decides it."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

        ax = axes[0]
        labels = list(ages)
        parts = ax.violinplot([ages[k] for k in labels], showextrema=False,
                              widths=0.85)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(_colour(i))
            body.set_alpha(0.6)
        ax.scatter(range(1, len(labels) + 1),
                   [np.median(ages[k]) for k in labels], color="black", s=18,
                   zorder=3, label="median")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel("Retirement age")
        ax.set_title("When a wealth trigger actually retires people",
                     fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[1]
        floors = sorted(summary["working_income_floor"].unique())
        variants = list(summary[summary["working_income_floor"] == floors[0]]
                        ["variant"])
        y = np.arange(len(variants))
        width = 0.8 / max(len(floors), 1)
        for i, floor in enumerate(floors):
            block = (summary[summary["working_income_floor"] == floor]
                     .set_index("variant").reindex(variants))
            base = float(block.loc[baseline, metric]) if baseline in block.index \
                else float(block[metric].iloc[0])
            offset = (i - (len(floors) - 1) / 2) * width
            ax.barh(y + offset, (block[metric] / base - 1.0) * 100, width,
                    color=_colour(i),
                    label=("no working-income floor" if floor == 0
                           else f"floor at {floor:.0%} of average earnings"))
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(variants, fontsize=7.5)
        ax.set_xlabel(f"CEC vs {baseline} (%)")
        ax.set_title("Most of the flexibility premium is the\n"
                     "model's missing safety net", fontsize=10)
        ax.legend(fontsize=7.5)

        ax = axes[2]
        ax.plot(lottery["mean_window_return"] * 100,
                lottery["median_retirement_consumption"], "-o",
                color=_colour(0), linewidth=2, label="median consumption")
        ax2 = ax.twinx()
        ax2.plot(lottery["mean_window_return"] * 100,
                 lottery["prob_ruin"] * 100, "-s", color=_colour(1),
                 linewidth=2, label="probability of ruin")
        ax2.set_ylabel("Probability of ruin (%)", color=_colour(1))
        ax2.grid(False)
        ax.set_xlabel("Annualised real return over the decade around retirement (%)")
        ax.set_ylabel("Median real retirement consumption", color=_colour(0))
        ax.set_title("The retirement-date lottery", fontsize=10)
        handles = (ax.get_legend_handles_labels()[0]
                   + ax2.get_legend_handles_labels()[0])
        labs = (ax.get_legend_handles_labels()[1]
                + ax2.get_legend_handles_labels()[1])
        ax.legend(handles, labs, fontsize=8, loc="upper left")
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 10 - conditioning the savings rate
# ---------------------------------------------------------------------------
def plot_saving(profiles: pd.DataFrame, frontier: pd.DataFrame,
                conditioning: pd.DataFrame, directory: str | Path,
                target_mean: float = 0.10,
                name: str = "fig25_savings_rate") -> Path:
    """The solved savings hump, the level question, and what conditioning adds."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

        ax = axes[0]
        for i, gamma in enumerate(sorted(profiles["risk_aversion"].unique())):
            block = profiles[profiles["risk_aversion"] == gamma] \
                .sort_values("age")
            ax.plot(block["age"], block["savings_rate"] * 100, "-o",
                    markersize=3, color=_colour(i), linewidth=1.8,
                    label=f"solved, γ = {gamma:g}")
        ax.axhline(target_mean * 100, color="black", linestyle="--",
                   linewidth=1.4, label=f"flat {target_mean:.0%} (same average)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Savings rate (% of labour income)")
        ax.set_title("Save least when young, most in peak-earning years",
                     fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[1]
        cec_cols = [c for c in frontier.columns if c.startswith("cec_gamma")]
        for i, col in enumerate(cec_cols):
            ax.plot(frontier["savings_rate"] * 100, frontier[col], "-o",
                    markersize=4, color=_colour(i),
                    label=col.replace("cec_gamma", "γ = "))
            peak = frontier.loc[frontier[col].idxmax()]
            ax.scatter([peak["savings_rate"] * 100], [peak[col]],
                       color=_colour(i), s=70, zorder=4, edgecolor="white",
                       linewidth=1.2)
        ax.set_xlabel("Constant savings rate (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        ax.set_title("The model cannot identify the level\n"
                     "(dots mark each optimum)", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[2]
        for i, rule in enumerate(sorted(conditioning["rule"].unique())):
            block = conditioning[conditioning["rule"] == rule] \
                .sort_values("sensitivity")
            ax.plot(block["sensitivity"], block["vs_base_pct"], "-o",
                    markersize=4, color=_colour(i), linewidth=1.8, label=rule)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_xlabel("Sensitivity of the savings rate to the signal")
        ax.set_ylabel("CEC vs the same rule with no conditioning (%)")
        ax.set_title("Conditioning on your own position beats\n"
                     "conditioning on the market", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 11 - taking the accumulation signal apart
# ---------------------------------------------------------------------------
#: Short axis-legend forms of the three gap definitions in :mod:`src.accumulation`.
FORM_SHORT: Mapping[str, str] = {
    "level": "level gap (income multiples)",
    "proportional": "funded ratio",
    "log": "log funded ratio",
}


def plot_response_forms(forms: pd.DataFrame, policy: pd.DataFrame,
                        directory: str | Path,
                        name: str = "fig26_savings_response_form") -> Path:
    """Which functional form, and what policy each one implies."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        for i, form in enumerate(sorted(forms["form"].unique())):
            block = forms[forms["form"] == form].sort_values("rate_move_pp")
            ax.plot(block["rate_move_pp"], block["matched_value_pct"],
                    marker=_marker(i), markersize=6, color=_colour(i),
                    linewidth=2.0, label=FORM_SHORT.get(form, form))
            peak = block.loc[block["matched_value_pct"].idxmax()]
            ax.scatter([peak["rate_move_pp"]], [peak["matched_value_pct"]],
                       s=110, color=_colour(i), zorder=5,
                       edgecolor="white", linewidth=1.6)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Extra saving when a quarter behind target\n"
                      "(percentage points of income, mid-career)")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("How hard to lean, on one comparable scale", fontsize=10)
        ax.legend(fontsize=8, title="Gap measured as", title_fontsize=8)

        ax = axes[1]
        for i, form in enumerate(sorted(policy["form"].unique())):
            block = policy[policy["form"] == form].sort_values("funded_ratio")
            ax.plot(block["funded_ratio"], block["savings_rate"] * 100,
                    marker=_marker(i), markersize=5, color=_colour(i),
                    linewidth=2.0, markevery=3,
                    label=FORM_SHORT.get(form, form))
        ax.axvline(1.0, color="grey", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Wealth as a fraction of the age target")
        ax.set_ylabel("Prescribed savings rate (%)")
        ax.set_title("The policy each form implies, at its own optimum",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)



def plot_asymmetry(asymmetry: pd.DataFrame, directory: str | Path,
                   name: str = "fig27_savings_asymmetry") -> Path:
    """Saving more when behind and saving less when ahead, priced separately."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

        grid = asymmetry.pivot_table(index="k_behind", columns="k_ahead",
                                     values="matched_value_pct")
        ax = axes[0]
        limit = float(np.nanmax(np.abs(grid.to_numpy(dtype=float))))
        im = ax.imshow(grid.to_numpy(dtype=float), origin="lower",
                       aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels([f"{r:g}" for r in grid.index])
        best = asymmetry.loc[asymmetry["matched_value_pct"].idxmax()]
        ax.scatter([list(grid.columns).index(best["k_ahead"])],
                   [list(grid.index).index(best["k_behind"])],
                   marker="*", s=280, color="black", zorder=5,
                   edgecolor="white", linewidth=1.2)
        for yi, row_key in enumerate(grid.index):
            for xi, col_key in enumerate(grid.columns):
                value = grid.iloc[yi, xi]
                if np.isfinite(value):
                    ax.text(xi, yi, f"{value:+.1f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if abs(value) < 0.6 * limit else "white")
        ax.set_xlabel("Response when ahead of target (k)")
        ax.set_ylabel("Response when behind target (k)")
        ax.set_title("Value of conditioning (%), by which half is switched on\n"
                     "(★ marks the best pair)", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="CEC vs matched constant (%)")

        ax = axes[1]
        behind_only = asymmetry[asymmetry["k_ahead"] == 0.0] \
            .sort_values("k_behind")
        ahead_only = asymmetry[asymmetry["k_behind"] == 0.0] \
            .sort_values("k_ahead")
        symmetric = asymmetry[asymmetry["symmetric"]].sort_values("k_behind")
        for i, (block, x, label) in enumerate((
                (behind_only, "k_behind", "catch up only (ease off never)"),
                (ahead_only, "k_ahead", "ease off only (catch up never)"),
                (symmetric, "k_behind", "both, same coefficient"))):
            ax.plot(block[x], block["matched_value_pct"], marker=_marker(i),
                    markersize=6, color=_colour(i), linewidth=2.0, label=label)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Sensitivity of the savings rate to the funded ratio")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Which half of the rule earns its keep", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_signal_race(best: pd.DataFrame, race: pd.DataFrame,
                     combination: pd.DataFrame, directory: str | Path,
                     name: str = "fig28_savings_signal_race") -> Path:
    """Every candidate signal, swept over the same sensitivity grid."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))

        ax = axes[0]
        labels = list(best["signal_label"])
        values = best["matched_value_pct"].to_numpy(dtype=float)
        colours = [_colour(0) if v >= 0 else _colour(1) for v in values]
        bars = ax.barh(range(len(values)), values, color=colours, height=0.62)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=8)
        span = max(float(np.abs(values).max()), 1e-9)
        for bar, value in zip(bars, values):
            offset = 0.03 * span * (1 if value >= 0 else -1)
            ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                    f"{value:+.2f}%", va="center",
                    ha="left" if value >= 0 else "right", fontsize=8)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlim(min(0.0, values.min() - 0.25 * span),
                    max(0.0, values.max() + 0.30 * span))
        ax.set_xlabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Best of each signal, at its own best sensitivity",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.0)

        ax = axes[1]
        top = list(best.sort_values("matched_value_pct", ascending=False)
                   ["signal"].head(4))
        for i, signal in enumerate(top):
            block = race[race["signal"] == signal].sort_values("sensitivity")
            ax.plot(block["sensitivity"], block["matched_value_pct"],
                    marker=_marker(i), markersize=6, color=_colour(i),
                    linewidth=2.0,
                    label=block["signal_label"].iloc[0])
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Sensitivity of the savings rate to the signal")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("How sharply the leaders peak", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[2]
        grid = combination.pivot_table(index="k_first", columns="k_second",
                                       values="matched_value_pct")
        data = grid.to_numpy(dtype=float)
        limit = float(np.nanmax(np.abs(data)))
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-limit, vmax=limit)
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels([f"{r:g}" for r in grid.index])
        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                value = data[yi, xi]
                if np.isfinite(value):
                    ax.text(xi, yi, f"{value:+.1f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if abs(value) < 0.6 * limit else "white")
        peak = combination.loc[combination["matched_value_pct"].idxmax()]
        ax.scatter([list(grid.columns).index(peak["k_second"])],
                   [list(grid.index).index(peak["k_first"])], marker="*",
                   s=280, color="black", zorder=5, edgecolor="white",
                   linewidth=1.2)
        first = str(combination["first_signal"].iloc[0]).replace("_", " ")
        second = str(combination["second_signal"].iloc[0]).replace("_", " ")
        ax.set_xlabel(f"Sensitivity to {second}")
        ax.set_ylabel(f"Sensitivity to {first}")
        ax.set_title("The two leaders, layered\n(★ marks the best pair)",
                     fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8,
                     label="CEC vs matched constant (%)")
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_feasibility(feasibility: pd.DataFrame, fan: pd.DataFrame,
                     directory: str | Path, target_mean: float = 0.10,
                     name: str = "fig29_savings_feasibility") -> Path:
    """What survives when the contribution cannot move very far."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        block = feasibility.sort_values("width")
        ax.plot(block["width"] * 100, block["matched_value_pct"], marker="o",
                markersize=6, color=_colour(0), linewidth=2.0)
        unconstrained = float(block["matched_value_pct"].iloc[-1])
        ax.axhline(unconstrained, color=_colour(1), linestyle="--",
                   linewidth=1.6)
        # Direct-labelled rather than in a legend box: the reference line and
        # the rising curve leave no corner of this panel reliably empty.
        ax.annotate(f"unconstrained ({unconstrained:+.2f}%)",
                    (float(block["width"].iloc[0]) * 100, unconstrained),
                    textcoords="offset points", xytext=(4, 6), fontsize=8,
                    color=_colour(1))
        ax.axhline(0, color="black", linewidth=1.2)
        for _, row in block.iterrows():
            if row["width"] in (0.05, 0.03):
                ax.annotate(f"±{row['width']:.0%}: {row['matched_value_pct']:+.2f}%",
                            (row["width"] * 100, row["matched_value_pct"]),
                            textcoords="offset points", xytext=(6, -14),
                            fontsize=8)
        ax.set_xlabel("How far the rate may move from its average "
                      "(± percentage points)")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Value of conditioning, by how far the\n"
                     "contribution is allowed to move", fontsize=10)

        ax = axes[1]
        ax.fill_between(fan["age"], fan["q10"] * 100, fan["q90"] * 100,
                        color=_colour(0), alpha=0.18, label="10th-90th percentile")
        ax.fill_between(fan["age"], fan["q25"] * 100, fan["q75"] * 100,
                        color=_colour(0), alpha=0.34, label="25th-75th percentile")
        ax.plot(fan["age"], fan["q50"] * 100, color=_colour(0), linewidth=2.2,
                label="median")
        ax.axhline(target_mean * 100, color="black", linestyle="--",
                   linewidth=1.4, label=f"career average ({target_mean:.0%})")
        ax.set_xlabel("Age")
        ax.set_ylabel("Realised savings rate (%)")
        ax.set_title("What the unconstrained rule actually asks for",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_value_distribution(quantiles: pd.DataFrame, by_gamma: pd.DataFrame,
                            directory: str | Path,
                            name: str = "fig30_savings_value_distribution") -> Path:
    """Whether conditioning raises the middle or lifts the bottom."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        block = quantiles.sort_values("quantile")
        ax.plot(block["quantile"] * 100, block["gain_pct"], marker="o",
                markersize=6, color=_colour(0), linewidth=2.2)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Percentile of retirement consumption")
        ax.set_ylabel("Change vs no conditioning (%)")
        ax.set_title("Where in the distribution the gain lands", fontsize=10)
        for _, row in block.iloc[[0, -1]].iterrows():
            ax.annotate(f"p{int(round(row['quantile'] * 100))}: "
                        f"{row['gain_pct']:+.1f}%",
                        (row["quantile"] * 100, row["gain_pct"]),
                        textcoords="offset points", xytext=(8, -14),
                        ha="left" if row["quantile"] < 0.5 else "right",
                        fontsize=8)

        ax = axes[1]
        for i, gamma in enumerate(sorted(by_gamma["risk_aversion"].unique())):
            gblock = by_gamma[by_gamma["risk_aversion"] == gamma] \
                .sort_values("sensitivity")
            ax.plot(gblock["sensitivity"], gblock["matched_value_pct"],
                    marker=_marker(i), markersize=6, color=_colour(i),
                    linewidth=2.0, label=f"γ = {gamma:g}")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Sensitivity of the savings rate to the funded ratio")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Who wants it, and how much", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_when_it_matters(windows: pd.DataFrame, activity: pd.DataFrame,
                         directory: str | Path,
                         name: str = "fig31_savings_when_it_matters") -> Path:
    """Which years of a career the balance is worth reading in."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        block = windows.sort_values("matched_value_pct")
        values = block["matched_value_pct"].to_numpy(dtype=float)
        colours = [_colour(0) if v >= 0 else _colour(1) for v in values]
        bars = ax.barh(range(len(values)), values, color=colours, height=0.6)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([f"ages {w}" for w in block["window"]], fontsize=8)
        span = max(float(np.abs(values).max()), 1e-9)
        for bar, value in zip(bars, values):
            ax.text(value + 0.03 * span * (1 if value >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%",
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=8)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlim(min(0.0, values.min() - 0.25 * span),
                    max(0.0, values.max() + 0.30 * span))
        ax.set_xlabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Conditioning switched on only for these ages",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.0)

        ax = axes[1]
        ax.plot(activity["age"], activity["mean_abs_deviation"] * 100,
                marker="o", markersize=4, color=_colour(0), linewidth=2.0,
                markevery=3, label="average size of the adjustment")
        ax.plot(activity["age"], activity["mean_deviation"] * 100,
                marker="s", markersize=4, color=_colour(1), linewidth=2.0,
                markevery=3, label="average direction of the adjustment")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Age")
        ax.set_ylabel("Deviation from the base rate (percentage points)")
        ax.set_title("How active the rule is at each age", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_target_choice(targets: pd.DataFrame, paths: pd.DataFrame,
                       directory: str | Path,
                       name: str = "fig32_savings_target_choice") -> Path:
    """Does the target have to be right, and does aiming higher help?"""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        for i, target in enumerate(sorted(targets["target"].unique())):
            block = targets[targets["target"] == target].sort_values("factor")
            ax.plot(block["factor"], block["matched_value_pct"],
                    marker=_marker(i), markersize=6, color=_colour(i),
                    linewidth=2.0, label=target)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.axvline(1.0, color="grey", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Target scaled by")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Aiming higher than the median path", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[1]
        for i, target in enumerate(sorted(paths["target"].unique())):
            block = paths[paths["target"] == target].sort_values("age")
            ax.plot(block["age"], block["multiple"], marker=_marker(i),
                    markersize=4, color=_colour(i), linewidth=2.0,
                    markevery=4, label=target)
        ax.set_xlabel("Age")
        ax.set_ylabel("Wealth as a multiple of current income")
        ax.set_title("What each target actually asks for", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_accumulation_interactions(by_strategy: pd.DataFrame,
                                   by_income: pd.DataFrame,
                                   directory: str | Path,
                                   name: str = "fig33_savings_interactions") -> Path:
    """What the value of reading the balance depends on."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        block = by_strategy.sort_values("matched_value_pct")
        values = block["matched_value_pct"].to_numpy(dtype=float)
        colours = [_colour(0) if v >= 0 else _colour(1) for v in values]
        bars = ax.barh(range(len(values)), values, color=colours, height=0.6)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(block["strategy"], fontsize=8)
        span = max(float(np.abs(values).max()), 1e-9)
        for bar, value in zip(bars, values):
            ax.text(value + 0.03 * span * (1 if value >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%",
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=8)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlim(min(0.0, values.min() - 0.25 * span),
                    max(0.0, values.max() + 0.30 * span))
        ax.set_xlabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Value of conditioning, by what the money is invested in",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.0)

        ax = axes[1]
        block = by_income.sort_values("volatility_factor")
        ax.plot(block["volatility_factor"], block["matched_value_pct"],
                marker="o", markersize=7, color=_colour(0), linewidth=2.2)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.axvline(1.0, color="grey", linewidth=1.0, linestyle=":")
        for _, row in block.iterrows():
            ax.annotate(f"{row['matched_value_pct']:+.2f}%",
                        (row["volatility_factor"], row["matched_value_pct"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8)
        ax.set_xlabel("Labour-income shock volatility, relative to baseline")
        ax.set_ylabel("CEC vs a constant rate saving the same (%)")
        ax.set_title("Value of conditioning, by how risky the pay cheque is",
                     fontsize=10)
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 12 - the whole allocation, solved
# ---------------------------------------------------------------------------
#: One colour per asset, held fixed everywhere the four sleeves are drawn.
ASSET_COLOURS: Mapping[str, str] = {
    "dom_eq": PALETTE[0], "intl_eq": PALETTE[2],
    "bond": PALETTE[1], "bill": PALETTE[4],
}
ASSET_NAMES: Mapping[str, str] = {
    "dom_eq": "Domestic equity", "intl_eq": "International equity",
    "bond": "Bonds", "bill": "Bills",
}


def plot_full_allocation(schedules: pd.DataFrame, deviation: pd.DataFrame,
                         directory: str | Path, retire_age: int = 63,
                         name: str = "fig34_full_allocation") -> Path:
    """The solved four-asset schedule, and what each age of it is worth."""
    assets = list(ASSET_NAMES)
    gammas = sorted(schedules["risk_aversion"].unique())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(gammas) + 1,
                                 figsize=(4.6 * (len(gammas) + 1), 4.8))

        for ax, gamma in zip(axes, gammas):
            block = schedules[schedules["risk_aversion"] == gamma] \
                .sort_values("age")
            ax.stackplot(block["age"],
                         *[block[a] * 100 for a in assets],
                         labels=[ASSET_NAMES[a] for a in assets],
                         colors=[ASSET_COLOURS[a] for a in assets],
                         edgecolor="white", linewidth=0.3)
            ax.axvline(retire_age, color="black", linestyle="--",
                       linewidth=1.2)
            ax.annotate("retires", (retire_age, 101), fontsize=8,
                        ha="center", va="bottom")
            ax.set_xlim(block["age"].min(), block["age"].max())
            ax.set_ylim(0, 100)
            ax.set_xlabel("Age")
            ax.set_ylabel("Portfolio weight (%)")
            ax.set_title(f"Solved allocation, γ = {gamma:g}", fontsize=10)
            ax.grid(False)
        axes[0].legend(fontsize=8, loc="lower left", framealpha=0.9,
                       facecolor="white")

        ax = axes[-1]
        for i, gamma in enumerate(gammas):
            block = deviation[deviation["risk_aversion"] == gamma] \
                .sort_values("age")
            ax.plot(block["age"], block["cost_of_resetting_bp"],
                    marker=_marker(i), markersize=3.5, markevery=4,
                    color=_colour(i), linewidth=1.7, label=f"γ = {gamma:g}")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.annotate("1 basis point", (float(deviation["age"].min()), 1.0),
                    textcoords="offset points", xytext=(4, 5), fontsize=8)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("Age")
        ax.set_ylabel("Cost of resetting that age to the average (bp)")
        ax.set_title("What each age's allocation is worth", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_allocation_comparison(comparison: pd.DataFrame, phases: pd.DataFrame,
                               directory: str | Path,
                               name: str = "fig35_allocation_comparison") -> Path:
    """The solved schedule against the benchmarks, and its phase averages."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        gammas = sorted(comparison["risk_aversion"].unique())
        strategies = list(comparison[comparison["risk_aversion"] == gammas[0]]
                          .sort_values("cec", ascending=False)["strategy"])
        width = 0.8 / max(len(gammas), 1)
        positions = np.arange(len(strategies))
        for i, gamma in enumerate(gammas):
            block = comparison[comparison["risk_aversion"] == gamma] \
                .set_index("strategy").reindex(strategies)
            ax.barh(positions + i * width, block["gap_to_best_pct"],
                    height=width * 0.9, color=_colour(i),
                    label=f"γ = {gamma:g}")
        ax.set_yticks(positions + width * (len(gammas) - 1) / 2.0)
        ax.set_yticklabels([s.replace("_", " ") for s in strategies],
                           fontsize=8)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Gap to the best schedule at that risk aversion (%)")
        ax.set_title("The solved schedule against the benchmarks", fontsize=10)
        ax.grid(axis="y", alpha=0.0)
        ax.legend(fontsize=8, loc="lower left")

        ax = axes[1]
        assets = list(ASSET_NAMES)
        labels = [f"γ = {r.risk_aversion:g}\n{r.phase}"
                  for r in phases.itertuples()]
        bottom = np.zeros(len(phases))
        x = np.arange(len(phases))
        for a in assets:
            values = phases[a].to_numpy(dtype=float) * 100
            ax.bar(x, values, bottom=bottom, color=ASSET_COLOURS[a],
                   label=ASSET_NAMES[a], width=0.62, edgecolor="white",
                   linewidth=0.6)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Average weight (%)")
        ax.set_title("Average solved weights by phase", fontsize=10)
        ax.grid(axis="x", alpha=0.0)
        ax.legend(fontsize=8, ncol=2, loc="lower center")
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 13 - leverage
# ---------------------------------------------------------------------------
def plot_leverage_surface(sweep: pd.DataFrame, optimal: pd.DataFrame,
                          directory: str | Path,
                          name: str = "fig36_leverage_surface") -> Path:
    """What leverage is worth across the price of credit."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

        ax = axes[0]
        for i, spread in enumerate(sorted(sweep["spread"].unique())):
            block = sweep[sweep["spread"] == spread].sort_values("leverage")
            ax.plot(block["leverage"], block["vs_unlevered_pct"],
                    marker=_marker(i), markersize=5, color=_colour(i),
                    linewidth=1.9, label=f"{spread:.1%} over bills")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Leverage ratio")
        ax.set_ylabel("CEC vs the unlevered portfolio (%)")
        ax.set_title("The value of borrowing, by what it costs", fontsize=10)
        ax.legend(fontsize=8, title="Annual spread", title_fontsize=8)

        ax = axes[1]
        block = optimal.sort_values("spread")
        ax.plot(block["spread"] * 100, block["leverage"], marker="o",
                markersize=7, color=_colour(0), linewidth=2.2,
                drawstyle="steps-post")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
        # Left end: the right end is where the step function lands on the line
        # and the point labels already crowd it.
        ax.annotate("unlevered", (float(block["spread"].min()) * 100, 1.0),
                    textcoords="offset points", xytext=(4, 6), fontsize=8,
                    ha="left")
        for _, row in block.iterrows():
            ax.annotate(f"{row['vs_unlevered_pct']:+.2f}%",
                        (row["spread"] * 100, row["leverage"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5)
        ax.set_xlabel("Annual borrowing spread over the real bill rate (%)")
        ax.set_ylabel("Optimal leverage ratio")
        ax.set_title("Optimal leverage collapses as credit gets dearer",
                     fontsize=10)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_leverage_detail(detail: pd.DataFrame, schedule: pd.DataFrame,
                         directory: str | Path,
                         name: str = "fig37_leverage_detail") -> Path:
    """What leverage does to the shape of the outcome, and to its age profile."""
    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

        ax = axes[0]
        base = detail[np.isclose(detail["leverage"], 1.0)].iloc[0]
        for i, (_, row) in enumerate(
                detail[~np.isclose(detail["leverage"], 1.0)].iterrows()):
            change = [(row[f"p{q}_retirement_consumption"]
                       / base[f"p{q}_retirement_consumption"] - 1.0) * 100.0
                      for q in quantiles]
            ax.plot(quantiles, change, marker=_marker(i), markersize=5,
                    color=_colour(i), linewidth=1.9,
                    label=f"{row['leverage']:g}× leverage")
        ax.axhline(0, color="black", linewidth=1.2)
        # The top percentile runs to several hundred percent while the left
        # tail moves by a few; on a linear axis the tail that matters is a
        # flat line at zero.
        ax.set_yscale("symlog", linthresh=10.0)
        ax.set_xlabel("Percentile of retirement consumption")
        ax.set_ylabel("Change vs the unlevered portfolio (%), symlog")
        ax.set_title("Leverage widens both tails", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[1]
        for i, spread in enumerate(sorted(schedule["spread"].unique())):
            block = schedule[schedule["spread"] == spread].sort_values("age")
            ax.plot(block["age"], block["leverage"], color=_colour(i),
                    linewidth=0.9, alpha=0.35)
            smooth = block["leverage"].rolling(5, center=True,
                                               min_periods=1).mean()
            ax.plot(block["age"], smooth, marker=_marker(i), markersize=3.5,
                    markevery=5, color=_colour(i), linewidth=2.0,
                    label=f"{spread:.1%} over bills")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Age")
        ax.set_ylabel("Solved leverage ratio")
        ax.set_title("Solving a leverage ratio for every age\n"
                     "(faint: raw solution, bold: 5-year mean)", fontsize=10)
        ax.legend(fontsize=8, title="Annual spread", title_fontsize=8)
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 14 - where the data comes from
# ---------------------------------------------------------------------------
def plot_provenance(era: pd.DataFrame, contamination: pd.DataFrame,
                    tail: pd.DataFrame, countries: pd.DataFrame,
                    directory: str | Path,
                    name: str = "fig38_data_provenance") -> Path:
    """What the panel is made of, and the one test its source fails.

    The middle panel used to show how much of an investor's international leg
    was simulated. That is zero now, and a bar chart of zeros says nothing, so
    it shows the panel's actual shape instead: how many years each country
    contributes, which is what the history-weighted country draw acts on.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

        ax = axes[0]
        block = era.copy()
        x = np.arange(len(block))
        # Cells, not country-years: one country, one year, one return series.
        total = block.get("return_cells", block["country_years"]).to_numpy(float)
        simulated = block["simulated"].to_numpy(float)
        observed = total - simulated
        ax.bar(x, observed, color=_colour(0), width=0.62, label="observed")
        if simulated.any():
            ax.bar(x, simulated, bottom=observed, color=_colour(1), width=0.62,
                   label="generated")
        # Bar height tracks era length as much as coverage, so the label
        # carries the number that does not: how many countries were actually
        # in the cross-section.
        if "mean_countries_available" in block:
            for i, value in enumerate(block["mean_countries_available"]):
                ax.text(i, total[i] * 1.02, f"{value:.0f} countries",
                        ha="center", fontsize=8, color="0.35")
            ax.set_ylim(0, total.max() * 1.16)
        ax.set_xticks(x)
        ax.set_xticklabels(block["era"], fontsize=8)
        ax.set_ylabel("Return cells (country x year x series)")
        title = ("Every return cell in the panel is an observation"
                 if not simulated.any() else
                 "A generated block has returned to the panel")
        ax.set_title(f"{title}\n(bar height also reflects era length)",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="x", alpha=0.0)

        ax = axes[1]
        block = countries.sort_values("usable_years")
        y = np.arange(len(block))
        ax.barh(y, block["usable_years"], color=_colour(2), height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(block["iso"], fontsize=7.5)
        ax.set_xlabel("Usable country-years in the panel")
        span = block["usable_years"]
        ax.set_title(f"Histories run {int(span.min())}-{int(span.max())} years, "
                     f"so weighting the country\ndraw by history barely "
                     f"differs from weighting it evenly", fontsize=10)
        ax.grid(axis="y", alpha=0.0)

        ax = axes[2]
        block = tail.sort_values("ratio")
        y = np.arange(len(block))
        ax.barh(y, block["ratio"], color=_colour(1), height=0.62)
        ax.axvline(1.0, color="black", linewidth=1.3)
        ax.annotate("equal variance", (1.0, len(block) - 0.4),
                    textcoords="offset points", xytext=(5, 0), fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(block["iso"], fontsize=7.5)
        ax.set_xlabel("Tail s.d. \u00f7 reference s.d. (equity returns)")
        ax.set_title("Every country's last five years are\n"
                     "smoother than its history", fontsize=10)
        ax.grid(axis="y", alpha=0.0)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_housing(audit: pd.DataFrame, directory: str | Path,
                 name: str = "fig39_housing_smoothing") -> Path:
    """Why the observed housing series stays out of the investable set.

    Left: risk and return by country, housing as published and again with its
    own smoothing undone, against that country's equity. Right: the lag-one
    autocorrelation that does the work, country by country.
    """
    block = audit.sort_values("sd").reset_index(drop=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                 gridspec_kw={"width_ratios": [1.25, 1.0]})

        ax = axes[0]
        series = [
            ("sd", "mean", 0, "o", "Housing, as published"),
            ("sd_desmoothed", "mean", 1, "s", "Housing, de-smoothed"),
            ("equity_sd", "equity_mean", 2, "^", "Domestic equity"),
        ]
        for x_col, y_col, colour, marker, label in series:
            ax.scatter(block[x_col] * 100, block[y_col] * 100,
                       s=46, color=_colour(colour), marker=marker,
                       edgecolor="white", linewidth=0.6, label=label, zorder=3)
        # One arrow per country, published -> de-smoothed, so the correction is
        # readable as a movement rather than as two unrelated clouds.
        for row in block.itertuples():
            ax.annotate("", xy=(row.sd_desmoothed * 100, row.mean * 100),
                        xytext=(row.sd * 100, row.mean * 100),
                        arrowprops={"arrowstyle": "->", "linewidth": 0.7,
                                    "color": "0.55", "shrinkA": 3,
                                    "shrinkB": 3}, zorder=2)
        ax.set_xlabel("Standard deviation of real annual returns (%)")
        ax.set_ylabel("Mean real annual return (%)")
        ax.set_title("Housing earns equity-like returns at a fraction of the\n"
                     "risk; de-smoothing closes part of that gap, not all",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower right")

        ax = axes[1]
        order = block.sort_values("autocorrelation")
        y = np.arange(len(order))
        height = 0.38
        ax.barh(y + height / 2, order["autocorrelation"], height,
                color=_colour(0), label="Housing")
        ax.barh(y - height / 2, order["equity_autocorrelation"], height,
                color=_colour(2), label="Domestic equity")
        ax.axvline(0.0, color="black", linewidth=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels(order["iso"], fontsize=7.5)
        ax.set_xlabel("First-order autocorrelation of real returns")
        ax.set_title("Housing returns are persistent where\n"
                     "equity returns are not", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.0)
        fig.tight_layout()
    return _save(fig, directory, name)
