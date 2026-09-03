"""Publication-quality figures for the replication.

Every function takes already-computed results, writes one PNG into the
configured figure directory and returns the path, so the plotting layer has
no opinions about how the numbers were produced.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

#: The paper prints every figure into a 16.2 cm text column -- 6.4 inches --
#: and scales whatever it is handed down to fit. A figure authored twenty-one
#: inches wide therefore arrives on the page at three-tenths of the size it
#: was drawn at, and takes its 8 pt tick labels down to 2.5 pt with it. Every
#: figure below is authored at the width it is printed at, so a label is the
#: size it says it is. Nothing here should be made wider without checking
#: what the page does to it.
PAGE_WIDTH_IN: float = 6.4

STYLE: Dict[str, Any] = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
}


def _save(fig: Figure, directory: str | Path, name: str) -> Path:
    """Write one figure, with the crop applied here rather than at the call site.

    Most functions below close their ``rc_context`` before returning, so a
    ``savefig.bbox`` set in :data:`STYLE` would never reach ``savefig`` and
    long rotated tick labels would be cropped off the bottom of the canvas.
    Passing the crop explicitly makes the saved image independent of where
    the call happens to sit.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=STYLE["savefig.dpi"], bbox_inches="tight",
                pad_inches=0.08, facecolor=fig.get_facecolor())
    plt.close(fig)
    LOGGER.info("wrote figure %s", path)
    return path


def _grid(n: int, panel_height: float, *, max_cols: int = 2,
          width: float = PAGE_WIDTH_IN, span_last: bool = True,
          width_ratios: Sequence[float] | None = None,
          spare: bool = False, **kwargs: Any) -> Any:
    """``n`` panels wrapped into a page-width grid, returned flat.

    A single row of four panels is twenty-one inches wide and the page gives
    a figure six and a half, so the row has to wrap for its panels to be
    wide enough to read once printed. Axes come back in the order they were
    laid out, so a caller still indexes ``axes[0]``, ``axes[1]``, ...

    When the grid has a hole -- three panels in two columns -- the last panel
    spans it rather than leaving a gap, unless ``span_last`` says otherwise.
    Pass ``spare`` to get the empty cells back as a third value, for a caller
    that would rather put its legend in the hole than below the figure.
    """
    n = int(n)
    ncols = max(1, min(n, int(max_cols)))
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(float(width), float(panel_height) * nrows))
    spec: Dict[str, Any] = dict(kwargs)
    if width_ratios is not None:
        spec["width_ratios"] = list(width_ratios)
    gs = fig.add_gridspec(nrows, ncols, **spec)
    hole = nrows * ncols - n
    axes = []
    for i in range(n):
        row, col = divmod(i, ncols)
        if span_last and hole and i == n - 1 and row == nrows - 1:
            axes.append(fig.add_subplot(gs[row, col:]))
        else:
            axes.append(fig.add_subplot(gs[row, col]))
    out = np.array(axes, dtype=object)
    if spare:
        empty = [gs[divmod(i, ncols)] for i in range(n, nrows * ncols)]
        return fig, out, empty
    return fig, out


#: Marker shapes carry series identity alongside colour. Two entries of the
#: Okabe-Ito palette (the green and the pink) sit in the 6-8 CVD separation
#: band, which is only legal with a second encoding; every multi-series panel
#: below therefore varies the marker as well as the hue.
MARKERS: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X", "*")


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


#: Display names for the international-sleeve weighting schemes, so a figure
#: legend never prints a bare config key.
LABEL_OF: Dict[str, str] = {
    "equal": "Equal-weighted", "gdp": "Real GDP", "pop": "Population",
    "gdp_pc": "GDP per capita", "inverse_vol": "Inverse volatility",
}


def _wrap(text: str, width: int) -> str:
    """Soft-wrap a tick label so long strategy names do not collide.

    Line breaks already present in ``text`` are kept, so a short form that
    chose its own break points is not re-flowed into something worse.
    """
    parts = [textwrap.fill(part, width) if part.strip() else part
             for part in str(text).split("\n")]
    return "\n".join(parts) or str(text)


#: Tick-length forms of the wedge positions in Section #franking. The names in
#: ``src/franking.ANCHORS`` are written for a table column, where a line is as
#: wide as the page; on a category axis each gets about a bar's width.
POSITION_LABEL: Dict[str, str] = {
    "neither tax": "neither\ntax",
    "withholding only": "withholding\nonly",
    "a taxed fund, nothing franked": "taxed fund,\nunfranked",
    "an Australian fund still accumulating": "accumulating\nfund",
    "an Australian fund paying a pension": "pension-phase\nfund",
}

#: Display names for the panel's return series, so no axis prints a bare
#: column key at a reader.
SERIES_LABEL: Dict[str, str] = {
    "dom_eq": "Domestic equity",
    "intl_eq": "International equity",
    "bond": "Domestic bonds",
    "bill": "Bills",
    "inflation": "Inflation",
    "housing": "Housing",
}

#: The tightest form of a series name, for a heatmap axis with five categories
#: and a third of a panel to put them in. The line breaks are chosen here
#: rather than left to ``textwrap``, which would split "International" itself.
#: Short forms for the three readings of a correlated pay cheque. The full
#: names are sentences and belong in the prose, not on a legend.
#: Short forms for the pension regimes, for a category axis.
PENSION_TICK: Dict[str, str] = {
    "us_social_security": "US, 10%\nsaving",
    "us_matched_saving": "US, 20.2%\nsaving",
    "age_pension_untested": "AP rate,\nno means test",
    "age_pension_matched": "Age Pension,\n10% saving",
    "australia_as_legislated": "Australia as\nlegislated",
    "australia_non_homeowner": "Australia,\nnon-homeowner",
}

MODE_TICK: Dict[str, str] = {
    "home": "home market only",
    "strict": "home only, foreign pinned to 0",
    "diagonal": "both markets equally",
}

SERIES_ABBR: Dict[str, str] = {
    "dom_eq": "Dom.\nequity", "intl_eq": "Intl.\nequity",
    "bond": "Dom.\nbonds", "bill": "Bills", "inflation": "Inflation",
    "housing": "Housing",
}

#: Short forms for the configured strategy names. The labels in ``config.yaml``
#: are written for a table column, where a line is as wide as the page; on a
#: category axis they have to fit under a single bar, and the long ones ran
#: off the canvas. Keys cover both the configured label and the strategy key,
#: because different figures have one or the other to hand.
#: ``tests/test_plots.py`` asserts every configured strategy is covered.
STRATEGY_LABEL: Dict[str, str] = {
    "50/50 Domestic/International Equity": "50/50 domestic/international",
    "100% Domestic Equity": "100% domestic equity",
    "100% International Equity": "100% international equity",
    "60/40 Domestic Equity/Domestic Bonds": "60/40 domestic equity/bonds",
    "Target-Date Fund (glide path)": "Target-date fund",
    "100% Bills (cash)": "100% bills (cash)",
}

#: Names for the solved schedules, which are not configured strategies and so
#: have no label of their own; without these a figure prints the raw key.
STRATEGY_LABEL.update({
    "full_simplex_optimal": "Solved (four assets, by age)",
    "free_form_optimal": "Solved (free-form glide)",
    "parametric_optimal": "Solved (parametric glide)",
    "docs07_free_form": "Solved (free-form glide)",
    "docs07_parametric": "Solved (parametric glide)",
})

#: Which configured label each strategy key carries, so a figure holding only
#: the key gets the same short form as one holding the label.
STRATEGY_KEYS: Dict[str, str] = {
    "balanced_all_equity": "50/50 Domestic/International Equity",
    "domestic_equity": "100% Domestic Equity",
    "international_equity": "100% International Equity",
    "sixty_forty": "60/40 Domestic Equity/Domestic Bonds",
    "target_date_fund": "Target-Date Fund (glide path)",
    "bills_only": "100% Bills (cash)",
}
for _key, _label in STRATEGY_KEYS.items():
    STRATEGY_LABEL[_key] = STRATEGY_LABEL[_label]
del _key, _label


def _flat(text: Any, width: int = 26) -> str:
    """A category label sized for an axis rather than for a table column.

    Some tables carry a strategy key with its underscores already turned into
    spaces, so the spaced form is looked up too rather than falling through to
    the raw key.
    """
    key = str(text)
    source = (STRATEGY_LABEL.get(key)
              or STRATEGY_LABEL.get(key.replace(" ", "_"))
              or SERIES_LABEL.get(key) or key)
    return _wrap(str(source).replace("_", " "), width)


def _abbr(text: Any) -> str:
    """The shortest readable form of a series name, wrapped for a tick."""
    return SERIES_ABBR.get(str(text), _flat(text, 10))


def _legend(text: Any) -> str:
    """The same name on one line, for a legend entry.

    A legend sits inside the axes and lays its own text out, so it needs the
    unwrapped form. Flattening :func:`_abbr` instead would join a label broken
    mid-word back together with a space in the middle of it.
    """
    return _flat(text, 200).replace("\n", " ")


#: Roughly how many characters of a 9 pt title fit in an inch of axes.
CHARS_PER_INCH: float = 15.0


def _title(ax: Any, text: str, width: int | None = None,
           **kwargs: Any) -> None:
    """A panel title, wrapped to the panel it sits over.

    A title is centred on its axes and overflows both edges when it is longer
    than the axes are wide, and the overflow lands on whatever is beside it.
    How much fits depends on the panel: a full-width figure takes eighty
    characters, one of two panels in a row about forty, so the limit is read
    off the axes rather than fixed.

    A title that already breaks itself into lines that fit is left as it is,
    because those break points were chosen for the sense. One whose own lines
    are still too long is re-flowed from scratch, which reads better than
    wrapping each of its lines separately and leaving a one-word third line.
    """
    if width is None:
        inches = ax.get_position().width * ax.figure.get_figwidth()
        width = max(20, int(inches * CHARS_PER_INCH))
    lines = str(text).split("\n")
    if all(len(line) <= width for line in lines):
        ax.set_title(str(text), **kwargs)
    else:
        ax.set_title(textwrap.fill(" ".join(" ".join(lines).split()), width),
                     **kwargs)


def _strategy_ticks(ax: Any, labels: Sequence[Any],
                    fontsize: float = 6.5) -> None:
    """Strategy names on a category axis: short, angled, never truncated.

    Upright they need close to an inch each and a panel in this paper is
    three inches wide, so six of them run into one another. Angled, each one
    only needs its diagonal, and :func:`_save` crops to whatever they use, so
    nothing is lost off the bottom of the canvas.

    They are kept to one line each: a wrapped label set at an angle drops its
    second line across the label beside it.
    """
    ax.set_xticklabels([_flat(v, 999) for v in labels], rotation=30,
                       ha="right", rotation_mode="anchor", fontsize=fontsize)


def _variant(text: Any) -> str:
    """Compress a retirement-timing variant name onto a category axis.

    The configured names ("Wealth trigger 20x income") are written to be read
    in a table column; eight of them under one 3-inch panel are not.
    """
    out = (str(text).replace("Fixed age ", "Age ")
           .replace("Wealth trigger ", "Trigger ")
           .replace("Flexible +/-", "Flex \u00b1")
           .replace(" years,", "y,")
           .replace(" income", ""))
    return out.replace(" (baseline)", " (base)")


# ---------------------------------------------------------------------------
# Step 1 - panel figures
# ---------------------------------------------------------------------------
def plot_coverage(coverage: pd.DataFrame, directory: str | Path,
                  name: str = "fig01_coverage_matrix") -> Path:
    """Heatmap of usable observations by country and decade."""
    with plt.rc_context(STYLE):
        height = max(2.8, 0.155 * coverage.shape[0] + 1.3)
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, height))
        data = coverage.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_yticks(range(coverage.shape[0]))
        ax.set_yticklabels(coverage.index)
        ax.set_xticks(range(coverage.shape[1]))
        ax.set_xticklabels([str(c) for c in coverage.columns], rotation=90)
        _title(ax, "Share of each decade with a complete return record")
        ax.set_xlabel("Decade")
        ax.grid(False)
        bar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                           label="Share of the decade")
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
        fig, axes = _grid(len(present), 3.6, max_cols=len(present),
                          width_ratios=widths, span_last=False)
        for ax in axes[1:]:
            ax.sharey(axes[0])
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
                rotation=90, fontsize=7, ha="center")
            _title(ax, title, fontsize=8.5)
            ax.set_xlim(-0.7, len(block) - 0.3)
        axes[0].set_ylabel("Geometric mean real return (% p.a.)")
        axes[0].legend(loc="upper left")
        fig.suptitle("Long-run real returns by country "
                     "(years of usable data in brackets)", fontsize=10)
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
        fig, axes = _grid(3, 3.0)

        ax = axes[0]
        x = np.arange(len(moments))
        ax.bar(x - 0.2, moments["panel_pooled_mean"] * 100, 0.4,
               label="panel", color=_colour(0))
        ax.bar(x + 0.2, moments["bootstrap_mean"] * 100, 0.4,
               label="bootstrap", color=_colour(1))
        ax.set_xticks(x)
        ax.set_xticklabels([_abbr(v) for v in moments["series"]])
        ax.set_ylabel("Mean (% p.a.)")
        _title(ax, "Sample mean preservation")
        ax.legend()

        ax = axes[1]
        for i, series in enumerate(autocorr["series"].unique()):
            sub = autocorr[autocorr["series"] == series]
            ax.plot(sub["lag"], sub["panel"], "--", color=_colour(i), alpha=0.7)
            ax.plot(sub["lag"], sub["bootstrap"], "-o", color=_colour(i),
                    markersize=3, label=_flat(series, 999))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (years)")
        ax.set_ylabel("Autocorrelation")
        _title(ax, "Persistence: panel (dashed) vs bootstrap (solid)")
        ax.legend(fontsize=6.5, ncol=2)

        ax = axes[2]
        im = ax.imshow(gap.to_numpy(), cmap="RdBu_r", vmin=-0.15, vmax=0.15,
                       aspect="auto")
        ax.set_xticks(range(gap.shape[1]))
        ax.set_xticklabels([_abbr(c) for c in gap.columns], fontsize=7)
        ax.set_yticks(range(gap.shape[0]))
        ax.set_yticklabels([_flat(i, 20) for i in gap.index], fontsize=7)
        _title(ax, "Correlation gap (bootstrap - panel)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_block_sensitivity(sensitivity: pd.DataFrame, directory: str | Path,
                           name: str = "fig04_block_length_sensitivity") -> Path:
    """Dispersion of 68-year annualised outcomes against block length."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.9))
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
        _title(ax, "Block-length sensitivity")
        ax.legend(loc="lower left")
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 3 - lifecycle figures
# ---------------------------------------------------------------------------
def plot_glide_paths(glide: pd.DataFrame, labels: Mapping[str, str],
                     directory: str | Path,
                     name: str = "fig05_glide_paths") -> Path:
    """Equity share by age for each candidate portfolio."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.7))
        for i, column in enumerate(glide.columns):
            ax.plot(glide.index, glide[column] * 100, color=_colour(i),
                    linewidth=2, label=_flat(labels.get(column, column), 999))
        ax.set_xlabel("Age")
        ax.set_ylabel("Equity share (%)")
        ax.set_ylim(-5, 105)
        _title(ax, "Strategy equity share over the lifecycle")
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_terminal_wealth_cdf(outcomes: Mapping[str, Any], labels: Mapping[str, str],
                             directory: str | Path,
                             name: str = "fig06_terminal_wealth_cdf",
                             clip_percentile: float = 99.0) -> Path:
    """Empirical CDF of the real bequest at age 93, by strategy."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 2.8)
        upper = max(np.percentile(o.bequest, clip_percentile)
                    for o in outcomes.values())
        for i, (key, outcome) in enumerate(outcomes.items()):
            values = np.sort(outcome.bequest)
            grid = np.arange(1, values.size + 1) / values.size
            for ax in axes:
                ax.plot(values, grid, color=_colour(i), linewidth=1.8,
                        label=_flat(labels.get(key, key), 999))
        axes[0].set_xlim(0, upper)
        _title(axes[0], "Bequest CDF (linear scale)")
        axes[1].set_xscale("symlog", linthresh=1.0)
        _title(axes[1], "Bequest CDF (log scale)")
        for ax in axes:
            ax.set_xlabel("Real bequest (multiples of initial annual income)")
            ax.set_ylabel("Cumulative probability")
            ax.set_ylim(0, 1)
        axes[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
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
        fig, axes = _grid(2, 2.8)

        ax = axes[0]
        parts = ax.violinplot(data, showextrema=False, widths=0.85)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(_colour(i))
            body.set_alpha(0.55)
        medians = [np.median(d) for d in data]
        ax.scatter(range(1, len(keys) + 1), medians, color="black", zorder=3,
                   s=18, label="median")
        ax.set_xticks(range(1, len(keys) + 1))
        _strategy_ticks(ax, [labels.get(k, k) for k in keys])
        ax.set_ylabel("Mean real retirement consumption")
        ax.set_ylim(0, float(np.percentile(np.concatenate(data), 99)))
        _title(ax, "Retirement consumption distribution")
        ax.legend(fontsize=7)

        ax = axes[1]
        for i, key in enumerate(keys):
            values = np.sort(data[i])
            grid = np.arange(1, values.size + 1) / values.size
            ax.plot(values, grid, color=_colour(i), linewidth=1.8,
                    label=_flat(labels.get(key, key), 999))
        ax.set_xlim(0, float(np.percentile(np.concatenate(data), 99)))
        ax.set_xlabel("Mean real retirement consumption")
        ax.set_ylabel("Cumulative probability")
        _title(ax, "Retirement consumption CDF")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
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
        fig, axes = _grid(2, 2.8)
        for i, key in enumerate(keys):
            axes[0].plot(grid, curves[key] * 100, color=_colour(i),
                         linewidth=1.8, label=_flat(labels.get(key, key), 999))
        axes[0].set_xlabel("Real retirement consumption target")
        axes[0].set_ylabel("P(consumption < target)  (%)")
        _title(axes[0], "Shortfall probability")
        axes[0].legend(fontsize=7)

        base = curves.get(reference)
        if base is not None:
            for i, key in enumerate(keys):
                if key == reference:
                    continue
                axes[1].plot(grid, (curves[key] - base) * 100, color=_colour(i),
                             linewidth=1.8, label=_flat(labels.get(key, key), 999))
            axes[1].axhline(0, color="black", linewidth=1.0)
        axes[1].set_xlabel("Real retirement consumption target")
        axes[1].set_ylabel(_wrap(
            f"Shortfall gap vs {_flat(labels.get(reference, reference), 999)}"
            " (pp)", 34))
        _title(axes[1], "Excess shortfall probability")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_cec_by_risk_aversion(table: pd.DataFrame, directory: str | Path,
                              name: str = "fig09_cec_by_risk_aversion") -> Path:
    """Grouped bars of certainty equivalent consumption by risk aversion."""
    cec_cols = [c for c in table.columns if c.startswith("cec_crra_gamma")]
    ez_cols = [c for c in table.columns if c.startswith("cec_ez_")]
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 2.8)
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
            _strategy_ticks(ax, table["label"])
            ax.set_ylabel("CEC (multiples of initial annual income)")
            _title(ax, title)
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_wealth_fan(outcomes: Mapping[str, Any], labels: Mapping[str, str],
                    ages: np.ndarray, directory: str | Path,
                    keys: Sequence[str] = ("balanced_all_equity",
                                           "target_date_fund", "sixty_forty"),
                    name: str = "fig10_wealth_fan") -> Path:
    """Percentile fan of the real wealth trajectory for selected strategies."""
    present = [k for k in keys if k in outcomes]
    with plt.rc_context(STYLE):
        fig, axes = _grid(len(present), 2.6, max_cols=len(present),
                          span_last=False)
        for ax in axes[1:]:
            ax.sharey(axes[0])
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
            _title(ax, labels.get(key, key), fontsize=8.5)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Real financial wealth\n(multiples of initial income)")
        fig.suptitle("Wealth trajectories: median with 25-75 and 5-95 percentile bands",
                     fontsize=9)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save(fig, directory, name)


def plot_ruin_probability(table: pd.DataFrame, directory: str | Path,
                          name: str = "fig11_ruin_probability") -> Path:
    """Probability of exhausting financial wealth before age 93."""
    ordered = table.sort_values("prob_ruin")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.5))
        colours = [_colour(i) for i in range(len(ordered))]
        ax.barh([_flat(v, 22) for v in ordered["label"]],
                ordered["prob_ruin"] * 100, color=colours)
        for y, value in enumerate(ordered["prob_ruin"] * 100):
            ax.text(value + 0.6, y, f"{value:.1f}%", va="center", fontsize=7)
        ax.set_xlabel("Probability of wealth depletion before age 93 (%)")
        _title(ax, "Ruin probability under the 4% real withdrawal rule")
        fig.tight_layout()
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
        fig, axes = _grid(2, 2.8)
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
            ax.legend(fontsize=7)
        _title(axes[0], "Home bias is costly")
        _title(axes[1], "More equity is better, at every risk aversion")
        fig.suptitle("Certainty equivalent consumption along the allocation "
                     "dials (dots mark each curve's optimum)", fontsize=9)
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
        fig, axes = _grid(2, 2.8)
        for i, column in enumerate(wide.columns):
            axes[0].plot(wide.index, wide[column], "-o", markersize=3,
                         color=_colour(i), label=_flat(column, 999))
        axes[0].set_xlabel("CRRA risk aversion γ")
        axes[0].set_ylabel("Certainty equivalent consumption")
        _title(axes[0], "CEC falls with risk aversion, but the order holds")
        axes[0].legend(fontsize=7)

        base_label = labels.get(challenger, challenger)
        if base_label in wide.columns:
            for i, column in enumerate(wide.columns):
                if column == base_label:
                    continue
                axes[1].plot(wide.index,
                             (wide[base_label] / wide[column] - 1.0) * 100,
                             "-o", markersize=3, color=_colour(i), label=_flat(column, 999))
            axes[1].axhline(0, color="black", linewidth=1.0)
        axes[1].set_xlabel("CRRA risk aversion γ")
        axes[1].set_ylabel(f"CEC advantage of\n{base_label} (%)")
        _title(axes[1], "Advantage stays positive across the whole grid")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_withdrawal_sensitivity(frame: pd.DataFrame, swr: pd.DataFrame,
                                target_ruin: float, directory: str | Path,
                                name: str = "fig14_withdrawal_sensitivity"
                                ) -> Path:
    """Ruin probability against withdrawal rate, plus the implied safe rate."""
    ruin = frame.pivot_table(index="withdrawal_rate", columns="label",
                             values="prob_ruin").sort_index()
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 2.6, width_ratios=[1.0, 1.15])
        for i, column in enumerate(ruin.columns):
            axes[0].plot(ruin.index * 100, ruin[column] * 100, "-o",
                         markersize=3, color=_colour(i),
                         label=_flat(column, 999))
        axes[0].axhline(target_ruin * 100, color="black", linestyle="--",
                        linewidth=1.0, label=f"{target_ruin:.0%} ruin target")
        axes[0].axvline(4.0, color="grey", linestyle=":", linewidth=1.0)
        # Low on the axis: the legend takes the top-left corner.
        axes[0].text(4.05, 30, 'the "4% rule"', fontsize=6.5, color="grey")
        axes[0].set_xlabel("Real withdrawal rate (% of wealth at retirement)")
        axes[0].set_ylabel("Probability of ruin before age 93 (%)")
        _title(axes[0], "Ruin probability by withdrawal rate")
        axes[0].legend(fontsize=5.5, loc="upper left", labelspacing=0.3,
                       handlelength=1.4)

        column = [c for c in swr.columns if c.startswith("safe_withdrawal_rate")][0]
        ordered = swr.sort_values(column)
        axes[1].barh([_flat(v, 22) for v in ordered["label"]],
                     ordered[column] * 100,
                     color=[_colour(i) for i in range(len(ordered))])
        axes[1].axvline(4.0, color="grey", linestyle=":", linewidth=1.2)
        axes[1].text(4.05, -0.4, '4% rule', fontsize=7, color="grey")
        for y, value in enumerate(ordered[column] * 100):
            axes[1].text(value + 0.05, y, f"{value:.2f}%", va="center",
                         fontsize=7)
        axes[1].set_xlabel(f"Withdrawal rate giving a {target_ruin:.0%} "
                           "ruin probability (%)")
        _title(axes[1], "Safe withdrawal rate by strategy")
        axes[1].tick_params(axis="y", labelsize=6.5)
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
    with plt.rc_context(STYLE):
        fig, flat, holes = _grid(len(items), 2.5, span_last=False,
                                 spare=True)
        for ax, (title, frame, column) in zip(flat, items):
            wide = frame.pivot_table(index=column, columns="label",
                                     values=metric).sort_index()
            for i, series in enumerate(wide.columns):
                ax.plot(wide.index, wide[series], "-o", markersize=3,
                        color=_colour(i), label=series)
            ax.set_xlabel(title)
            ax.set_ylabel("CEC")
        handles, labels = flat[0].get_legend_handles_labels()
        fig.suptitle("Certainty equivalent consumption across planning "
                     "parameters (γ = 5)", fontsize=9)
        if holes:
            # The legend goes in the empty cell rather than under the grid,
            # which would otherwise leave the hole as white space.
            box = fig.add_subplot(holes[0])
            box.axis("off")
            box.legend(handles, [_flat(v, 26) for v in labels],
                       loc="center", fontsize=6.5)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            fig.legend(handles, labels, loc="lower center", fontsize=6.5,
                       ncol=min(2, len(labels)))
            fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.96))
    return _save(fig, directory, name)


def plot_tornado(tornado: pd.DataFrame, directory: str | Path,
                 incumbent: str = "target_date_fund",
                 name: str = "fig16_tornado") -> Path:
    """Range of the all-equity advantage across every swept assumption."""
    block = tornado[tornado["incumbent"] == incumbent].sort_values("range_pp")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 0.26 * len(block) + 1.7))
        y = np.arange(len(block))
        lo = block["min_advantage_pct"].to_numpy()
        hi = block["max_advantage_pct"].to_numpy()
        median = block["median_advantage_pct"].to_numpy()
        ax.barh(y, hi - lo, left=lo, height=0.6, color=_colour(0), alpha=0.55)
        ax.scatter(median, y, color=_colour(1), zorder=4, s=28, label="median")
        ax.axvline(0, color="black", linewidth=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(block["dimension"], fontsize=8)
        ax.set_xlabel(_wrap(
            "CEC advantage of the 50/50 all-equity portfolio over the "
            f"{_flat(incumbent, 999).lower()} (%)", 70))
        _title(ax, "Every bar sits entirely to the right of zero:\n"
                     "no tested assumption reverses the ranking", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
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
        # Sixteen rule names, several of them fifty characters long: the
        # ranking gets a row of its own so its labels have the whole page
        # width to sit in, and the legend that names the curves is the same
        # list, so the top panel does without one.
        fig, axes = _grid(2, 3.0, max_cols=1)

        ax = axes[0]
        for i, (variant, block) in enumerate(rated.groupby("variant")):
            block = block.sort_values("rate")
            ax.plot(block["rate"] * 100, block[metric], "-o", markersize=3,
                    color=_colour(i), label=_wrap(variant, 34))
            peak = block.loc[block[metric].idxmax()]
            ax.scatter([peak["rate"] * 100], [peak[metric]], color=_colour(i),
                       s=45, zorder=4, edgecolor="white", linewidth=1.0)
        ax.set_xlabel("Spending rate (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "Each rule has an interior optimum (dots mark it)",
               fontsize=8.5)
        # Sixteen entries need a band of their own; without the extra room
        # below the curves the legend sits on top of their right-hand tails.
        low, high = ax.get_ylim()
        ax.set_ylim(low - 0.55 * (high - low), high)
        ax.legend(fontsize=5.5, ncol=3, loc="lower center", labelspacing=0.3,
                  handlelength=1.4, columnspacing=1.0)

        ax = axes[1]
        ordered = best.sort_values(metric)
        colours = [_colour(2) if "Constant real" in v else _colour(0)
                   for v in ordered["variant"]]
        # One line each: sixteen rows in three inches leave about thirteen
        # points per row, and a wrapped label needs more than that.
        ax.barh([str(v) for v in ordered["variant"]], ordered[metric],
                color=colours)
        ax.set_xlim(min(ordered[metric]) * 0.95, max(ordered[metric]) * 1.02)
        for y, value in enumerate(ordered[metric]):
            ax.text(value + 0.002, y, f"{value:.3f}", va="center", fontsize=6.5)
        ax.set_xlabel("CEC at each rule's own optimal rate")
        _title(ax, "Ranking at each rule's optimum", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=6)
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
        fig, axes = _grid(3, 2.9)

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
            _title(ax, title, fontsize=8.5)

        ax = axes[2]
        for _, row in best.iterrows():
            variant = str(row["variant"])
            ax.scatter(row["consumption_volatility"], row[metric],
                       s=40 + 3.0 * float(row["median_bequest"]),
                       color=colour_of.get(variant, _colour(0)), alpha=0.8,
                       edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xlabel("Consumption volatility (sd of log real spending)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "Smoother is not better: the flattest rule\n"
                     "ranks last (marker size = median bequest)", fontsize=8.5)

        handles = [plt.Line2D([], [], color=colour_of[v], linewidth=1.6,
                              label=_wrap(v, 34)) for v in variants]
        fig.legend(handles=handles, loc="lower center", fontsize=6.5,
                   ncol=2)
        fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    return _save(fig, directory, name)


def plot_spending_bequest_pivot(frame: pd.DataFrame, directory: str | Path,
                                name: str = "fig19_spending_bequest_pivot"
                                ) -> Path:
    """How the spending-rule ranking turns on the strength of the bequest motive."""
    wide = frame.pivot_table(index="bequest_weight", columns="variant",
                             values="cec").sort_index()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.5))
        for i, column in enumerate(wide.columns):
            ax.plot(wide.index, wide[column], "-o", markersize=4,
                    color=_colour(i), linewidth=2, label=_flat(column, 999))
        ax.set_xlabel("Weight on the bequest in the utility aggregator")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "The ranking of spending rules turns on the bequest motive")
        ax.legend(fontsize=6.5, loc="lower center", ncol=2, frameon=True,
                  framealpha=0.92, edgecolor="none", labelspacing=0.3)
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
        fig, axes = _grid(3, 2.9)

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
        _title(ax, "The solved path barely glides", fontsize=8.5)
        ax.legend(fontsize=6.5, loc="lower left")

        ax = axes[1]
        for i, gamma in enumerate(gammas):
            block = free[free["risk_aversion"] == gamma].sort_values("age")
            ax.plot(block["age"], block["domestic_share_of_equity"] * 100,
                    "-o", markersize=3, color=_colour(i), linewidth=1.8,
                    label=f"γ = {gamma:g}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Domestic share of the equity sleeve (%)")
        ax.set_ylim(-5, 105)
        _title(ax, "Home bias stays low at every age", fontsize=8.5)
        ax.legend(fontsize=7)

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
        ax.text(26, 1.4, "±1bp: indistinguishable from flat", fontsize=6.5,
                color="grey")
        ax.set_xlabel("Age")
        ax.set_ylabel("Cost of forcing 100% equity here (bp)")
        _title(ax, "Only the retirement date is worth anything", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_glide_comparison(comparison: pd.DataFrame, trace: pd.DataFrame,
                          directory: str | Path,
                          name: str = "fig21_glide_comparison") -> Path:
    """What the solved schedules buy, and how the search converged."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 2.8, width_ratios=[1.25, 1.0])

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
        ax.set_yticklabels([_flat(v, 22) for v in names], fontsize=7)
        ax.set_xlabel("Certainty equivalent consumption")
        _title(ax, "Solved schedules against fixed benchmarks")
        ax.legend(fontsize=7)

        ax = axes[1]
        for i, gamma in enumerate(sorted(trace["gamma"].unique())):
            block = trace[trace["gamma"] == gamma].sort_values("sweep")
            ax.plot(block["sweep"], block["cec"], "-o", color=_colour(i),
                    label=f"γ = {gamma:g}")
        ax.set_xlabel("Coordinate-ascent sweep")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "Convergence")
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_retirement_anchor(anchor: pd.DataFrame, retire_age: int,
                           directory: str | Path,
                           name: str = "fig22_retirement_anchor") -> Path:
    """Does the dip at retirement survive a spending rule with no anchor?"""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 3.6))
        for i, rule in enumerate(sorted(anchor["rule"].unique())):
            block = anchor[anchor["rule"] == rule].sort_values("age")
            ax.plot(block["age"], block["equity_share"] * 100, "-o",
                    markersize=3.5, color=_colour(i), linewidth=1.8,
                    label=rule)
        ax.axvline(retire_age, color="black", linestyle="--", linewidth=1.2)
        ax.text(retire_age + 0.4, 4, "retirement", fontsize=7)
        ax.set_xlabel("Age")
        ax.set_ylabel("Optimal equity share (%)")
        ax.set_ylim(-5, 108)
        _title(ax, "The dip at retirement belongs to the withdrawal rule,\n"
                     "not to the investment problem")
        ax.legend(fontsize=7, loc="lower left")
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
        fig, axes = _grid(3, 2.9)

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
        _title(ax, "Hedging is worth little, and only in small doses" if pays
                     else "Every hedge ratio loses, even when hedging is free",
                     fontsize=8.5)
        ax.legend(fontsize=7)

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
                            ha="center", fontsize=8)
                else:
                    ax.text(x, top * 0.04, "never\nworth it", ha="center",
                            fontsize=8, color=_colour(1))
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{t:.0f}%" for t in ticks])
            ax.set_ylim(0, top * 1.22)
            ax.set_xlabel("Share of the international sleeve hedged")
            ax.set_ylabel("Break-even annual hedging cost (bp)")
            _title(ax, "What you could afford to pay", fontsize=8.5)
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
                        xytext=(-4, 5), ha="right", fontsize=7)
            ax.set_xlabel("Share of the international sleeve hedged (%)")
            ax.set_ylabel("5th-percentile retirement consumption")
            _title(ax, "The loss lands in the left tail,\nwhich is what the "
                         "certainty equivalent weighs", fontsize=8.5)

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
        _title(ax, "Why: hedging cuts standalone risk but\n"
                     "raises correlation with the home market", fontsize=8.5)
        handles = (ax.get_legend_handles_labels()[0]
                   + ax2.get_legend_handles_labels()[0])
        labels = (ax.get_legend_handles_labels()[1]
                  + ax2.get_legend_handles_labels()[1])
        ax.legend(handles, labels, fontsize=6.5, loc="upper center",
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
        fig, axes = _grid(3, 2.9)

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
        ax.set_xticklabels([_variant(v) for v in labels], rotation=40,
                           ha="right", fontsize=6.5)
        ax.set_ylabel("Retirement age")
        _title(ax, "When a wealth trigger actually retires people",
                     fontsize=8.5)
        ax.legend(fontsize=7)

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
                           else f"floor at {floor:.0%} of\naverage earnings"))
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(_variant(v), 18) for v in variants],
                           fontsize=6.5)
        ax.set_xlabel(f"CEC vs {_variant(baseline)} (%)")
        _title(ax, "Most of the flexibility premium is the\n"
                     "model's missing safety net", fontsize=8.5)
        ax.legend(fontsize=6, loc="lower left")

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
        ax.set_xlabel(_wrap("Annualised real return over the decade "
                            "around retirement (%)", 42))
        ax.set_ylabel("Median real retirement consumption", color=_colour(0))
        _title(ax, "The retirement-date lottery", fontsize=8.5)
        handles = (ax.get_legend_handles_labels()[0]
                   + ax2.get_legend_handles_labels()[0])
        labs = (ax.get_legend_handles_labels()[1]
                + ax2.get_legend_handles_labels()[1])
        ax.legend(handles, labs, fontsize=7, loc="upper left")
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
        fig, axes = _grid(3, 2.9)

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
        _title(ax, "Save least when young, most in peak-earning years",
                     fontsize=8.5)
        ax.legend(fontsize=7)

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
        _title(ax, "The model cannot identify the level\n"
                     "(dots mark each optimum)", fontsize=8.5)
        ax.legend(fontsize=7)

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
        ax.set_ylabel("CEC vs the same rule, unconditioned (%)")
        _title(ax, "Conditioning on your own position beats\n"
                     "conditioning on the market", fontsize=8.5)
        ax.legend(fontsize=7)
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
        fig, axes = _grid(2, 3.0)

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
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "How hard to lean, on one comparable scale", fontsize=8.5)
        ax.legend(fontsize=7, title="Gap measured as", title_fontsize=7)

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
        _title(ax, "The policy each form implies, at its own optimum",
                     fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)



def plot_asymmetry(asymmetry: pd.DataFrame, directory: str | Path,
                   name: str = "fig27_savings_asymmetry") -> Path:
    """Saving more when behind and saving less when ahead, priced separately."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 2.9)

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
                            fontsize=6.5,
                            color="black" if abs(value) < 0.6 * limit else "white")
        ax.set_xlabel("Response when ahead of target (k)")
        ax.set_ylabel("Response when behind target (k)")
        _title(ax, "Value of conditioning (%), by which half is switched on\n"
                     "(★ marks the best pair)", fontsize=8.5)
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
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "Which half of the rule earns its keep", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_signal_race(best: pd.DataFrame, race: pd.DataFrame,
                     combination: pd.DataFrame, directory: str | Path,
                     name: str = "fig28_savings_signal_race") -> Path:
    """Every candidate signal, swept over the same sensitivity grid."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(3, 2.8)

        ax = axes[0]
        labels = list(best["signal_label"])
        values = best["matched_value_pct"].to_numpy(dtype=float)
        colours = [_colour(0) if v >= 0 else _colour(1) for v in values]
        bars = ax.barh(range(len(values)), values, color=colours, height=0.62)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([_wrap(v, 26) for v in labels], fontsize=6)
        span = max(float(np.abs(values).max()), 1e-9)
        for bar, value in zip(bars, values):
            offset = 0.03 * span * (1 if value >= 0 else -1)
            ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                    f"{value:+.2f}%", va="center",
                    ha="left" if value >= 0 else "right", fontsize=7)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlim(min(0.0, values.min() - 0.25 * span),
                    max(0.0, values.max() + 0.30 * span))
        ax.set_xlabel("CEC vs a matched constant rate (%)")
        _title(ax, "Best of each signal, at its own best sensitivity",
               fontsize=8.5)
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
        ax.set_xlabel("Sensitivity of the rate to the signal")
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "How sharply the leaders peak", fontsize=8.5)
        ax.legend(fontsize=5.5, loc="upper left",
                  labelspacing=0.3, handlelength=1.4)

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
                            fontsize=6.5,
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
        _title(ax, "The two leaders, layered\n(★ marks the best pair)",
                     fontsize=8.5)
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
        fig, axes = _grid(2, 3.0)

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
                    textcoords="offset points", xytext=(4, 6), fontsize=7,
                    color=_colour(1))
        ax.axhline(0, color="black", linewidth=1.2)
        for _, row in block.iterrows():
            if row["width"] in (0.05, 0.03):
                ax.annotate(f"±{row['width']:.0%}: {row['matched_value_pct']:+.2f}%",
                            (row["width"] * 100, row["matched_value_pct"]),
                            textcoords="offset points", xytext=(6, -14),
                            fontsize=7)
        ax.set_xlabel(_wrap("How far the rate may move from its average "
                            "(± points)", 34))
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "Value of conditioning, by how far the\n"
               "contribution may move", fontsize=8.5)

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
        _title(ax, "What the unconstrained rule actually asks for",
                     fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_value_distribution(quantiles: pd.DataFrame, by_gamma: pd.DataFrame,
                            directory: str | Path,
                            name: str = "fig30_savings_value_distribution") -> Path:
    """Whether conditioning raises the middle or lifts the bottom."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

        ax = axes[0]
        block = quantiles.sort_values("quantile")
        ax.plot(block["quantile"] * 100, block["gain_pct"], marker="o",
                markersize=6, color=_colour(0), linewidth=2.2)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Percentile of retirement consumption")
        ax.set_ylabel("Change vs no conditioning (%)")
        _title(ax, "Where in the distribution the gain lands", fontsize=8.5)
        for _, row in block.iloc[[0, -1]].iterrows():
            ax.annotate(f"p{int(round(row['quantile'] * 100))}: "
                        f"{row['gain_pct']:+.1f}%",
                        (row["quantile"] * 100, row["gain_pct"]),
                        textcoords="offset points", xytext=(8, -14),
                        ha="left" if row["quantile"] < 0.5 else "right",
                        fontsize=7)

        ax = axes[1]
        for i, gamma in enumerate(sorted(by_gamma["risk_aversion"].unique())):
            gblock = by_gamma[by_gamma["risk_aversion"] == gamma] \
                .sort_values("sensitivity")
            ax.plot(gblock["sensitivity"], gblock["matched_value_pct"],
                    marker=_marker(i), markersize=6, color=_colour(i),
                    linewidth=2.0, label=f"γ = {gamma:g}")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Sensitivity of the savings rate to the funded ratio")
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "Who wants it, and how much", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_when_it_matters(windows: pd.DataFrame, activity: pd.DataFrame,
                         directory: str | Path,
                         name: str = "fig31_savings_when_it_matters") -> Path:
    """Which years of a career the balance is worth reading in."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

        ax = axes[0]
        block = windows.sort_values("matched_value_pct")
        values = block["matched_value_pct"].to_numpy(dtype=float)
        colours = [_colour(0) if v >= 0 else _colour(1) for v in values]
        bars = ax.barh(range(len(values)), values, color=colours, height=0.6)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([f"ages {w}" for w in block["window"]], fontsize=7)
        span = max(float(np.abs(values).max()), 1e-9)
        for bar, value in zip(bars, values):
            ax.text(value + 0.03 * span * (1 if value >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%",
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=7)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlim(min(0.0, values.min() - 0.25 * span),
                    max(0.0, values.max() + 0.30 * span))
        ax.set_xlabel("CEC vs a matched constant rate (%)")
        _title(ax, "Conditioning switched on only for these ages",
                     fontsize=8.5)
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
        ax.set_ylabel("Deviation from the base rate (pp)")
        _title(ax, "How active the rule is at each age", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_target_choice(targets: pd.DataFrame, paths: pd.DataFrame,
                       directory: str | Path,
                       name: str = "fig32_savings_target_choice") -> Path:
    """Does the target have to be right, and does aiming higher help?"""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

        ax = axes[0]
        for i, target in enumerate(sorted(targets["target"].unique())):
            block = targets[targets["target"] == target].sort_values("factor")
            ax.plot(block["factor"], block["matched_value_pct"],
                    marker=_marker(i), markersize=6, color=_colour(i),
                    linewidth=2.0, label=target)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.axvline(1.0, color="grey", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Target scaled by")
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "Aiming higher than the median path", fontsize=8.5)
        ax.legend(fontsize=7)

        ax = axes[1]
        for i, target in enumerate(sorted(paths["target"].unique())):
            block = paths[paths["target"] == target].sort_values("age")
            ax.plot(block["age"], block["multiple"], marker=_marker(i),
                    markersize=4, color=_colour(i), linewidth=2.0,
                    markevery=4, label=target)
        ax.set_xlabel("Age")
        ax.set_ylabel("Wealth as a multiple of current income")
        _title(ax, "What each target actually asks for", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_accumulation_interactions(by_strategy: pd.DataFrame,
                                   by_income: pd.DataFrame,
                                   directory: str | Path,
                                   name: str = "fig33_savings_interactions") -> Path:
    """What the value of reading the balance depends on."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

        ax = axes[0]
        block = by_strategy.sort_values("matched_value_pct")
        values = block["matched_value_pct"].to_numpy(dtype=float)
        colours = [_colour(0) if v >= 0 else _colour(1) for v in values]
        bars = ax.barh(range(len(values)), values, color=colours, height=0.6)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([_flat(v, 22) for v in block["strategy"]],
                           fontsize=7)
        span = max(float(np.abs(values).max()), 1e-9)
        for bar, value in zip(bars, values):
            ax.text(value + 0.03 * span * (1 if value >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2, f"{value:+.2f}%",
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=7)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlim(min(0.0, values.min() - 0.25 * span),
                    max(0.0, values.max() + 0.30 * span))
        ax.set_xlabel("CEC vs a matched constant rate (%)")
        _title(ax, "Value of conditioning, by what the money is invested in",
               fontsize=8.5)
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
                        ha="center", fontsize=7)
        # Room either side for the point labels, which sit over the ends.
        ax.margins(x=0.14, y=0.18)
        ax.set_xlabel("Labour-income shock volatility, relative to baseline")
        ax.set_ylabel("CEC vs a matched constant rate (%)")
        _title(ax, "Value of conditioning, by how risky the pay cheque is",
               fontsize=8.5)
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
        fig, axes = _grid(len(gammas) + 1, 2.7)

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
            ax.annotate("retires", (retire_age, 101), fontsize=7,
                        ha="center", va="bottom")
            ax.set_xlim(block["age"].min(), block["age"].max())
            ax.set_ylim(0, 100)
            ax.set_xlabel("Age")
            ax.set_ylabel("Portfolio weight (%)")
            _title(ax, f"Solved allocation, γ = {gamma:g}", fontsize=8.5)
            ax.grid(False)
        axes[0].legend(fontsize=7, loc="lower left", framealpha=0.9,
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
                    textcoords="offset points", xytext=(4, 5), fontsize=7)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("Age")
        ax.set_ylabel("Cost of resetting that age (bp)")
        _title(ax, "What each age's allocation is worth", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_allocation_comparison(comparison: pd.DataFrame, phases: pd.DataFrame,
                               directory: str | Path,
                               name: str = "fig35_allocation_comparison") -> Path:
    """The solved schedule against the benchmarks, and its phase averages."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

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
        ax.set_yticklabels([_flat(s, 22) for s in strategies], fontsize=7)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Gap to the best schedule at that risk aversion (%)")
        _title(ax, "The solved schedule against the benchmarks", fontsize=8.5)
        ax.grid(axis="y", alpha=0.0)
        ax.legend(fontsize=7, loc="lower left")

        ax = axes[1]
        assets = list(ASSET_NAMES)
        labels = [f"γ={r.risk_aversion:g}\n{r.phase}"
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
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Average weight (%)")
        _title(ax, "Average solved weights by phase", fontsize=8.5)
        ax.grid(axis="x", alpha=0.0)
        ax.legend(fontsize=5.5, ncol=2, loc="lower center", frameon=True,
                  framealpha=0.92, edgecolor="none", labelspacing=0.3,
                  handlelength=1.2, columnspacing=1.0)
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
        fig, axes = _grid(2, 2.9)

        ax = axes[0]
        for i, spread in enumerate(sorted(sweep["spread"].unique())):
            block = sweep[sweep["spread"] == spread].sort_values("leverage")
            ax.plot(block["leverage"], block["vs_unlevered_pct"],
                    marker=_marker(i), markersize=5, color=_colour(i),
                    linewidth=1.9, label=f"{spread:.1%} over bills")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Leverage ratio")
        ax.set_ylabel("CEC vs the unlevered portfolio (%)")
        _title(ax, "The value of borrowing, by what it costs", fontsize=8.5)
        ax.legend(fontsize=7, title="Annual spread", title_fontsize=7)

        ax = axes[1]
        block = optimal.sort_values("spread")
        ax.plot(block["spread"] * 100, block["leverage"], marker="o",
                markersize=7, color=_colour(0), linewidth=2.2,
                drawstyle="steps-post")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
        # Left end: the right end is where the step function lands on the line
        # and the point labels already crowd it.
        ax.annotate("unlevered", (float(block["spread"].min()) * 100, 1.0),
                    textcoords="offset points", xytext=(4, 6), fontsize=7,
                    ha="left")
        for _, row in block.iterrows():
            ax.annotate(f"{row['vs_unlevered_pct']:+.2f}%",
                        (row["spread"] * 100, row["leverage"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=6.5)
        # Headroom for the point labels, which otherwise reach the title.
        ax.margins(y=0.18)
        ax.set_xlabel("Annual borrowing spread over the real bill rate (%)")
        ax.set_ylabel("Optimal leverage ratio")
        _title(ax, "Optimal leverage collapses as credit gets dearer",
               fontsize=8.5)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_leverage_detail(detail: pd.DataFrame, schedule: pd.DataFrame,
                         directory: str | Path,
                         name: str = "fig37_leverage_detail") -> Path:
    """What leverage does to the shape of the outcome, and to its age profile."""
    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

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
        ax.set_ylabel("Change vs unlevered (%), symlog")
        _title(ax, "Leverage widens both tails", fontsize=8.5)
        ax.legend(fontsize=7, loc="upper left")

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
        _title(ax, "Solving a leverage ratio for every age\n"
                     "(faint: raw solution, bold: 5-year mean)", fontsize=8.5)
        ax.legend(fontsize=7, title="Annual spread", title_fontsize=7)
        fig.tight_layout()
    return _save(fig, directory, name)


# ---------------------------------------------------------------------------
# Step 14 - where the data comes from
# ---------------------------------------------------------------------------
def plot_provenance(era: pd.DataFrame, contamination: pd.DataFrame,
                    tail: pd.DataFrame, countries: pd.DataFrame,
                    directory: str | Path,
                    name: str = "fig38_data_provenance") -> Path:
    """Does the last stretch of every equity series look like the rest of it?

    The same test read two ways. Left: each country's tail standard deviation
    as a ratio of its own long-run standard deviation, so one bar per country
    and a line at parity. Right: the two standard deviations against each
    other with the 45-degree line, which turns "every country is smoother"
    from a claim into something the eye checks in one pass.

    ``era`` and ``contamination`` are accepted for interface stability and are
    not drawn: both describe the mix of recorded and generated data, and the
    panel they describe contains nothing generated.
    """
    block = tail.dropna(subset=["ratio"]).copy()
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0, width_ratios=[1.15, 1.0])

        ax = axes[0]
        ordered = block.sort_values("ratio")
        y = np.arange(len(ordered))
        ax.barh(y, ordered["ratio"], color=_colour(1), height=0.66)
        ax.axvline(1.0, color="black", linewidth=1.3)
        ax.annotate("equal variance", (1.0, len(ordered) - 0.4),
                    textcoords="offset points", xytext=(5, 0), fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["iso"], fontsize=6.5)
        ax.set_xlabel("Tail s.d. \u00f7 long-run s.d. (real equity returns)")
        _title(ax, "Every country's final years are smoother\n"
                     "than its own history", fontsize=8.5)
        ax.grid(axis="y", alpha=0.0)

        ax = axes[1]
        ref = block["sd_reference"] * 100
        late = block["sd_tail"] * 100
        top = float(max(ref.max(), late.max())) * 1.08
        ax.plot([0, top], [0, top], color="black", linewidth=1.2,
                label="equal variance")
        ax.scatter(ref, late, s=52, color=_colour(1), edgecolor="white",
                   linewidth=0.6, zorder=3)
        # Ordered by the x coordinate and then alternated above and below the
        # marker: the cluster in the middle of this panel is tight enough that
        # a single band of labels reads as one word.
        for k, (_, row) in enumerate(block.sort_values("sd_reference")
                                     .iterrows()):
            x = float(row["sd_reference"]) * 100
            # Labels flip to the left near the right edge so none is clipped.
            flip = x > top * 0.82
            above = bool(k % 2)
            ax.annotate(str(row["iso"]), (x, float(row["sd_tail"]) * 100),
                        textcoords="offset points",
                        xytext=(-6 if flip else 5, 4 if above else -8),
                        ha="right" if flip else "left",
                        va="bottom" if above else "top",
                        fontsize=6, color="0.35")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("Long-run standard deviation (%)")
        ax.set_ylabel("S.d. over the final years (%)")
        _title(ax, "Every point sits below the line,\n"
                     "which one country alone would not do", fontsize=8.5)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_housing(audit: pd.DataFrame, directory: str | Path,
                 name: str = "fig39_housing_smoothing") -> Path:
    """How much of the housing series' apparent free lunch is measurement.

    The audit behind the four-asset headline: how far the published index
    understates the risk an owner bears, which is what has to be corrected
    before housing can be compared with a traded asset at all.

    Left: risk and return by country, housing as published and again with its
    own smoothing undone, against that country's equity. Right: the lag-one
    autocorrelation that does the work, country by country.
    """
    block = audit.sort_values("sd").reset_index(drop=True)
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0, width_ratios=[1.25, 1.0])

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
        _title(ax, "Housing earns equity-like returns at a fraction of the\n"
                     "risk; de-smoothing closes part of that gap, not all",
                     fontsize=8.5)
        ax.legend(fontsize=7, loc="lower right")

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
        ax.set_yticklabels(order["iso"], fontsize=6.5)
        ax.set_xlabel("First-order autocorrelation of real returns")
        _title(ax, "Housing returns are persistent where\n"
                     "equity returns are not", fontsize=8.5)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.0)
        fig.tight_layout()
    return _save(fig, directory, name)


def plot_valuation(predictive: pd.DataFrame, buckets: pd.DataFrame,
                   advantage: pd.DataFrame, domestic: np.ndarray,
                   blended: np.ndarray, position: Mapping[str, Any],
                   directory: str | Path,
                   name: str = "fig40_starting_valuation",
                   boundaries: pd.DataFrame | None = None) -> Path:
    """What the starting yield predicts, and what it does to a lifetime.

    Three readings of one variable: that it forecasts returns at all, where
    the present sits in its distribution, and whether the headline ranking
    survives conditioning on it.
    """
    panels = 4 if boundaries is not None and len(boundaries) else 3
    with plt.rc_context(STYLE):
        fig, axes = _grid(panels, 2.9)

        ax = axes[0]
        block = predictive.sort_values("horizon_years")
        x = np.arange(len(block))
        width = 0.38
        ax.bar(x - width / 2, block["forward_return_expensive"] * 100, width,
               color=_colour(1), label="started expensive (low yield)")
        ax.bar(x + width / 2, block["forward_return_cheap"] * 100, width,
               color=_colour(0), label="started cheap (high yield)")
        for i, row in enumerate(block.itertuples()):
            ax.text(i, max(row.forward_return_cheap,
                           row.forward_return_expensive) * 100 + 0.15,
                    f"{row.correlation:+.2f}", ha="center", fontsize=7,
                    color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}y" for h in block["horizon_years"]])
        ax.set_xlabel("Horizon over which the return is measured")
        ax.set_ylabel("Annualised real equity return (%)")
        pays = bool((block["gap"] > 0).all())
        _title(ax, 
            ("A cheap start pays at every horizon\n"
             "(label = correlation of yield with return)") if pays else
            ("A cheap start does not pay at every horizon\n"
             "(label = correlation of yield with return)"), fontsize=8.5)
        ax.legend(fontsize=6, loc="upper right", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")
        ax.grid(axis="x", alpha=0.0)

        ax = axes[1]
        values = blended[np.isfinite(blended)] * 100
        ax.hist(values, bins=40, color=_colour(2), alpha=0.85)
        here = float(position.get("blended_yield", np.nan)) * 100
        if np.isfinite(here):
            ax.axvline(here, color=_colour(1), linewidth=2.0)
            # On a white plate: the marker line sits at the left edge of the
            # distribution, so the label unavoidably runs over the bars.
            ax.annotate(f"{position.get('iso', '')} "
                        f"{int(position.get('year', 0))}: {here:.1f}%\n"
                        f"{position.get('blended_percentile', float('nan')):.0f}th "
                        f"percentile",
                        (here, ax.get_ylim()[1] * 0.90),
                        textcoords="offset points", xytext=(9, 0),
                        fontsize=7.5, color=_colour(1), va="top",
                        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                                  edgecolor=_colour(1), linewidth=0.7,
                                  alpha=0.92))
        ax.set_xlabel("Blended starting dividend yield (%)")
        ax.set_ylabel("Country-years in the panel")
        _title(ax, "Where a reader starting today sits in\n"
                     "the panel's own distribution", fontsize=8.5)
        ax.grid(axis="x", alpha=0.0)

        # The advantage as bars, the level it is an advantage *over* as a
        # line. Showing only the bars invites the reading that valuation does
        # not matter; it is the line that moves.
        ax = axes[2]
        block = advantage.copy()
        x = np.arange(len(block))
        ax.bar(x, block["advantage_pct"], width=0.55, color=_colour(0),
               label="lead over the glide path (left)")
        for i, row in enumerate(block.itertuples()):
            offset = 0.25 if row.advantage_pct >= 0 else -0.55
            ax.text(i, row.advantage_pct + offset,
                    f"{row.advantage_pct:.1f}%", ha="center", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(block["bucket"], fontsize=8)
        ax.set_xlabel("Valuation the lifetime started at")
        ax.set_ylabel("Certainty-equivalent advantage (%)")
        # Never clip a negative bar out of the frame: an exception to the
        # ranking is exactly what this panel exists to make visible.
        low = min(float(block["advantage_pct"].min()), 0.0)
        high = max(float(block["advantage_pct"].max()), 1.0)
        # Room above the tallest bar for the legend, which has nowhere else
        # to sit in a panel this size.
        pad = max((high - low) * 0.45, 0.5)
        ax.set_ylim(low - pad, high + pad)

        twin = ax.twinx()
        twin.plot(x, block["challenger_cec"], color=_colour(1),
                  marker=_marker(1), linewidth=1.8,
                  label="all-equity CEC level (right)")
        twin.set_ylabel("Certainty-equivalent consumption")
        twin.grid(False)

        holds = bool((block["advantage_pct"] > 0).all())
        spread = float(block["advantage_pct"].max()
                       - block["advantage_pct"].min())
        cec_move = (float(block["challenger_cec"].max())
                    / float(block["challenger_cec"].min()) - 1.0) * 100.0
        _title(ax, 
            (f"The ranking holds everywhere (spread {spread:.1f}pp);\n"
             f"the level it wins at moves {cec_move:.1f}%") if holds else
            ("The ranking does NOT hold at every\n"
             "starting valuation"), fontsize=8.5)
        handles, labels_ = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles + h2, labels_ + l2, fontsize=6,
                  loc="upper left", labelspacing=0.3, handlelength=1.4)
        ax.grid(axis="x", alpha=0.0)

        if panels == 4:
            # The boundaries an investor could have drawn, as they drift. A
            # fixed pair of cut-points is what the pooled split assumes, and
            # the distance between the lines and it is the look-ahead.
            ax = axes[3]
            year = boundaries["year"].to_numpy(dtype=float)
            lo = boundaries["cut_expensive_middling"].to_numpy() * 100
            hi = boundaries["cut_middling_cheap"].to_numpy() * 100
            ax.plot(year, lo, color=_colour(1), linewidth=1.8,
                    label="expensive / middling")
            ax.plot(year, hi, color=_colour(0), linewidth=1.8,
                    label="middling / cheap")
            ax.fill_between(year, lo, hi, color=_colour(2), alpha=0.18,
                            label="the middling third")
            ax.set_xlabel("Year the lifetime begins")
            ax.set_ylabel("Blended dividend yield (%)")
            _title(ax, "The boundaries an investor could\n"
                         "actually have drawn, as they drift", fontsize=8.5)
            ax.legend(fontsize=7, loc="lower left")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_housing_sweep(sweep: pd.DataFrame, audit: pd.DataFrame,
                       raw_sweep: pd.DataFrame | None,
                       break_even: float, directory: str | Path,
                       name: str = "fig41_housing_cost_sweep") -> Path:
    """What housing is worth, and what it costs to make it worth nothing.

    Four readings: how much volatility the de-smoothing restores, how the
    optimal allocation rearranges itself as the holding cost rises, what the
    fifth asset adds to lifetime welfare at each cost, and how much of that
    conclusion is an artefact of leaving the index smoothed.
    """
    five = sweep[sweep["investable_set"] == "five assets"].sort_values(
        "holding_cost")
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. the smoothing the index carries ---------------------------
        ax = axes[0]
        block = audit.sort_values("autocorrelation")
        y = np.arange(len(block))
        ax.barh(y - 0.2, block["sd_raw"] * 100, 0.4, color=_colour(5),
                label="as published")
        ax.barh(y + 0.2, block["sd_desmoothed"] * 100, 0.4, color=_colour(1),
                label="de-smoothed")
        ax.plot(block["equity_sd"] * 100, y, linestyle="none",
                marker=_marker(2), color="0.2", markersize=5,
                label="domestic equity")
        ax.set_yticks(y)
        ax.set_yticklabels(block["iso"], fontsize=7)
        ax.set_xlabel("Standard deviation of real returns (%)")
        _title(ax, "Undoing the appraisal smoothing")
        # Above the plot rather than inside it: the shortest bars sit at the
        # bottom, which is the only space an inset legend could use.
        ax.legend(fontsize=7, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, -0.30))
        ax.grid(axis="y", visible=False)

        # -- 2. the allocation, cost by cost ------------------------------
        ax = axes[1]
        assets = ("mean_housing", "mean_dom_eq", "mean_intl_eq", "mean_bond",
                  "mean_bill")
        labels = ("Housing", "Domestic equity", "International equity",
                  "Bonds", "Bills")
        x = five["holding_cost"].to_numpy(dtype=float) * 100
        bottom = np.zeros(len(five))
        for i, (column, label) in enumerate(zip(assets, labels)):
            if column not in five.columns:
                continue
            height = five[column].to_numpy(dtype=float) * 100
            ax.bar(x, height, width=0.7, bottom=bottom, label=label,
                   color=_colour(i))
            bottom += height
        ax.set_xlabel("Annual holding cost on housing (%)")
        ax.set_ylabel("Weight in the optimal portfolio (%)")
        _title(ax, "What the optimum holds")
        ax.legend(fontsize=7, ncol=2, loc="upper center")
        ax.set_ylim(0, 128)

        # -- 3. what the fifth asset is worth -----------------------------
        ax = axes[2]
        ax.plot(x, five["advantage_pct"], marker=_marker(0),
                color=_colour(0), label="de-smoothed")
        if raw_sweep is not None and len(raw_sweep):
            other = raw_sweep[
                raw_sweep["investable_set"] == "five assets"].sort_values(
                "holding_cost")
            ax.plot(other["holding_cost"] * 100, other["advantage_pct"],
                    marker=_marker(1), linestyle="--", color=_colour(4),
                    label="raw (still smoothed)")
        ax.axhline(0.0, color="0.3", linewidth=1.0)
        if np.isfinite(break_even):
            ax.axvline(break_even * 100, color=_colour(1), linestyle=":",
                       linewidth=1.4)
            ax.text(break_even * 100, ax.get_ylim()[1] * 0.92,
                    f"  housing drops out\n  at {break_even:.1%}", fontsize=7,
                    color=_colour(1), va="top")
        ax.set_xlabel("Annual holding cost on housing (%)")
        ax.set_ylabel("Gain over the four-asset optimum (%)")
        _title(ax, "What adding housing is worth")
        ax.legend(fontsize=7)

        # -- 4. where the weight comes from -------------------------------
        ax = axes[3]
        if "mean_housing" in five.columns:
            ax.plot(x, five["mean_housing"] * 100, marker=_marker(0),
                    color=_colour(0), label="Housing")
            for i, (column, label) in enumerate(
                    (("equity", "Equity (both legs)"),
                     ("fixed_income", "Bonds and bills")), start=1):
                if column in five.columns:
                    ax.plot(x, five[column] * 100, marker=_marker(i),
                            color=_colour(i), label=label)
        ax.set_xlabel("Annual holding cost on housing (%)")
        ax.set_ylabel("Weight in the optimal portfolio (%)")
        _title(ax, "What housing displaces")
        ax.legend(fontsize=7)

        fig.tight_layout()
        return _save(fig, directory, name)


def plot_mortgage(sweep: pd.DataFrame, schedule: pd.DataFrame,
                  curve: pd.DataFrame, break_even: float,
                  directory: str | Path,
                  name: str = "fig42_mortgage",
                  profile: pd.DataFrame | None = None) -> Path:
    """How much of the house to borrow, at what age, and at what price.

    Four readings: the shape of the loan-to-value decision at one price, how
    that decision varies with the price of credit, what the borrowing is worth
    against an unlevered control, and how often the borrower's right to walk
    away is what makes the answer work.
    """
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. the solved schedule, age by age ---------------------------
        ax = axes[0]
        working = schedule[schedule["phase"] == "working"]
        retired = schedule[schedule["phase"] == "retired"]
        ax.step(schedule["age"], schedule["lvr"] * 100, where="mid",
                color=_colour(0), linewidth=1.9, label="solved LVR")
        if len(retired):
            ax.axvspan(float(retired["age"].min()), float(schedule["age"].max()),
                       color="0.85", alpha=0.45, zorder=0)
            ax.text(float(retired["age"].min()) + 0.6, 4.0, "retired",
                    fontsize=7, color="0.35")
        for block, label, colour in ((working, "working mean", 1),
                                     (retired, "retired mean", 2)):
            if len(block):
                ax.axhline(float(block["lvr"].mean()) * 100, color=_colour(colour),
                           linestyle="--", linewidth=1.2, label=label)
        ax.set_xlabel("Age")
        ax.set_ylabel("Loan-to-value ratio (%)")
        ax.set_ylim(0, 88)
        if profile is not None and len(profile):
            # Shade what the schedule is worth. Where the bars are invisible
            # the search moved the ratio for free, and the line above them
            # carries no information.
            band = ax.twinx()
            band.bar(profile["age"], profile["cost_of_resetting_bp"],
                     width=0.9, color="0.72", alpha=0.55, zorder=0,
                     label="value of this age's choice (right)")
            band.set_ylabel("Cost of resetting one age (bp)")
            band.grid(False)
            band.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            _title(ax, "How much of the house to borrow, by age\n"
                         "(grey = what each age is actually worth)",
                         fontsize=8.5)
        else:
            _title(ax, "How much of the house to borrow,\nby age",
                         fontsize=8.5)
        ax.legend(fontsize=7, loc="lower left")

        # -- 2. the curve at one price ------------------------------------
        ax = axes[1]
        ax.plot(curve["lvr"] * 100, curve["cec"], marker=_marker(0),
                color=_colour(0))
        best = curve.loc[curve["cec"].idxmax()]
        ax.axvline(float(best["lvr"]) * 100, color=_colour(1), linestyle=":",
                   linewidth=1.4)
        ax.text(float(best["lvr"]) * 100, float(curve["cec"].min()),
                f"  best flat LVR {float(best['lvr']):.0%}", fontsize=7,
                color=_colour(1), va="bottom")
        ax.set_xlabel("Loan-to-value ratio, held for life (%)")
        ax.set_ylabel("Certainty-equivalent consumption")
        top = float(curve["lvr"].max())
        at_corner = float(best["lvr"]) >= top - 1e-9
        at_zero = float(best["lvr"]) <= 1e-9
        _title(ax, 
            ("Held flat, the ratio runs to the cap:\n"
             "the ceiling binds, it is not an optimum") if at_corner else
            ("Held flat, no borrowing is worth\ntaking at this price")
            if at_zero else
            ("The decision is interior,\nnot a corner"), fontsize=8.5)

        # -- 3. what the price of credit does ------------------------------
        ax = axes[2]
        x = sweep["spread"].to_numpy(dtype=float) * 100
        ax.plot(x, sweep["mean_lvr"] * 100, marker=_marker(0),
                color=_colour(0), label="mean LVR")
        if "lvr_working" in sweep.columns:
            ax.plot(x, sweep["lvr_working"] * 100, marker=_marker(1),
                    linestyle="--", color=_colour(1), label="while working")
            ax.plot(x, sweep["lvr_retired"] * 100, marker=_marker(2),
                    linestyle="--", color=_colour(2), label="in retirement")
        if np.isfinite(break_even):
            ax.axvline(break_even * 100, color="0.4", linestyle=":",
                       linewidth=1.4)
            ax.text(break_even * 100, 4.0, f"  borrowing stops paying\n"
                    f"  at {break_even:.1%}", fontsize=7, color="0.35")
        ax.set_xlabel("Mortgage spread over the domestic short rate (%)")
        ax.set_ylabel("Optimal loan-to-value ratio (%)")
        _title(ax, "What the price of credit does", fontsize=8.5)
        ax.legend(fontsize=7)

        # -- 4. what it is worth, and what props it up ---------------------
        ax = axes[3]
        ax.plot(x, sweep["gain_vs_unlevered_pct"], marker=_marker(0),
                color=_colour(0), label="gain over no mortgage (left)")
        ax.axhline(0.0, color="0.3", linewidth=1.0)
        ax.set_xlabel("Mortgage spread over the domestic short rate (%)")
        ax.set_ylabel("Gain over an unlevered house (%)")
        twin = ax.twinx()
        twin.plot(x, sweep["negative_equity_share"] * 100, marker=_marker(1),
                  linestyle="--", color=_colour(1),
                  label="path-years in negative equity (right)")
        twin.set_ylabel("Path-years wiped out (%)")
        twin.grid(False)
        _title(ax, "What borrowing buys, and how often\n"
                     "the right to walk away is what pays", fontsize=8.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left",
                  labelspacing=0.3, handlelength=1.4)

        fig.tight_layout()
        return _save(fig, directory, name)


def plot_sleeve_weighting(concentration: pd.DataFrame, ranking: pd.DataFrame,
                          spectrum: pd.DataFrame, directory: str | Path,
                          name: str = "fig43_sleeve_weighting") -> Path:
    """Whether the headline needs an equal-weighted international sleeve.

    Three readings: how concentrated each weighting scheme actually is,
    what each does to every strategy's certainty equivalent, and whether the
    advantage all-international holds over the 50/50 split tracks the
    concentration of the sleeve or the direction it tilts.
    """
    order = list(spectrum["weighting"]) if len(spectrum) else []
    with plt.rc_context(STYLE):
        fig, axes = _grid(3, 2.9)

        # -- 1. concentration through time, one line per scheme ------------
        ax = axes[0]
        ends: List[Tuple[float, float]] = []
        for k, scheme in enumerate(order):
            block = concentration[concentration["weighting"] == scheme] \
                .sort_values("year")
            if not len(block):
                continue
            label = str(block["label"].iloc[0])
            ax.plot(block["year"], block["effective_markets"],
                    color=_colour(k), linewidth=2.0, label=label,
                    linestyle="--" if scheme == "equal" else "-")
            ends.append((float(block["effective_markets"].iloc[-1]),
                         float(block["year"].iloc[-1])))
        # Schemes that end within a market of one another would print their
        # end values on top of each other, so only the extremes are labelled.
        for value, year in ({min(ends), max(ends)} if ends else ()):
            ax.annotate(f"{value:.1f}", xy=(year, value), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5,
                        va="center", color="0.25")
        _title(ax, "Effective number of markets")
        ax.set_xlabel("Year")
        ax.set_ylabel("1 / Herfindahl index")
        ax.set_ylim(0.0, None)
        ax.legend(fontsize=6, loc="center left", labelspacing=0.3,
                  handlelength=1.4)

        # -- 2. the headline under every scheme ----------------------------
        ax = axes[1]
        schemes = [c for c in order if c in ranking.columns]
        if len(ranking) and schemes:
            frame = ranking.sort_values(schemes[-1] if "equal" not in schemes
                                        else "equal")
            y = np.arange(len(frame), dtype=float)
            n = len(schemes)
            height = 0.80 / max(n, 1)
            for k, scheme in enumerate(schemes):
                offset = (k - (n - 1) / 2) * height
                ax.barh(y + offset, frame[scheme], height=height * 0.92,
                        color=_colour(order.index(scheme)),
                        label=LABEL_OF.get(scheme, scheme))
            ax.set_yticks(y)
            ax.set_yticklabels([_flat(v, 24) for v in frame["label"]],
                               fontsize=7)
            ax.set_xlim(0.0, float(frame[schemes].to_numpy().max()) * 1.10)
        _title(ax, "Certainty-equivalent consumption")
        ax.set_xlabel("CEC (annual, real, relative to age-25 income)")
        # Opaque: the legend has to sit over the shortest bars, and the panel
        # has nowhere else to put it.
        ax.legend(fontsize=6, loc="lower right", frameon=True,
                  framealpha=0.92, edgecolor="none", labelspacing=0.3,
                  handlelength=1.4)

        # -- 3. does the gap track concentration, or the tilt? -------------
        ax = axes[2]
        if len(spectrum):
            for k, (_, row) in enumerate(spectrum.iterrows()):
                ax.scatter(float(row["effective_markets"]),
                           float(row["gap_pct"]),
                           s=110, color=_colour(order.index(row["weighting"])),
                           marker=_marker(k), zorder=3,
                           label=str(row["label"]))
                ax.annotate(f"  {str(row['tilts_towards'])}",
                            xy=(float(row["effective_markets"]),
                                float(row["gap_pct"])),
                            xytext=(6, -10), textcoords="offset points",
                            fontsize=6.5, color="0.35")
            ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
            ax.set_xlim(0.0, float(spectrum["effective_markets"].max()) * 1.30)
        _title(ax, "Advantage of all-international over 50/50")
        ax.set_xlabel("Effective number of markets in the sleeve")
        ax.set_ylabel("Gap in certainty-equivalent\nconsumption (%)")
        ax.legend(fontsize=7, loc="lower right")

        fig.tight_layout()
        return _save(fig, directory, name)


def plot_panel_robustness(influence: pd.DataFrame, period: pd.DataFrame,
                          floor: Mapping[str, Any], jack: Mapping[str, Any],
                          directory: str | Path,
                          name: str = "fig44_panel_robustness",
                          profile: pd.DataFrame | None = None) -> Path:
    """Whether the headline rests on any one country, or any one era.

    Three readings: what each country's removal does to the headline gap,
    against the noise a re-seeded run produces on an unchanged panel; the same
    gap on expanding windows of history; and the jackknife interval the
    sixteen-country panel actually supports.
    """
    baseline = float(jack.get("baseline_gap_pct", float("nan")))
    noise = float(floor.get("range_pct", 0.0)) / 2.0
    with plt.rc_context(STYLE):
        n_panels = 4 if profile is not None and len(profile) else 3
        fig, axes = _grid(n_panels, 3.0)

        # -- 1. per-country influence --------------------------------------
        ax = axes[0]
        if len(influence):
            frame = influence.sort_values("shift_pct")
            y = np.arange(len(frame), dtype=float)
            material = frame["shift_pct"].abs() > noise
            ax.barh(y, frame["shift_pct"],
                    color=[_colour(1) if m else "0.72" for m in material],
                    height=0.72)
            if noise > 0:
                ax.axvspan(-noise, noise, color="0.85", alpha=0.55, zorder=0)
                # Above the top bar rather than across it: the band is at
                # zero, which is where the shortest bars are.
                ax.text(0.0, len(frame) - 0.1, "re-seeding noise",
                        fontsize=6.5, color="0.35", ha="center", va="bottom")
                ax.set_ylim(-0.8, len(frame) + 0.6)
            ax.axvline(0.0, color="0.35", linewidth=1.0)
            ax.set_yticks(y)
            ax.set_yticklabels(frame["dropped"], fontsize=6.5)
        _title(ax, "Effect of removing one country")
        ax.set_xlabel("Change in the all-international lead (pp)")

        # -- 2. the gap on expanding windows -------------------------------
        ax = axes[1]
        if len(period):
            expanding = period[~period["window"].str.contains("half")]
            halves = period[period["window"].str.contains("half")]
            x = np.arange(len(expanding), dtype=float)
            ax.plot(x, expanding["gap_pct"], marker=_marker(0),
                    color=_colour(0), linewidth=2.0, markersize=9,
                    label="expanding window")
            for k, (_, row) in enumerate(halves.iterrows()):
                ax.axhline(float(row["gap_pct"]), color=_colour(2 + k),
                           linestyle="--", linewidth=1.6,
                           label=str(row["window"]))
            ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
            ax.set_xticks(x)
            ax.set_xticklabels([w.split("-")[-1] for w in expanding["window"]],
                               fontsize=8)
            ax.set_ylim(min(0.0, float(period["gap_pct"].min()) * 1.2),
                        float(period["gap_pct"].max()) * 1.25)
        _title(ax, "The lead, by how much history you have")
        ax.set_xlabel("Data available through")
        ax.set_ylabel("All-international over 50/50 (pp)")
        ax.legend(fontsize=7, loc="lower right")

        # -- 3. what the panel actually supports ---------------------------
        ax = axes[2]
        lo = float(jack.get("ci_low", float("nan")))
        hi = float(jack.get("ci_high", float("nan")))
        se = float(jack.get("standard_error", float("nan")))
        if np.isfinite(baseline):
            ax.errorbar([0.0], [baseline],
                        yerr=[[baseline - lo], [hi - baseline]],
                        fmt=_marker(0), color=_colour(0), markersize=12,
                        capsize=10, capthick=2.0, elinewidth=2.0)
            ax.axhline(0.0, color=_colour(1), linewidth=1.6, linestyle="--")
            ax.text(0.12, 0.0, " no advantage", fontsize=8, color=_colour(1),
                    va="bottom")
            ax.text(0.12, baseline, f" {baseline:.2f} ± {se:.2f}",
                    fontsize=8.5, va="center", color="0.25")
            ax.text(0.12, hi, f" 95%: [{lo:.2f}, {hi:.2f}]", fontsize=7,
                    va="bottom", color="0.4")
            ax.set_xlim(-0.5, 1.4)
            ax.set_ylim(min(-0.4, lo * 1.4), hi * 1.35)
        ax.set_xticks([])
        _title(ax, "Jackknife interval from 16 countries")
        ax.set_ylabel("All-international over 50/50 (pp)")

        # -- 4. why: what the deletion does to the sleeve's compound return
        if n_panels == 4:
            ax = axes[3]
            merged = profile.merge(influence[["dropped", "shift_pct"]],
                                   left_on="iso", right_on="dropped")
            ax.scatter(merged["sleeve_geometric_delta"], merged["shift_pct"],
                       s=70, color=_colour(0), zorder=3)
            # Only the outliers are named, and they alternate above and below
            # their marker: two countries that moved the lead by the same
            # amount would otherwise print one label over the other.
            labelled = merged[(merged["shift_pct"].abs() > 0.35)
                              | (merged["sleeve_geometric_delta"].abs() > 0.15)]
            for k, (_, row) in enumerate(
                    labelled.sort_values("sleeve_geometric_delta").iterrows()):
                above = bool(k % 2)
                ax.annotate(str(row["iso"]),
                            xy=(float(row["sleeve_geometric_delta"]),
                                float(row["shift_pct"])),
                            xytext=(6, 4 if above else -6),
                            textcoords="offset points",
                            va="bottom" if above else "top",
                            fontsize=6.5, color="0.3")
            # Room at the edges: a label sits to the right of its marker.
            ax.margins(x=0.14, y=0.12)
            ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
            ax.axvline(0.0, color="0.4", linewidth=1.0, linestyle=":")
            _title(ax, "Why: the sleeve's compound return")
            ax.set_xlabel("Change in the sleeve's geometric mean (pp)")
            ax.set_ylabel("Change in the lead (pp)")

        fig.tight_layout()
        return _save(fig, directory, name)


def plot_fees(common: pd.DataFrame, differential: pd.DataFrame,
              anchors: pd.DataFrame, findings: Mapping[str, Any],
              directory: str | Path, name: str = "fig45_fees") -> Path:
    """How small a cost undoes the headline.

    Left: a fee charged on every asset alike, the control. Right: a fee on
    the international sleeve alone, which all-international pays on everything
    and the 50/50 split pays on half — with the expense ratios a real
    investor has actually faced marked on it.
    """
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.1)

        for ax, frame, key, title, xlabel in (
                (axes[0], common, "fee", "A fee on every asset alike",
                 "Annual fee on all four sleeves (bp)"),
                (axes[1], differential, "differential",
                 "A fee on the international sleeve alone",
                 "Extra annual fee on the foreign leg (bp)")):
            if not len(frame):
                continue
            x = frame[key].to_numpy(dtype=float) * 1e4
            y = frame["gap_pct"].to_numpy(dtype=float)
            ax.plot(x, y, marker=_marker(0), color=_colour(0), linewidth=2.0,
                    markersize=8)
            ax.fill_between(x, 0.0, y, where=(y > 0), color=_colour(0),
                            alpha=0.10)
            ax.fill_between(x, 0.0, y, where=(y <= 0), color=_colour(1),
                            alpha=0.14)
            ax.axhline(0.0, color=_colour(1), linewidth=1.6, linestyle="--")
            _title(ax, title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("All-international over 50/50 (pp)")

        # The break-even and the real-world anchors go on the right panel,
        # where the question actually lives.
        ax = axes[1]
        be = float(findings.get("break_even_differential_bp", float("inf")))
        if np.isfinite(be):
            ax.axvline(be, color="0.35", linewidth=1.4, linestyle=":")
            ax.annotate(f"break-even\n{be:.0f} bp", xy=(be, 0.0),
                        xytext=(6, 18), textcoords="offset points",
                        fontsize=8, color="0.25")
        for k, (_, row) in enumerate(anchors.iterrows()):
            bp = float(row["basis_points"])
            if bp > ax.get_xlim()[1]:
                continue
            ax.scatter([bp], [float(row["gap_pct"])], s=90, zorder=4,
                       color=_colour(2 + k), marker=_marker(1 + k),
                       label=f"{str(row['label'])} ({bp:.0f} bp)")
        ax.legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        return _save(fig, directory, name)


def plot_cohorts(census: pd.DataFrame, detail: pd.DataFrame,
                 realised: pd.DataFrame, by_country: pd.DataFrame,
                 directory: str | Path, interval: Mapping[str, Any] | None = None,
                 name: str = "fig46_cohorts") -> Path:
    """The realised record: who is in it, what it paid, and why it is thin."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. how many lifetimes each market actually supports -----------
        ax = axes[0]
        block = census.sort_values("cohorts")
        y = np.arange(len(block), dtype=float)
        ax.barh(y, block["cohorts"], color=_colour(0), height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(block["iso"], fontsize=6.5)
        ax.set_xlabel("Complete lifetimes on the record")
        _title(ax, "A market closure removes a lifetime, not a year")
        ax.grid(axis="y", alpha=0.0)

        # -- 2. the distribution of the realised gap ----------------------
        ax = axes[1]
        gaps = detail["gap_pct"].to_numpy(dtype=float)
        ax.hist(gaps, bins=30, color=_colour(0), alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.axvline(float(np.mean(gaps)), color=_colour(1), linewidth=1.6,
                   linestyle="--")
        share = float((gaps > 0).mean())
        ax.annotate(f"ahead in {share:.0%}\nof cohorts",
                    xy=(0.97, 0.95), xycoords="axes fraction", ha="right",
                    va="top", fontsize=6.5, color="0.3")
        ax.set_xlabel("All-international over 50/50, one realised lifetime (%)")
        ax.set_ylabel("Cohorts")
        _title(ax, "Every lifetime the panel can actually run")

        # -- 3. the long-run legs, which is the mechanism ------------------
        ax = axes[2]
        ax.scatter(realised["domestic_annualised"] * 100,
                   realised["sleeve_annualised"] * 100, s=14,
                   color=_colour(0), alpha=0.55, zorder=3)
        lo = float(min(realised["domestic_annualised"].min(),
                       realised["sleeve_annualised"].min()) * 100) - 0.6
        hi = float(max(realised["domestic_annualised"].max(),
                       realised["sleeve_annualised"].max()) * 100) + 0.6
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2,
                label="equal returns")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Home market, 68-year annualised (%)")
        ax.set_ylabel("The sleeve, same years (%)")
        # Classified, not asserted: the claim is about this panel's record.
        worst_sleeve = float(realised["sleeve_annualised"].min()) * 100
        worst_home = float(realised["domestic_annualised"].min()) * 100
        _title(ax, (f"The sleeve never fell below {worst_sleeve:.1f}%; "
                    f"a home market reached {worst_home:.1f}%")
               if worst_sleeve > worst_home else
               "The worst realised lifetime was the sleeve's")
        ax.legend(fontsize=6.5, loc="lower right")

        # -- 4. by market, with the interval the countries support ---------
        ax = axes[3]
        block = by_country.sort_values("mean_gap_pct")
        y = np.arange(len(block), dtype=float)
        colours = [_colour(0) if v > 0 else _colour(1)
                   for v in block["mean_gap_pct"]]
        ax.barh(y, block["mean_gap_pct"], color=colours, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(block["iso"], fontsize=6.5)
        ax.axvline(0.0, color="black", linewidth=1.2)
        if interval:
            ax.axvspan(float(interval["ci_low"]), float(interval["ci_high"]),
                       color="0.85", alpha=0.55, zorder=0)
            ax.axvline(float(interval["mean_gap_pct"]), color="0.35",
                       linewidth=1.4, linestyle=":")
            ax.annotate("95% over countries", xy=(float(interval["ci_high"]), 0.2),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=6, color="0.35", va="bottom")
        ax.set_xlabel("Mean realised lead (%)")
        ahead = int((block["mean_gap_pct"] > 0).sum())
        _title(ax, (f"All {len(block)} markets favour the sleeve"
                    if ahead == len(block) else
                    f"{ahead} of {len(block)} markets, but not all"))
        ax.grid(axis="y", alpha=0.0)

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_out_of_sample(frame: pd.DataFrame, benchmarks: pd.DataFrame,
                       directory: str | Path,
                       name: str = "fig47_out_of_sample") -> Path:
    """What a solved schedule keeps when it meets history it was not fitted to."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.2)

        # -- 1. in-sample gain against what transferred -------------------
        ax = axes[0]
        labels = [f"{r.family}\n{r.train_window} to {r.test_window}"
                  for r in frame.itertuples()]
        y = np.arange(len(frame), dtype=float)
        height = 0.36
        ax.barh(y + height / 2, frame["in_sample_gain_pct"], height=height,
                color="0.72", label="measured where it was solved")
        ax.barh(y - height / 2, frame["transfer_gain_pct"], height=height,
                color=[_colour(0) if v > 0 else _colour(1)
                       for v in frame["transfer_gain_pct"]],
                label="measured on the other half")
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Gain over the best fixed strategy (%)")
        _title(ax, "The gain is measured where the search ran")
        ax.margins(y=0.10)
        ax.legend(fontsize=6, loc="upper left", frameon=True,
                  framealpha=0.92, edgecolor="none", labelspacing=0.3,
                  handlelength=1.4)
        ax.grid(axis="y", alpha=0.0)

        # -- 2. the fixed strategies on both halves -----------------------
        ax = axes[1]
        windows = list(dict.fromkeys(benchmarks["window"]))
        strategies = list(benchmarks[benchmarks["window"] == windows[0]]
                          .sort_values("cec", ascending=False)["strategy"])
        y = np.arange(len(strategies), dtype=float)
        width = 0.8 / max(len(windows), 1)
        for i, window in enumerate(windows):
            block = (benchmarks[benchmarks["window"] == window]
                     .set_index("strategy").reindex(strategies))
            offset = (i - (len(windows) - 1) / 2) * width
            ax.barh(y + offset, block["cec"], height=width * 0.9,
                    color=_colour(i), label=window)
        ax.set_yticks(y)
        ax.set_yticklabels([_flat(s, 22) for s in strategies], fontsize=6.5)
        ax.set_xlabel("Certainty equivalent consumption")
        survivors = ", ".join(dict.fromkeys(
            str(v) for v in benchmarks.sort_values(["window", "rank"])
            .groupby("window").head(1)["strategy"]))
        _title(ax, f"Both halves are won by {_flat(survivors, 999)}"
               if "," not in survivors else
               "The two halves are won by different strategies")
        ax.legend(fontsize=6.5, loc="lower right", frameon=True,
                  framealpha=0.92, edgecolor="none")
        ax.grid(axis="y", alpha=0.0)

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_human_capital(curve: pd.DataFrame, ranking: pd.DataFrame,
                       directory: str | Path,
                       name: str = "fig48_human_capital") -> Path:
    """What happens when the pay cheque is a claim on the home market."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(2, 3.0)

        ax = axes[0]
        modes = (list(dict.fromkeys(curve["mode"])) if "mode" in curve.columns
                 else [None])
        home = curve if modes == [None] else curve[curve["mode"] == modes[0]]
        for i, mode in enumerate(modes):
            block = (curve if mode is None
                     else curve[curve["mode"] == mode]).sort_values(
                         "correlation")
            ax.plot(block["correlation"].to_numpy(dtype=float),
                    block["gap_pct"].to_numpy(dtype=float),
                    marker=_marker(i), color=_colour(i), linewidth=1.7,
                    markersize=3.5,
                    label=MODE_TICK.get(str(mode), str(mode)))
        ax.axhline(0.0, color="black", linewidth=1.2)
        y = home.sort_values("correlation")["gap_pct"].to_numpy(dtype=float)
        base = float(y[0]) if y.size else float("nan")
        ax.axhline(base, color="0.5", linewidth=1.0, linestyle=":")
        ax.annotate("independent human capital",
                    xy=(float(home["correlation"].min()), base),
                    xytext=(4, 6), textcoords="offset points", fontsize=6,
                    color="0.35")
        ax.set_xlabel("Correlation of the pay cheque with equity")
        ax.set_ylabel("All-international over 50/50 (%)")
        widens = bool(y.size and y[-1] > y[0])
        _title(ax, "Correlated human capital argues against the home market"
               if widens else
               "Correlated human capital argues for the home market")
        if len(modes) > 1:
            ax.legend(fontsize=5.5, loc="best", labelspacing=0.3,
                      handlelength=1.6, frameon=True, framealpha=0.92,
                      edgecolor="none")

        ax = axes[1]
        levels = [c for c in ranking.columns if isinstance(c, float)]
        for i, (_, row) in enumerate(ranking.iterrows()):
            ax.plot(levels, [float(row[c]) for c in levels],
                    marker=_marker(i), color=_colour(i), linewidth=1.5,
                    markersize=3.5, label=_flat(row["label"], 999))
        ax.set_xlabel("Correlation of the pay cheque with the home market")
        ax.set_ylabel("Certainty equivalent consumption")
        orders = {tuple(ranking.sort_values(c, ascending=False)["strategy"])
                  for c in levels}
        _title(ax, "Nothing changes places" if len(orders) <= 1 else
               "The order is not the same at every correlation")
        ax.margins(y=0.16)
        ax.legend(fontsize=5.5, ncol=2, loc="lower left", labelspacing=0.3,
                  handlelength=1.4, columnspacing=1.0, frameon=True,
                  framealpha=0.92, edgecolor="none")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_sequence(frame: pd.DataFrame, ranking: pd.DataFrame,
                  comparison: pd.DataFrame, focus: str,
                  directory: str | Path,
                  name: str = "fig55_sequence_risk") -> Path:
    """How much of a lifetime's outcome is the order the returns arrived in."""
    order = [p for p in ("none", "accumulation", "retirement", "both")
             if p in set(frame["phase"])]
    tick = {"none": "nothing\nshuffled", "accumulation": "working\nyears",
            "retirement": "retired\nyears", "both": "whole\nlifetime"}
    block = frame[frame["strategy"] == focus].set_index("phase")

    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. the share of variance that is pure ordering ----------------
        ax = axes[0]
        share = [float(block.loc[p, "sequence_share"]) * 100.0 for p in order]
        colours = [_colour(2) if p == "retirement" else _colour(0)
                   for p in order]
        ax.bar(range(len(order)), share, color=colours, width=0.64)
        ax.set_xticks(np.arange(len(order)))
        _strategy_ticks(ax, [tick.get(p, p) for p in order], fontsize=5.5)
        ax.set_ylabel("Share of outcome variance (%)")
        _title(ax, "How much of the risk is only the order")

        # -- 2. the same in consumption units, split ------------------------
        ax = axes[1]
        seqsd = np.array([float(block.loc[p, "sd_sequence"]) for p in order])
        levsd = np.array([float(block.loc[p, "sd_level"]) for p in order])
        ax.bar(range(len(order)), levsd, color=_colour(1), width=0.64,
               label="the returns drawn")
        ax.bar(range(len(order)), seqsd, bottom=levsd, color=_colour(2),
               width=0.64, label="the order they came in")
        ax.set_xticks(np.arange(len(order)))
        _strategy_ticks(ax, [tick.get(p, p) for p in order], fontsize=5.5)
        ax.set_ylabel("Standard deviation of retirement\nconsumption")
        _title(ax, "Ordering risk stacked on return risk")
        ax.legend(fontsize=5.5, loc="upper left", labelspacing=0.3,
                  handlelength=1.2, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 3. what it costs ----------------------------------------------
        ax = axes[2]
        cec = [float(block.loc[p, "cec"]) for p in order]
        ax.plot(range(len(order)), cec, marker=_marker(0), color=_colour(0),
                linewidth=1.7, markersize=4.0, label="certainty equivalent")
        ax.set_xticks(np.arange(len(order)))
        _strategy_ticks(ax, [tick.get(p, p) for p in order], fontsize=5.5)
        ax.set_ylabel("Certainty equivalent consumption")
        twin = ax.twinx()
        ruin = [float(block.loc[p, "prob_ruin"]) * 100.0 for p in order]
        twin.plot(range(len(order)), ruin, marker=_marker(1), color=_colour(2),
                  linewidth=1.5, markersize=3.4, linestyle="--",
                  label="P(ruin)")
        twin.set_ylabel("P(ruin), %", fontsize=7)
        twin.tick_params(labelsize=6.5)
        _title(ax, "What a random order costs")
        handles = ax.get_legend_handles_labels()[0] + \
            twin.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + \
            twin.get_legend_handles_labels()[1]
        ax.legend(handles, labels, fontsize=5.5, loc="center left",
                  labelspacing=0.3, handlelength=1.4, frameon=True,
                  framealpha=0.92, edgecolor="none")

        # -- 4. where the risk sits depends on the withdrawal rule ---------
        ax = axes[3]
        if len(comparison):
            idx = np.arange(len(comparison))
            ax.barh(idx - 0.19,
                    comparison["share_accumulation"].to_numpy(dtype=float)
                    * 100.0, height=0.36, color=_colour(0),
                    label="working years")
            ax.barh(idx + 0.19,
                    comparison["share_retirement"].to_numpy(dtype=float)
                    * 100.0, height=0.36, color=_colour(2),
                    label="retired years")
            ax.set_yticks(idx)
            ax.set_yticklabels([_wrap(str(v), 18)
                                for v in comparison["label"]], fontsize=5.5)
            ax.legend(fontsize=5.5, loc="lower right", labelspacing=0.3,
                      handlelength=1.2, frameon=True, framealpha=0.92,
                      edgecolor="none")
        ax.set_xlabel("Share of outcome variance that is ordering (%)")
        _title(ax, "The withdrawal rule decides where the risk lands")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_withholding(curve: pd.DataFrame, crossings: pd.DataFrame,
                     anchors: pd.DataFrame, drag: pd.DataFrame,
                     optima: pd.DataFrame, sweep: pd.DataFrame,
                     dom_param: Mapping[str, float], column: str,
                     challenger: str, rivals: Sequence[str],
                     directory: str | Path,
                     name: str = "fig53_withholding") -> Path:
    """What a tax on foreign dividends does to the case for foreign equity."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. the drag a statutory rate produces, by era -----------------
        ax = axes[0]
        eras = drag[drag["era"] != "whole panel"]
        if len(eras):
            ax.bar(range(len(eras)), eras["drag_bp"].to_numpy(dtype=float),
                   color=_colour(0), width=0.62)
            whole = drag[drag["era"] == "whole panel"]
            if len(whole):
                ax.axhline(float(whole["drag_bp"].iloc[0]), color="0.4",
                           linewidth=1.0, linestyle=":")
            ax.set_xticks(np.arange(len(eras)))
            _strategy_ticks(ax, list(eras["era"]), fontsize=6.0)
        ax.set_ylabel("Drag on the sleeve (bp a year)")
        _title(ax, "The same law costs less as dividend yields fall")

        # -- 2. every strategy's certainty equivalent against the rate -----
        ax = axes[1]
        keys = [challenger] + list(rivals)
        for i, key in enumerate(keys):
            col = f"cec_{key}"
            if col not in curve.columns:
                continue
            block = curve.sort_values("rate_pct")
            ax.plot(block["rate_pct"].to_numpy(dtype=float),
                    block[col].to_numpy(dtype=float), marker=_marker(i),
                    color=_colour(i), linewidth=1.6, markersize=3.2,
                    label=_legend(key))
        for _, r in anchors.iterrows():
            ax.axvline(float(r["rate_pct"]), color="0.75", linewidth=0.8,
                       linestyle=":", zorder=0)
        ax.set_xlabel("Withholding rate on foreign dividends (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "Only the foreign leg pays it")
        ax.legend(fontsize=5.5, loc="lower left", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 3. the lead, and where it runs out ----------------------------
        ax = axes[2]
        block = curve.sort_values("rate_pct")
        x = block["rate_pct"].to_numpy(dtype=float)
        for i, rival in enumerate(rivals):
            col = f"lead_over_{rival}_pct"
            if col not in block.columns:
                continue
            y = block[col].to_numpy(dtype=float)
            ax.plot(x, y, marker=_marker(i), color=_colour(i), linewidth=1.6,
                    markersize=3.2, label=_legend(rival))
        ax.axhline(0.0, color="black", linewidth=1.1)
        for _, r in crossings.iterrows():
            if bool(r["reached_on_grid"]):
                ax.axvline(float(r["crossing_pct"]), color="0.5",
                           linewidth=0.9, linestyle="--", zorder=0)
        ax.set_xlabel("Withholding rate on foreign dividends (%)")
        ax.set_ylabel("All-international's lead (%)")
        survives = bool(len(block) and
                        all(float(block[f"lead_over_{r}_pct"].iloc[-1]) > 0
                            for r in rivals
                            if f"lead_over_{r}_pct" in block.columns))
        _title(ax, "The lead survives every rate tested" if survives
               else "Where the lead runs out")
        ax.legend(fontsize=5.5, loc="lower left", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 4. what to hold at each rate ----------------------------------
        ax = axes[3]
        rates = sorted(sweep["rate"].unique())
        for i, rate in enumerate(rates):
            sub = sweep[(np.isclose(sweep["rate"], rate))
                        & (sweep["strategy"].isin(dom_param))].copy()
            if not len(sub):
                continue
            sub["share"] = [dom_param[k] for k in sub["strategy"]]
            sub = sub.sort_values("share")
            ax.plot(sub["share"].to_numpy(dtype=float) * 100.0,
                    sub[column].to_numpy(dtype=float), color=_colour(i),
                    linewidth=1.3, label=f"{rate:.0%}")
        if len(optima):
            ax.scatter(
                optima["optimal_domestic_share"].to_numpy(dtype=float) * 100.0,
                optima["cec_at_optimum"].to_numpy(dtype=float), s=38,
                facecolors="none", edgecolors="black", linewidths=1.1,
                zorder=5)
        ax.set_xlabel("Domestic share of the equity sleeve (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "The optimum walks home as the tax rises")
        ax.legend(fontsize=5.0, ncol=2, loc="lower left", labelspacing=0.25,
                  handlelength=1.2, columnspacing=0.8, frameon=True,
                  framealpha=0.92, edgecolor="none", title="rate",
                  title_fontsize=5.0)

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_franking(curve: pd.DataFrame, crossings: pd.DataFrame,
                  anchors: pd.DataFrame, era: pd.DataFrame,
                  optima: pd.DataFrame, comparison: pd.DataFrame,
                  wedge_optimal: pd.DataFrame,
                  challenger: str, rivals: Sequence[str],
                  swept_rate: float,
                  directory: str | Path,
                  name: str = "fig56_franking") -> Path:
    """What a credit on home dividends does to the case for foreign equity."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. the lead, and where the credit runs it out -----------------
        ax = axes[0]
        block = curve.sort_values("credit_pct")
        x = block["credit_pct"].to_numpy(dtype=float)
        for i, rival in enumerate(rivals):
            col = f"lead_over_{rival}_pct"
            if col not in block.columns:
                continue
            ax.plot(x, block[col].to_numpy(dtype=float), marker=_marker(i),
                    color=_colour(i), linewidth=1.6, markersize=3.2,
                    label=_legend(rival))
        ax.axhline(0.0, color="black", linewidth=1.1)
        ax.axvline(0.0, color="0.6", linewidth=0.9)
        for _, r in crossings.iterrows():
            if bool(r["reached_on_grid"]):
                ax.axvline(float(r["crossing_pct"]), color="0.5",
                           linewidth=0.9, linestyle="--", zorder=0)
        for _, r in anchors.iterrows():
            if float(r["credit"]) > 0:
                ax.axvline(float(r["credit_pct"]), color="0.78",
                           linewidth=0.8, linestyle=":", zorder=0)
        ax.set_xlabel("Imputation credit on home dividends (% of the dividend)")
        ax.set_ylabel("All-international's lead (%)")
        _title(ax, "Where the credit runs the lead out")
        ax.legend(fontsize=5.5, loc="upper right", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 2. the credit's worth by era, against the withholding drag ----
        ax = axes[1]
        eras = era[era["era"] != "whole panel"]
        if len(eras):
            ax.bar(range(len(eras)), eras["credit_bp"].to_numpy(dtype=float),
                   color=_colour(2), width=0.62)
            whole = era[era["era"] == "whole panel"]
            if len(whole):
                ax.axhline(float(whole["credit_bp"].iloc[0]), color="0.4",
                           linewidth=1.0, linestyle=":")
            ax.set_xticks(np.arange(len(eras)))
            _strategy_ticks(ax, list(eras["era"]), fontsize=6.0)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_ylabel("Worth of the credit (bp a year)")
        _title(ax, "The same credit is worth less as yields fall")

        # -- 3. the wedge: each rival's lead over the challenger -----------
        #
        # Levels would compress a one-per-cent reversal into indistinguishable
        # bars and truncating the axis to fix that would exaggerate it. The
        # difference is the quantity in question, so it is what is drawn: a
        # bar above the line is a rival beating all-international.
        ax = axes[2]
        keys = [k for k in rivals if f"cec_{k}" in comparison.columns]
        base = (comparison[f"cec_{challenger}"].to_numpy(dtype=float)
                if f"cec_{challenger}" in comparison.columns else None)
        idx = np.arange(len(comparison))
        width = 0.8 / max(len(keys), 1)
        if base is not None:
            for i, key in enumerate(keys):
                lead = (comparison[f"cec_{key}"].to_numpy(dtype=float) / base
                        - 1.0) * 100.0
                ax.bar(idx + (i - (len(keys) - 1) / 2.0) * width, lead,
                       width=width, color=_colour(i + 1), label=_legend(key))
        ax.axhline(0.0, color="black", linewidth=1.1)
        ax.set_xticks(idx)
        _strategy_ticks(ax, [POSITION_LABEL.get(str(v), _wrap(str(v), 14))
                             for v in comparison["position"]], fontsize=5.0)
        ax.set_ylabel(f"Lead over {_legend(challenger)} (%)")
        _title(ax, "Both blades of the scissors, together")
        ax.legend(fontsize=5.0, ncol=1, loc="lower left", labelspacing=0.25,
                  handlelength=1.1, columnspacing=0.8, frameon=True,
                  framealpha=0.92, edgecolor="none")

        # -- 4. what to hold at each credit --------------------------------
        ax = axes[3]
        if len(optima):
            block = optima.sort_values("credit")
            ax.plot(block["credit"].to_numpy(dtype=float) * 100.0,
                    block["optimal_domestic_share"].to_numpy(dtype=float)
                    * 100.0, marker=_marker(0), color=_colour(0),
                    linewidth=1.7, markersize=4.0, drawstyle="steps-mid",
                    label="swept credit")
        # Only the positions drawn on the same panel as the curve. A position
        # at a different withholding rate has a different optimum for a reason
        # this axis cannot show, and plotting it here would read as a
        # contradiction rather than as a second experiment.
        named = (wedge_optimal[np.isclose(wedge_optimal["rate"], swept_rate)]
                 if {"credit", "rate"} <= set(wedge_optimal.columns)
                 else wedge_optimal.iloc[:0])
        if len(named):
            ax.scatter(named["credit"].to_numpy(dtype=float) * 100.0,
                       named["optimal_domestic_share"].to_numpy(dtype=float)
                       * 100.0, s=40, facecolors="none", edgecolors="black",
                       linewidths=1.1, zorder=5,
                       label="a position an investor holds")
        ax.axvline(0.0, color="0.6", linewidth=0.9)
        ax.set_xlabel("Imputation credit on home dividends (%)")
        ax.set_ylabel("Optimal domestic share of equity (%)")
        _title(ax, "The optimum walks home as the credit rises")
        ax.legend(fontsize=5.5, loc="upper left", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_inflation_timing(birth: pd.DataFrame, retirement: pd.DataFrame,
                          ret_eq: pd.DataFrame, ret_dom: pd.DataFrame,
                          ret_sweep: pd.DataFrame,
                          eq_param: Mapping[str, float],
                          dom_param: Mapping[str, float], column: str,
                          strategy: str, directory: str | Path,
                          name: str = "fig54_inflation_timing") -> Path:
    """The same state variable read at two dates, and the retiree's choice."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(3, 3.0)

        # -- 1. the level, conditioned at each date ------------------------
        ax = axes[0]
        for i, (label, frame) in enumerate((("at age 25", birth),
                                            ("at retirement", retirement))):
            block = frame[frame["strategy"] == strategy]
            if not len(block):
                continue
            idx = np.arange(len(block)) + (i - 0.5) * 0.36
            ax.bar(idx, block[column].to_numpy(dtype=float), width=0.34,
                   color=_colour(i), label=label)
        block = birth[birth["strategy"] == strategy]
        if len(block):
            ax.set_xticks(np.arange(len(block)))
            _strategy_ticks(ax, [_wrap(str(b), 12) for b in block["bucket"]],
                            fontsize=5.5)
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "When the state variable is read decides whether it matters")
        ax.legend(fontsize=5.5, loc="lower left", labelspacing=0.3,
                  handlelength=1.2, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 2. what a retiree should hold: how much equity ----------------
        ax = axes[1]
        buckets = list(dict.fromkeys(ret_sweep["bucket"])) if len(ret_sweep) \
            else []
        for i, bucket in enumerate(buckets):
            sub = ret_sweep[(ret_sweep["bucket"] == bucket)
                            & (ret_sweep["strategy"].isin(eq_param))].copy()
            if not len(sub):
                continue
            sub["share"] = [eq_param[k] for k in sub["strategy"]]
            sub = sub.sort_values("share")
            ax.plot(sub["share"].to_numpy(dtype=float) * 100.0,
                    sub[column].to_numpy(dtype=float), marker=_marker(i),
                    color=_colour(i), linewidth=1.5, markersize=3.0,
                    label=_wrap(str(bucket), 16))
        if len(ret_eq):
            ax.scatter(
                ret_eq["optimal_equity_share"].to_numpy(dtype=float) * 100.0,
                ret_eq["cec_at_optimum"].to_numpy(dtype=float), s=40,
                facecolors="none", edgecolors="black", linewidths=1.1,
                zorder=5)
        ax.set_xlabel("Equity share held from retirement (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "What a retiree should hold, given what they see")
        ax.legend(fontsize=5.5, loc="lower right", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 3. and how much of it at home ---------------------------------
        ax = axes[2]
        for i, bucket in enumerate(buckets):
            sub = ret_sweep[(ret_sweep["bucket"] == bucket)
                            & (ret_sweep["strategy"].isin(dom_param))].copy()
            if not len(sub):
                continue
            sub["share"] = [dom_param[k] for k in sub["strategy"]]
            sub = sub.sort_values("share")
            ax.plot(sub["share"].to_numpy(dtype=float) * 100.0,
                    sub[column].to_numpy(dtype=float), marker=_marker(i),
                    color=_colour(i), linewidth=1.5, markersize=3.0,
                    label=_wrap(str(bucket), 16))
        if len(ret_dom):
            ax.scatter(
                ret_dom["optimal_domestic_share"].to_numpy(dtype=float) * 100.0,
                ret_dom["cec_at_optimum"].to_numpy(dtype=float), s=40,
                facecolors="none", edgecolors="black", linewidths=1.1,
                zorder=5)
        ax.set_xlabel("Domestic share of retirement equity (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "And how much of it at home")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_inflation_state(grid: pd.DataFrame, ordering: pd.DataFrame,
                         advantage: pd.DataFrame, eq_optima: pd.DataFrame,
                         dom_optima: pd.DataFrame, sweep: pd.DataFrame,
                         eq_param: Mapping[str, float],
                         dom_param: Mapping[str, float], column: str,
                         directory: str | Path,
                         name: str = "fig52_inflation_state") -> Path:
    """What recent inflation predicts, and what it does to the portfolio."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)

        # -- 1. the damage by asset and horizon ----------------------------
        ax = axes[0]
        assets = [a for a in ("bond", "bill", "dom_eq", "intl_eq")
                  if a in set(grid["asset"])]
        for i, asset in enumerate(assets):
            block = grid[(grid["asset"] == asset)].groupby(
                "horizon_years")["gap"].mean().sort_index()
            ax.plot(block.index.to_numpy(dtype=float),
                    block.to_numpy(dtype=float) * 100.0,
                    marker=_marker(i), color=_colour(i), linewidth=1.6,
                    markersize=3.5, label=_abbr(asset).replace("\n", " "))
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Years ahead")
        ax.set_ylabel("High minus low inflation third\n(annualised real, pp)")
        _title(ax, "Inflation is a short-horizon risk to nominal assets")
        ax.legend(fontsize=5.5, loc="lower right", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 2. the ordering at the headline horizon -----------------------
        ax = axes[1]
        if len(ordering):
            y = ordering["gap"].to_numpy(dtype=float) * 100.0
            colours = [_colour(2) if v < 0 else _colour(0) for v in y]
            ax.barh(range(len(ordering)), y, color=colours, height=0.6)
            ax.set_yticks(np.arange(len(ordering)))
            ax.set_yticklabels([_abbr(a) for a in ordering["asset"]],
                               fontsize=6.5)
            ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("High minus low inflation third (pp a year)")
        _title(ax, "What a high-inflation start costs, by asset")

        # -- 3. the certainty-equivalent surface, bucket by bucket ---------
        ax = axes[2]
        buckets = list(dict.fromkeys(sweep["bucket"])) if len(sweep) else []
        for i, bucket in enumerate(buckets):
            block = sweep[(sweep["bucket"] == bucket)
                          & (sweep["strategy"].isin(eq_param))].copy()
            if not len(block):
                continue
            block["share"] = [eq_param[k] for k in block["strategy"]]
            block = block.sort_values("share")
            ax.plot(block["share"].to_numpy(dtype=float) * 100.0,
                    block[column].to_numpy(dtype=float), marker=_marker(i),
                    color=_colour(i), linewidth=1.5, markersize=3.0,
                    label=_wrap(str(bucket), 16))
        if len(eq_optima):
            ax.scatter(eq_optima["optimal_equity_share"].to_numpy(dtype=float)
                       * 100.0,
                       eq_optima["cec_at_optimum"].to_numpy(dtype=float),
                       s=42, facecolors="none", edgecolors="black",
                       linewidths=1.1, zorder=5)
        ax.set_xlabel("Equity share of the portfolio (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "How much equity each regime wants")
        ax.legend(fontsize=5.5, loc="lower right", labelspacing=0.3,
                  handlelength=1.4, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 4. the home/abroad axis, and the headline lead ----------------
        ax = axes[3]
        for i, bucket in enumerate(buckets):
            block = sweep[(sweep["bucket"] == bucket)
                          & (sweep["strategy"].isin(dom_param))].copy()
            if not len(block):
                continue
            block["share"] = [dom_param[k] for k in block["strategy"]]
            block = block.sort_values("share")
            ax.plot(block["share"].to_numpy(dtype=float) * 100.0,
                    block[column].to_numpy(dtype=float), marker=_marker(i),
                    color=_colour(i), linewidth=1.5, markersize=3.0,
                    label=_wrap(str(bucket), 16))
        if len(dom_optima):
            ax.scatter(dom_optima["optimal_domestic_share"].to_numpy(dtype=float)
                       * 100.0,
                       dom_optima["cec_at_optimum"].to_numpy(dtype=float),
                       s=42, facecolors="none", edgecolors="black",
                       linewidths=1.1, zorder=5)
        ax.set_xlabel("Domestic share of the equity sleeve (%)")
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "And how much of it at home")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_pension(gaps: pd.DataFrame, entitlement: pd.DataFrame,
                 replacement: pd.DataFrame, directory: str | Path,
                 name: str = "fig50_pension") -> Path:
    """A means-tested pension and a compulsory contribution, against the US."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(4, 3.0)
        order = gaps.reset_index(drop=True)
        ticks = [PENSION_TICK.get(str(k), str(k)) for k in order["system"]]

        def _bar(ax: Any, values: np.ndarray, highlight: str) -> None:
            colours = [_colour(2) if k == highlight
                       else _colour(0) if k == "us_social_security"
                       else _colour(1) for k in order["system"]]
            ax.bar(range(len(order)), values, color=colours, width=0.66)
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.set_xticks(np.arange(len(order)))
            _strategy_ticks(ax, ticks, fontsize=5.5)

        # -- 1. how much retirement consumption each system delivers -------
        ax = axes[0]
        lift = (order["best_lift_pct"].to_numpy(dtype=float)
                if "best_lift_pct" in order.columns
                else np.zeros(len(order)))
        _bar(ax, lift, "australia_as_legislated")
        ax.set_ylabel("Certainty equivalent vs the\nUS baseline (%)")
        _title(ax, "The Australian system delivers more"
               if float(lift[-1]) > 0 else "The Australian system delivers less")

        # -- 2. and what it does to the choice between portfolios ----------
        ax = axes[1]
        y = order["gap_pct"].to_numpy(dtype=float)
        _bar(ax, y, "australia_as_legislated")
        if "us_social_security" in set(order["system"]):
            ax.axhline(float(order.loc[order["system"] == "us_social_security",
                                       "gap_pct"].iloc[0]),
                       color="0.5", linewidth=1.0, linestyle=":")
        ax.set_ylabel("All-international over 50/50 (%)")
        _title(ax, "but compresses the choice between portfolios"
               if abs(float(y[-1])) < abs(float(y[0]))
               else "and widens the choice between portfolios")

        # -- 3. who is actually on the taper -------------------------------
        ax = axes[2]
        if len(entitlement):
            key = ("australia_as_legislated"
                   if "australia_as_legislated" in set(entitlement["system"])
                   else str(entitlement["system"].iloc[0]))
            block = entitlement[entitlement["system"] == key].sort_values("age")
            ages = block["age"].to_numpy(dtype=float)
            bands = [("share_no_pension", "no pension", 2),
                     ("share_part_rate", "part rate", 1),
                     ("share_full_rate", "full rate", 0)]
            ax.stackplot(ages, *[block[c].to_numpy(dtype=float) * 100
                                 for c, _, _ in bands],
                         colors=[_colour(i) for _, _, i in bands],
                         labels=[lab for _, lab, _ in bands], alpha=0.85)
            ax.set_xlim(float(ages.min()), float(ages.max()))
            ax.set_ylim(0, 100)
            ax.legend(fontsize=5.5, loc="center left", labelspacing=0.3,
                      handlelength=1.2, frameon=True, framealpha=0.92,
                      edgecolor="none")
        ax.set_xlabel("Age")
        ax.set_ylabel("Share of retirees (%)")
        _title(ax, "The taper switches on as the portfolio runs down")

        # -- 4. how much of retirement the state pays for ------------------
        ax = axes[3]
        if len(replacement):
            share = replacement["pension_share_of_consumption"].to_numpy(
                dtype=float) * 100.0
            ax.bar(range(len(replacement)), share, color=_colour(3), width=0.6)
            ax.set_xticks(np.arange(len(replacement)))
            _strategy_ticks(ax, [PENSION_TICK.get(str(k), str(k))
                                 for k in replacement["system"]], fontsize=5.5)
        ax.set_ylabel("Public pension as a share of\nretirement consumption (%)")
        _title(ax, "What the state is paying for")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_turnover(measured: pd.DataFrame, curve: pd.DataFrame,
                  strategies: Mapping[str, Any], spec: Any,
                  directory: str | Path,
                  name: str = "fig51_turnover") -> Path:
    """What the schedules trade, and what trading costs them."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(3, 3.0)

        # -- 1. turnover by strategy, against the drift floor --------------
        ax = axes[0]
        order = measured.reset_index(drop=True)
        idx = np.arange(len(order))
        ax.bar(idx - 0.19, order["turnover_total"].to_numpy(dtype=float) * 100,
               width=0.38, color=_colour(0), label="actually traded")
        ax.bar(idx + 0.19,
               order["turnover_drift_only"].to_numpy(dtype=float) * 100,
               width=0.38, color=_colour(1), label="drift alone")
        ax.set_xticks(idx)
        _strategy_ticks(ax, [_abbr(v) for v in order["label"]])
        ax.set_ylabel("One-way turnover a year (%)")
        _title(ax, "Most trading is drift, not a change of plan")
        ax.legend(fontsize=5.5, loc="upper right", labelspacing=0.3,
                  handlelength=1.2, frameon=True, framealpha=0.92,
                  edgecolor="none")

        # -- 2. where in a life the trading happens ------------------------
        ax = axes[1]
        from src import turnover as _tn
        for i, (key, strat) in enumerate(strategies.items()):
            own = _tn.schedule_turnover(strat.weights) * 100.0
            if own.max() <= 1e-9:
                continue
            ax.plot(spec.ages, own, color=_colour(i), linewidth=1.5,
                    label=_abbr(strat.label))
        ax.axvline(spec.age_retire, color="0.5", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Age")
        ax.set_ylabel("Turnover the schedule demands (%)")
        _title(ax, "The schedule's own trading, with returns switched off")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=5.5, loc="upper left", labelspacing=0.3,
                      handlelength=1.4, frameon=True, framealpha=0.92,
                      edgecolor="none")

        # -- 3. the lead against the cost of trading -----------------------
        ax = axes[2]
        x = curve["basis_points"].to_numpy(dtype=float)
        y = curve["gap_pct"].to_numpy(dtype=float)
        ax.plot(x, y, marker=_marker(0), color=_colour(0), linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=1.2)
        ax.fill_between(x, 0.0, y, where=y > 0, color=_colour(0), alpha=0.10)
        ax.fill_between(x, 0.0, y, where=y <= 0, color=_colour(2), alpha=0.10)
        ax.set_xlabel("One-way trading cost (basis points)")
        ax.set_ylabel("Solved schedule over the best\nfixed portfolio (%)")
        survives = bool(y.size and y[-1] > 0)
        _title(ax, "The solved schedule pays for its own trading"
               if survives else "Trading costs eat the solved schedule's edge")

        fig.tight_layout()
    return _save(fig, directory, name)


def plot_mortality(frame: pd.DataFrame, curve: pd.DataFrame,
                   survival: Mapping[str, np.ndarray], ages: np.ndarray,
                   directory: str | Path,
                   name: str = "fig49_mortality") -> Path:
    """A random lifespan, and what it does to the ranking."""
    with plt.rc_context(STYLE):
        fig, axes = _grid(3, 3.0)

        # -- 1. the mortality laws themselves ------------------------------
        ax = axes[0]
        for i, (label, curve_values) in enumerate(survival.items()):
            ax.plot(ages, np.asarray(curve_values) * 100, color=_colour(i),
                    linewidth=1.6, label=_wrap(label, 22))
        ax.set_xlabel("Age")
        ax.set_ylabel("Still alive (%)")
        _title(ax, "The laws swept")
        ax.legend(fontsize=5.5, loc="lower left", labelspacing=0.3,
                  handlelength=1.4)

        # -- 2. certainty equivalent under each law ------------------------
        ax = axes[1]
        laws = list(dict.fromkeys(frame["mortality"]))
        strategies = list(frame[frame["mortality"] == laws[0]]
                          .sort_values("cec", ascending=False)["strategy"])
        x = np.arange(len(laws), dtype=float)
        for i, key in enumerate(strategies):
            block = frame[frame["strategy"] == key].set_index("mortality") \
                .reindex(laws)
            ax.plot(x, block["cec"], marker=_marker(i), color=_colour(i),
                    linewidth=1.5, markersize=3.5, label=_flat(key, 999))
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap(str(v), 14) for v in laws], fontsize=5.5)
        ax.set_ylabel("Certainty equivalent consumption")
        _title(ax, "The ranking under each law")
        ax.margins(y=0.22)
        ax.legend(fontsize=5.5, ncol=2, loc="lower left", labelspacing=0.3,
                  handlelength=1.2, columnspacing=1.0, frameon=True,
                  framealpha=0.92, edgecolor="none")

        # -- 3. the lead, and what happens to ruin -------------------------
        ax = axes[2]
        x = np.arange(len(curve), dtype=float)
        ax.bar(x, curve["gap_pct"], width=0.55, color=_colour(0),
               label="lead of all-international (left)")
        ax.axhline(0.0, color="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap(str(v), 16) for v in curve["mortality"]],
                           fontsize=5.5)
        ax.set_ylabel("All-international over 50/50 (%)")
        ruin_col = [c for c in curve.columns if c.startswith("ruin_")]
        if ruin_col:
            twin = ax.twinx()
            twin.plot(x, curve[ruin_col[0]] * 100, marker=_marker(1),
                      color=_colour(1), linewidth=1.6,
                      label="chance of outliving the portfolio (right)")
            twin.set_ylabel("Outlives the portfolio (%)")
            twin.grid(False)
            handles = (ax.get_legend_handles_labels()[0]
                       + twin.get_legend_handles_labels()[0])
            labels = (ax.get_legend_handles_labels()[1]
                      + twin.get_legend_handles_labels()[1])
            ax.legend(handles, labels, fontsize=5.5, loc="lower left",
                      labelspacing=0.3, handlelength=1.4, frameon=True,
                      framealpha=0.92, edgecolor="none")
        ruin = curve[ruin_col[0]].to_numpy(dtype=float) if ruin_col else None
        kinder = bool(ruin is not None and ruin[1:].max() < ruin[0])
        _title(ax, "Outliving the money is a smaller risk when life is finite"
               if kinder else
               "The lead barely moves; the chance of outliving the money does")

        fig.tight_layout()
    return _save(fig, directory, name)
