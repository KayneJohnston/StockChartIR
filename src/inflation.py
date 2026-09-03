"""Recent inflation, and what it does to a lifetime.

Section #valuation conditions a lifetime on how expensive the market was when
it opened. This section conditions it on something an investor knows even
better than the dividend yield, and reads about every month: **what inflation
has just done**.

The two questions are not the same, and the second has a sharper mechanism
behind it. A dividend yield is a claim about expected returns that has to be
argued for. Trailing inflation is a claim about the *price level*, and every
return in this paper is already deflated by it -- so the channel is direct.
Nominal bonds and bills promise a fixed number of currency units; if inflation
is high and persistent, that promise is worth less in real terms, and the
asset whose real return inflation eats is the one this paper's rivals hold
most of. Equity is a claim on nominal cash flows that reprice, which is a
partial hedge in the long run and a famously poor one in the short.

That gives the section a hypothesis with a direction, which is what makes it
worth running:

* Trailing inflation should predict low forward **real bond and bill**
  returns, and much more weakly for equity.
* It should therefore push the optimal portfolio **toward equity**, and toward
  the *international* leg specifically, because a domestic inflation shock is
  a domestic phenomenon and foreign assets are not denominated in the currency
  that is losing value.
* And the reason it should predict anything at all is **persistence**: high
  inflation last year means high inflation next year. That link is measured
  here rather than assumed, because if it fails the rest has no mechanism.

The section would be surprising, and worth more, if the ranking moved. Section
#valuation found that starting valuation changes the *level* of what an
investor should expect but not *what they should hold*. Inflation has a better
claim to move the allocation, because unlike a valuation it does not affect
every asset in the same direction.

**The observable, and the constraint.** Nothing here may use information the
investor could not have had. Inflation in year ``t`` is unknown until year
``t`` is over, so the quantity observable on the first day of year ``t`` is
the annualised rate over the ``k`` years that have already finished:

    pi(t, k) = (prod_{j=1..k} (1 + infl_{t-j}))^(1/k) - 1

built from rows ``t-k`` through ``t-1`` and nothing later.
:func:`trailing_inflation` builds exactly that, and
:func:`depends_only_on_past` checks it structurally -- tampering with one
year's inflation and confirming no earlier row moves. A correlation test could
not do that job, because a leak and an honest signal look identical in a
correlation.

**Which inflation.** The domestic one. Every return in this model is already
expressed in the investor's own real terms, and the price level they consume
at is their own -- so the state variable is the domestic CPI, not a world
average. That is a choice, though, and a reader is entitled to ask whether the
signal is "your country is inflating" or merely "these were inflationary
years". :func:`global_inflation` builds the leave-one-out average that answers
it, and the two labellings are compared rather than one being asserted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import valuation as vln

LOGGER = logging.getLogger(__name__)

#: Lookback windows, in years. One year is what a newspaper reports; three and
#: five are what a central bank would call the recent trend. Whether they
#: differ is one of the questions.
DEFAULT_WINDOWS: Tuple[int, ...] = (1, 3, 5)

#: Tercile names in ``np.digitize`` order, so index 0 is the lowest inflation.
BUCKET_LABELS: Tuple[str, ...] = ("Low inflation", "Moderate", "High inflation")

#: Assets whose forward real return is regressed on trailing inflation. The
#: interesting comparison is equity against the nominal legs, so all four are
#: carried rather than just the one the headline holds.
FORWARD_ASSETS: Tuple[str, ...] = ("dom_eq", "intl_eq", "bond", "bill")


def trailing_inflation(inflation: np.ndarray, window: int) -> np.ndarray:
    """``(T, C)`` annualised inflation over the ``window`` years before each row.

    The value stored at row ``t`` uses rows ``t-window`` through ``t-1`` and
    nothing later, so pairing it with the return of year ``t`` onwards
    involves no look-ahead. Rows with an incomplete or non-finite window are
    left as NaN rather than filled: a lifetime that cannot be classified is
    excluded, not guessed at.

    Compounding rather than averaging matters at the rates this panel
    contains. A country that ran 100% one year and 0% the next did not
    experience "50% a year"; it experienced 41.4%, and the arithmetic mean
    would misplace it in the distribution.
    """
    values = np.asarray(inflation, dtype=float)
    window = int(window)
    if window < 1:
        raise ValueError("window must be at least one year")
    n_years, n_countries = values.shape
    out = np.full((n_years, n_countries), np.nan)
    growth = 1.0 + values
    for t in range(window, n_years):
        block = growth[t - window:t, :]
        ok = np.isfinite(block).all(axis=0) & (block > 0.0).all(axis=0)
        with np.errstate(invalid="ignore"):
            compounded = np.exp(np.log(np.where(ok, block, 1.0)).sum(axis=0)
                                / window) - 1.0
        out[t] = np.where(ok, compounded, np.nan)
    return out


def global_inflation(domestic: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of every *other* country's trailing inflation.

    The counterpart to the international sleeve's construction, and the check
    on whether the signal is a country's own price level or the era's. A
    column here is what the rest of the world was doing while country ``j``
    was doing whatever its own column says.
    """
    values = np.asarray(domestic, dtype=float)
    n_countries = values.shape[1]
    out = np.full_like(values, np.nan)
    for j in range(n_countries):
        others = np.delete(values, j, axis=1)
        present = np.isfinite(others).sum(axis=1)
        totals = np.nansum(others, axis=1)
        # A year in which no other market has an observation has no world
        # average, and np.nanmean would warn its way to a NaN. Divide only
        # where there is something to divide by.
        out[:, j] = np.where(present > 0, totals / np.maximum(present, 1),
                             np.nan)
    return out


def depends_only_on_past(inflation: np.ndarray, window: int,
                         year_index: int) -> bool:
    """Structural proof that no row is contaminated by year ``year_index``.

    Corrupt one year's inflation and confirm that every row up to and
    including that year is unchanged. If the trailing measure ever reached
    forward for a value, this fails; a statistical test could not tell the
    difference between a leak and a real signal, so the check is done on the
    arithmetic instead.
    """
    clean = trailing_inflation(inflation, window)
    tampered_input = np.array(inflation, dtype=float, copy=True)
    tampered_input[year_index, :] = tampered_input[year_index, :] + 10.0
    tampered = trailing_inflation(tampered_input, window)
    head_clean = clean[:year_index + 1]
    head_tampered = tampered[:year_index + 1]
    same = np.isclose(head_clean, head_tampered, equal_nan=True)
    return bool(same.all())


# ---------------------------------------------------------------------------
# What it predicts
# ---------------------------------------------------------------------------
#: Why every correlation in this module is a rank correlation.
#:
#: The panel contains real hyperinflations -- Germany 1923 at 1.06e9, Japan
#: 1945 at 976%, Italy 1944 at 344%. Those are observations, not errors, and
#: dropping them would delete precisely the episodes an inflation study exists
#: to look at. But a Pearson correlation computed on a series containing a
#: billion-per-cent observation is a statement about that one year and nothing
#: else: it reports -0.0004 for the persistence of inflation, a series whose
#: rank correlation with its own recent past is 0.58.
#:
#: So the headline statistic is Spearman's, the tercile comparison beside it
#: is rank-based by construction, and the Pearson figure is carried in its own
#: column as a diagnostic rather than suppressed. Winsorising would have
#: worked too, and would have required choosing a threshold; ranking requires
#: choosing nothing.
RANK_NOTE: str = (
    "Correlations are Spearman rank correlations. The panel contains real "
    "hyperinflations, and a Pearson correlation on a series with a 10^9 "
    "observation describes that observation rather than the relationship."
)


def _forward_annualised(series: np.ndarray, start: int, horizon: int
                        ) -> float:
    window = series[start:start + horizon]
    if window.size < horizon or not np.isfinite(window).all():
        return float("nan")
    if (window <= -1.0).any():
        return float("nan")
    return float(np.exp(np.mean(np.log1p(window))) - 1.0)


def predictive_power(trailing: np.ndarray, forward: np.ndarray,
                     horizons: Sequence[int] = (1, 5, 10, 30),
                     ) -> pd.DataFrame:
    """Does trailing inflation predict what follows it?

    One row per horizon: the correlation between the inflation an investor
    could see and the annualised outcome over the years after it, plus the
    means in the lowest and highest thirds of the inflation distribution. The
    ``gap`` is the high-inflation third minus the low, so a negative number
    means high inflation was followed by worse outcomes.

    ``correlation`` is Spearman's; ``pearson`` is carried beside it. See
    :data:`RANK_NOTE` for why that is not a stylistic preference.
    """
    rows: List[Dict[str, Any]] = []
    n_years, n_countries = np.asarray(forward).shape
    for h in horizons:
        pairs: List[Tuple[float, float]] = []
        for j in range(n_countries):
            column = np.asarray(forward)[:, j]
            for t in range(n_years - int(h)):
                state = trailing[t, j]
                if not np.isfinite(state):
                    continue
                outcome = _forward_annualised(column, t, int(h))
                if np.isfinite(outcome):
                    pairs.append((float(state), outcome))
        if len(pairs) < 50:
            continue
        frame = pd.DataFrame(pairs, columns=["trailing", "forward"])
        low = frame["trailing"].quantile(1 / 3)
        high = frame["trailing"].quantile(2 / 3)
        calm = float(frame[frame["trailing"] <= low]["forward"].mean())
        hot = float(frame[frame["trailing"] >= high]["forward"].mean())
        rows.append({
            "horizon_years": int(h),
            "observations": int(len(frame)),
            # Rank first, and it is the one to read. See RANK_NOTE.
            "correlation": float(frame["trailing"].corr(frame["forward"],
                                                        method="spearman")),
            "pearson": float(frame["trailing"].corr(frame["forward"])),
            "forward_low_inflation": calm,
            "forward_high_inflation": hot,
            "gap": hot - calm,
        })
    return pd.DataFrame.from_records(rows)


def predictive_grid(panel: Any, windows: Sequence[int] = DEFAULT_WINDOWS,
                    assets: Sequence[str] = FORWARD_ASSETS,
                    horizons: Sequence[int] = (1, 5, 10, 30),
                    ) -> pd.DataFrame:
    """Every lookback window against every asset's forward real return.

    Inflation itself is carried as a fifth "asset" because it is the
    mechanism: trailing inflation can only predict returns if it predicts
    inflation, and reporting that link beside the return columns lets a reader
    see whether the story holds together.
    """
    frames: List[pd.DataFrame] = []
    for window in windows:
        trailing = trailing_inflation(panel.inflation, int(window))
        targets = {a: panel.series(a) for a in assets}
        targets["inflation"] = panel.inflation
        for name, forward in targets.items():
            block = predictive_power(trailing, forward, horizons)
            if not len(block):
                continue
            block.insert(0, "asset", name)
            block.insert(0, "window_years", int(window))
            frames.append(block)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def window_choice(grid: pd.DataFrame, asset: str = "dom_eq",
                  horizon: int = 30) -> pd.DataFrame:
    """Which lookback window carries the most signal, at one asset and horizon.

    Reported rather than chosen by search: with three candidates and one
    panel, picking the best-performing window and then reporting its
    performance would be a selection effect wearing a result. The headline
    uses whichever window the config names, and this table says what the
    others would have given.
    """
    block = grid[(grid["asset"] == asset)
                 & (grid["horizon_years"] == int(horizon))]
    return block[["window_years", "observations", "correlation", "pearson",
                  "forward_low_inflation", "forward_high_inflation",
                  "gap"]].sort_values("window_years").reset_index(drop=True)


def persistence(grid: pd.DataFrame, window: int) -> Dict[str, Any]:
    """The mechanism check: does trailing inflation predict future inflation?"""
    block = grid[(grid["asset"] == "inflation")
                 & (grid["window_years"] == int(window))]
    if not len(block):
        return {"measured": False}
    short = block.sort_values("horizon_years").iloc[0]
    long = block.sort_values("horizon_years").iloc[-1]
    return {
        "measured": True,
        "short_horizon": int(short["horizon_years"]),
        "short_correlation": float(short["correlation"]),
        "long_horizon": int(long["horizon_years"]),
        "long_correlation": float(long["correlation"]),
        "persistent": bool(float(short["correlation"]) > 0.2),
        "decays": bool(float(long["correlation"]) < float(short["correlation"])),
    }


def asset_ordering(grid: pd.DataFrame, window: int, horizon: int
                   ) -> pd.DataFrame:
    """Forward real returns by asset in the calm and hot thirds, side by side.

    This is the table the allocation result has to be consistent with. If
    inflation hurts the nominal legs more than equity, the optimal portfolio
    should move toward equity; if it hurts everything equally, it should not
    move at all and the section is a level story like #valuation.
    """
    block = grid[(grid["window_years"] == int(window))
                 & (grid["horizon_years"] == int(horizon))
                 & (grid["asset"] != "inflation")].copy()
    return block[["asset", "correlation", "forward_low_inflation",
                  "forward_high_inflation", "gap"]].sort_values(
                      "gap").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------
# The bucketing, subsetting and per-bucket scoring are the same operations
# Section #valuation performs on the dividend yield, so they are imported
# rather than reimplemented: a second copy could drift, and the two sections
# comparing lifetimes the same way is part of what makes them comparable.
bucket_paths = vln.bucket_paths
expanding_cuts = vln.expanding_cuts
expanding_bucket_paths = vln.expanding_bucket_paths
bucket_agreement = vln.bucket_agreement
path_start_cells = vln.path_start_cells
by_bucket = vln.by_bucket
advantage_by_bucket = vln.advantage_by_bucket
locate = vln.locate
MIN_HISTORY = vln.MIN_HISTORY


def path_inflation_at(paths: Any, trailing: np.ndarray,
                      offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """The trailing inflation each lifetime faced at one age, and the year.

    ``offset`` is years since the lifetime began: ``0`` is the morning the
    investor started saving, ``spec.n_working`` the morning they retired. Both
    are observable *at that moment* -- the trailing rate uses only years
    already finished -- so conditioning on either involves no look-ahead for
    somebody standing there. What differs is who can act on it, which is the
    whole point of asking the question twice.
    """
    offset = int(offset)
    calendar = np.asarray(paths.calendar_index)
    country = np.asarray(paths.domestic_country)
    if not 0 <= offset < calendar.shape[1]:
        raise ValueError(
            f"offset {offset} is outside the simulated horizon "
            f"{calendar.shape[1]}")
    year = calendar[:, offset]
    return trailing[year, country[:, offset]], year


def path_starting_inflation(paths: Any, trailing: np.ndarray) -> np.ndarray:
    """The trailing inflation each simulated lifetime began at.

    A path is a chain of calendar windows and only the first is a starting
    condition, so the state variable is the trailing rate at the first drawn
    country-year -- what the investor would have read in the paper the morning
    they started saving.
    """
    return path_inflation_at(paths, trailing, 0)[0]


def source_comparison(domestic: np.ndarray, starting_domestic: np.ndarray,
                      starting_global: np.ndarray,
                      edges: Sequence[float],
                      labels: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Is the signal a country's own inflation, or the era's?

    Bucketing the same lifetimes by their own market's trailing inflation and
    by the leave-one-out average of everybody else's. High agreement means the
    section is really about inflationary *periods*, which the block bootstrap
    already samples jointly; low agreement means the domestic rate carries
    something of its own.
    """
    own_index, own_meta = bucket_paths(starting_domestic, edges, labels)
    world_index, world_meta = bucket_paths(starting_global, edges, labels)
    both = (own_index >= 0) & (world_index >= 0)
    agree = float((own_index[both] == world_index[both]).mean() * 100.0) \
        if both.any() else float("nan")
    finite = np.isfinite(starting_domestic) & np.isfinite(starting_global)
    correlation = float(pd.Series(starting_domestic[finite]).corr(
        pd.Series(starting_global[finite]), method="spearman")) \
        if finite.sum() > 2 else float("nan")
    rows = [{
        "source": "domestic (used)",
        "cut_low": own_meta["cuts"][0] if own_meta["cuts"] else float("nan"),
        "cut_high": own_meta["cuts"][-1] if own_meta["cuts"] else float("nan"),
        **{f"n_{lab}": n for lab, n in zip(labels, own_meta["counts"])},
    }, {
        "source": "rest of the world (check)",
        "cut_low": world_meta["cuts"][0] if world_meta["cuts"] else float("nan"),
        "cut_high": world_meta["cuts"][-1] if world_meta["cuts"] else float("nan"),
        **{f"n_{lab}": n for lab, n in zip(labels, world_meta["counts"])},
    }]
    return pd.DataFrame.from_records(rows), {
        "agreement_pct": agree,
        "correlation": correlation,
        "reassigned": int((~(own_index == world_index) & both).sum()),
        "n_compared": int(both.sum()),
    }


def current_position(trailing: np.ndarray, years: np.ndarray,
                     countries: Sequence[str], iso: str = "USA",
                     ) -> Dict[str, Any]:
    """Where the panel's last observation sits, so a reader can place themselves."""
    j = list(countries).index(iso) if iso in countries else 0
    column = trailing[:, j]
    finite = np.flatnonzero(np.isfinite(column))
    if not finite.size:
        return {"iso": iso, "year": -1, "trailing_inflation": float("nan"),
                "percentile": float("nan"),
                "panel_median": float(np.nanmedian(trailing))}
    last = int(finite[-1])
    return {
        "iso": iso,
        "year": int(np.asarray(years)[last]),
        "trailing_inflation": float(column[last]),
        "percentile": locate(float(column[last]), trailing),
        "panel_median": float(np.nanmedian(trailing)),
    }


# ---------------------------------------------------------------------------
# The optimal portfolio, bucket by bucket
# ---------------------------------------------------------------------------
#: Equity shares swept when asking how much equity each inflation regime wants.
DEFAULT_EQUITY_GRID: Tuple[float, ...] = tuple(round(x / 10, 2)
                                               for x in range(0, 11))

#: Domestic shares of the equity sleeve, for the second question: a domestic
#: inflation shock is a domestic phenomenon, so the foreign leg has a claim to
#: be a hedge against it.
DEFAULT_DOMESTIC_GRID: Tuple[float, ...] = tuple(round(x / 10, 2)
                                                 for x in range(0, 11))


def equity_share_strategies(spec: Any, shares: Sequence[float] = DEFAULT_EQUITY_GRID,
                            domestic_share: float = 0.5,
                            bond_share: float = 0.7
                            ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Constant-weight portfolios along the equity/fixed-income axis.

    The composition *within* each sleeve is held at the baseline -- equity
    split ``domestic_share``/rest, fixed income split ``bond_share``/rest --
    so the only thing moving along this grid is how much equity is held. That
    is what makes the argmax in each bucket interpretable as an answer to "how
    much equity does this regime want".
    """
    from . import lifecycle as lc
    out: Dict[str, Any] = {}
    parameter: Dict[str, float] = {}
    for share in shares:
        share = float(share)
        row = np.array([
            share * domestic_share,
            share * (1.0 - domestic_share),
            (1.0 - share) * bond_share,
            (1.0 - share) * (1.0 - bond_share),
        ])
        key = f"equity_{int(round(share * 100)):03d}"
        out[key] = lc.Strategy(key=key, label=f"{share:.0%} equity",
                               weights=np.tile(row, (spec.horizon, 1)))
        parameter[key] = share
    return out, parameter


def domestic_share_strategies(spec: Any,
                              shares: Sequence[float] = DEFAULT_DOMESTIC_GRID,
                              ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """All-equity portfolios along the home/abroad axis.

    Equity is held at 100% so that the only thing moving is where it sits.
    The question this answers is the one the paper is really about, asked of
    an inflation regime rather than of the whole panel.
    """
    from . import lifecycle as lc
    out: Dict[str, Any] = {}
    parameter: Dict[str, float] = {}
    for share in shares:
        share = float(share)
        row = np.array([share, 1.0 - share, 0.0, 0.0])
        key = f"domestic_{int(round(share * 100)):03d}"
        out[key] = lc.Strategy(key=key, label=f"{share:.0%} domestic equity",
                               weights=np.tile(row, (spec.horizon, 1)))
        parameter[key] = share
    return out, parameter


def optimum_by_bucket(frame: pd.DataFrame, parameter: Mapping[str, float],
                      column: str, name: str,
                      group: str = "bucket") -> pd.DataFrame:
    """The certainty-equivalent-maximising point of a swept grid, per bucket.

    Reports the interior optimum and the curvature around it, because a flat
    maximum and a sharp one are different findings: a grid whose best and
    second-best differ by a rounding error has not identified anything, and
    the table should let a reader see that rather than hiding it behind an
    argmax.

    ``group`` names the column the sweep was run across -- inflation terciles
    here, withholding rates in Section #withholding. The two studies ask the
    same question of the same grids, so they share the arithmetic.
    """
    rows: List[Dict[str, Any]] = []
    for label, block in frame.groupby(group, sort=False):
        scored = block[block["strategy"].isin(parameter)].copy()
        if not len(scored):
            continue
        scored[name] = [parameter[k] for k in scored["strategy"]]
        scored = scored.sort_values(name).reset_index(drop=True)
        best = scored.loc[scored[column].idxmax()]
        top = float(best[column])
        worst = float(scored[column].min())
        runners = scored[scored["strategy"] != best["strategy"]]
        second = float(runners[column].max()) if len(runners) else float("nan")
        # The two ends of the grid are the interpretable reference portfolios
        # -- all-international and all-domestic on one axis, all-bonds and
        # all-equity on the other -- and the margin over them is the question
        # a reader actually has. The runner-up margin answers a different and
        # much narrower one: whether the grid can tell one step from the next.
        low_end = float(scored[column].iloc[0])
        high_end = float(scored[column].iloc[-1])
        rows.append({
            group: label,
            "n_paths": int(best["n_paths"]),
            f"optimal_{name}": float(best[name]),
            "cec_at_optimum": top,
            "cec_at_worst": worst,
            "cec_at_low_end": low_end,
            "cec_at_high_end": high_end,
            "range_pct": (top / worst - 1.0) * 100.0 if worst > 0 else float("nan"),
            "margin_over_low_end_pct": (top / low_end - 1.0) * 100.0
            if low_end > 0 else float("nan"),
            "margin_over_high_end_pct": (top / high_end - 1.0) * 100.0
            if high_end > 0 else float("nan"),
            "margin_over_runner_up_pct": (top / second - 1.0) * 100.0
            if np.isfinite(second) and second > 0 else float("nan"),
            "at_grid_edge": bool(float(best[name]) in
                                 (float(scored[name].min()),
                                  float(scored[name].max()))),
        })
    return pd.DataFrame.from_records(rows)


def optimum_shift(optima: pd.DataFrame, name: str,
                  labels: Sequence[str] = BUCKET_LABELS) -> Dict[str, Any]:
    """How far the optimum moves between the calm and the hot bucket."""
    if not len(optima):
        return {"measured": False}
    indexed = optima.set_index("bucket")
    low, high = str(labels[0]), str(labels[-1])
    if low not in indexed.index or high not in indexed.index:
        return {"measured": False}
    a = float(indexed.loc[low, f"optimal_{name}"])
    b = float(indexed.loc[high, f"optimal_{name}"])
    margins = optima["margin_over_runner_up_pct"]
    return {
        "measured": True,
        "low_bucket": low,
        "high_bucket": high,
        f"optimal_{name}_low": a,
        f"optimal_{name}_high": b,
        "shift": b - a,
        "moves": bool(abs(b - a) > 1e-9),
        # A shift is only a finding if the grid could tell the difference. If
        # the winner beats the runner-up by less than the shift is worth, the
        # optimum has not really moved -- the surface is just flat.
        "smallest_margin_pct": float(margins.min()) if len(margins) else float("nan"),
        "identified": bool(len(margins) and float(margins.min()) > 0.05),
        "any_at_grid_edge": bool(optima["at_grid_edge"].any()),
        # Whether the interior optimum is worth anything against the corner a
        # reader would otherwise have held. A flat ridge between neighbouring
        # grid points can still sit well above both ends of the grid, and
        # reporting only the runner-up margin would hide that.
        "beats_low_end_by_pct": (float(optima["margin_over_low_end_pct"].min())
                                 if "margin_over_low_end_pct" in optima
                                 else float("nan")),
        "beats_low_end": bool("margin_over_low_end_pct" in optima
                              and float(optima["margin_over_low_end_pct"].min())
                              > 0.05),
    }


def verdict(advantage: pd.DataFrame, grid: pd.DataFrame, window: int,
            horizon: int, equity: Mapping[str, Any],
            domestic: Mapping[str, Any],
            persist: Mapping[str, Any]) -> Dict[str, Any]:
    """What conditioning on inflation does, classified from the tables."""
    ordering = asset_ordering(grid, window, horizon)
    nominal = ordering[ordering["asset"].isin(("bond", "bill"))]["gap"]
    equities = ordering[ordering["asset"].isin(("dom_eq", "intl_eq"))]["gap"]
    found: Dict[str, Any] = {
        "window_years": int(window),
        "horizon_years": int(horizon),
        "inflation_is_persistent": bool(persist.get("persistent", False)),
        "persistence_correlation": float(persist.get("short_correlation",
                                                     float("nan"))),
        # The hypothesis the section was built on, tested rather than assumed.
        "nominal_legs_hurt_more": bool(
            len(nominal) and len(equities)
            and float(nominal.mean()) < float(equities.mean())),
        "nominal_gap_pp": float(nominal.mean()) * 100.0 if len(nominal) else float("nan"),
        "equity_gap_pp": float(equities.mean()) * 100.0 if len(equities) else float("nan"),
        "worst_asset": str(ordering["asset"].iloc[0]) if len(ordering) else "",
        "best_asset": str(ordering["asset"].iloc[-1]) if len(ordering) else "",
    }
    if len(advantage):
        leads = advantage["advantage_pct"]
        found.update({
            "buckets": int(len(advantage)),
            "lead_low_pct": float(leads.iloc[0]),
            "lead_high_pct": float(leads.iloc[-1]),
            "lead_spread_pp": float(leads.max() - leads.min()),
            "lead_positive_everywhere": bool((leads > 0).all()),
            "ranking_survives": bool((leads > 0).all()),
        })
    found.update({f"equity_{k}": v for k, v in equity.items()})
    found.update({f"domestic_{k}": v for k, v in domestic.items()})
    return found


# ---------------------------------------------------------------------------
# The retiree's problem
# ---------------------------------------------------------------------------
def after_retirement(spec: Any, accumulation: np.ndarray,
                     swept: Mapping[str, Any],
                     prefix: str = "ret") -> Tuple[Dict[str, Any],
                                                   Dict[str, float]]:
    """Portfolios identical until retirement and swept afterwards.

    Sweeping a *lifetime* allocation answers a question no retiree can act on:
    they cannot go back and hold something else from twenty-five. What they can
    choose is what to hold from the day they stop working, with whatever
    accumulation they arrived with. These strategies are that choice --
    ``accumulation`` weights for the working years, the swept weights after --
    so the argmax in each bucket is an instruction a retiree could follow.

    ``swept`` is any mapping of key to strategy, so the same equity-share and
    domestic-share grids used for the lifetime question serve here unchanged.
    """
    from . import lifecycle as lc

    base = np.asarray(accumulation, dtype=float)
    if base.shape[0] != spec.horizon:
        raise ValueError(
            f"accumulation weights cover {base.shape[0]} years, not the "
            f"lifecycle's {spec.horizon}")
    out: Dict[str, Any] = {}
    parameter: Dict[str, float] = {}
    for key, strat in swept.items():
        weights = np.vstack([base[:spec.n_working],
                             np.asarray(strat.weights)[spec.n_working:]])
        name = f"{prefix}_{key}"
        out[name] = lc.Strategy(key=name,
                                label=f"{strat.label}, from retirement",
                                weights=weights)
        parameter[name] = float(key.rsplit("_", 1)[-1]) / 100.0
    return out, parameter


def level_spread(frame: pd.DataFrame, strategy: str, column: str,
                 labels: Sequence[str] = BUCKET_LABELS) -> Dict[str, Any]:
    """How much the *level* of retirement consumption moves across buckets.

    Distinct from the ranking question and, for a retiree, the more pressing
    one: not "does inflation change what I should hold" but "how much worse
    off am I for retiring into it". Reported for one strategy so the number is
    a consumption difference rather than a mixture of portfolio effects.
    """
    block = frame[frame["strategy"] == strategy].set_index("bucket")
    present = [str(x) for x in labels if str(x) in block.index]
    if len(present) < 2:
        return {"measured": False}
    values = {name: float(block.loc[name, column]) for name in present}
    low, high = values[present[0]], values[present[-1]]
    best, worst = max(values.values()), min(values.values())
    return {
        "measured": True,
        "strategy": strategy,
        "low_bucket": present[0], "high_bucket": present[-1],
        "cec_low": low, "cec_high": high,
        "high_over_low_pct": (high / low - 1.0) * 100.0 if low else float("nan"),
        "spread_pct": (best / worst - 1.0) * 100.0 if worst else float("nan"),
        "high_inflation_is_worse": bool(high < low),
    }


def timing_comparison(birth: Mapping[str, Any], retirement: Mapping[str, Any],
                      ) -> Dict[str, Any]:
    """Birth-date conditioning against retirement-date conditioning.

    The point of running both. A sixty-eight-year lifetime averages a
    short-horizon shock away; a thirty-year decumulation cannot. If the level
    spread is materially wider when the state variable is read at retirement,
    the null in the lifetime version was about the horizon rather than about
    inflation.
    """
    if not (birth.get("measured") and retirement.get("measured")):
        return {"measured": False}
    a = abs(float(birth["high_over_low_pct"]))
    b = abs(float(retirement["high_over_low_pct"]))
    return {
        "measured": True,
        "birth_spread_pct": float(birth["high_over_low_pct"]),
        "retirement_spread_pct": float(retirement["high_over_low_pct"]),
        "ratio": b / a if a > 1e-9 else float("inf"),
        "retirement_matters_more": bool(b > a),
        # A doubling is the threshold at which the horizon story stops being
        # a quibble and starts being the finding.
        "retirement_matters_much_more": bool(b > 2.0 * a),
        "same_sign": bool((float(birth["high_over_low_pct"]) > 0)
                          == (float(retirement["high_over_low_pct"]) > 0)),
    }
