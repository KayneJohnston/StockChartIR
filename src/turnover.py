"""What the solved schedule costs to actually run.

Section #fees prices the expense ratio and finds the differential that would
undo the headline. It says nothing about the other cost of running a
portfolio, which is trading it, and that omission lands hardest on exactly the
part of the paper that is most exposed to it. A fixed 50/50 portfolio trades
because its assets drifted apart. A schedule solved age by age over the whole
weight simplex trades for that reason *and* because it decided to hold
something different this year -- and it was solved by an optimiser that was
never charged for the decision.

That is a real objection and this section answers it in three steps.

**Measure the turnover.** Rebalancing to a target is a trade, and the trade is
observable. Turnover is reported one-way -- half the sum of absolute weight
changes, so selling one asset to buy another counts once -- and split three
ways:

* **total**, the trade actually required each year;
* **drift**, what a portfolio holding last year's weights constant would have
  had to trade anyway, which is the part no schedule can avoid;
* **schedule**, the deterministic move the target made between two ages, which
  is what a schedule would cost in a world with no returns at all.

The three do not add up, and the reason is worth stating: a schedule that cuts
equity in a year equity outperformed is trading *with* the drift rather than
against it, so its total can be below its drift counterfactual. A glide path
is partly self-rebalancing, and that is a point in its favour that a naive
turnover count would miss.

**Charge for it.** The cost is proportional to the value turned over and is
taken at the rebalance, before the year's return compounds on what is left.
Sweeping it re-runs the whole comparison at each level, so the cost falls on
every strategy at once and the question is only ever which one it costs more.

**Find the price that matters.** The number to report is the break-even: the
trading cost at which the solved schedule's advantage over the best fixed
portfolio disappears. If that cost is above anything an index investor pays,
the objection is answered. If it is below, the optimisation sections have been
spending money they did not have, and this is where a reader finds out.

**What this does not model.** Costs are proportional and symmetric; there is
no bid-ask asymmetry, no market impact, and no minimum ticket. Contributions
during accumulation are invested pro-rata rather than steered toward the
underweight asset, which overstates turnover for a real saver making regular
contributions -- so the break-even here is, if anything, too easily reached.
Tax on realised gains is a cost of trading in a taxable account and is absent,
along with every other tax in this paper.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: One-way proportional costs to sweep, as fractions of the value traded.
#: The top of the grid is well past a liquid index fund and into the territory
#: of a retail investor trading small parcels by hand.
DEFAULT_COSTS: Tuple[float, ...] = (0.0, 0.0005, 0.0010, 0.0025, 0.0050, 0.0100)


def specs(spec: Any, costs: Sequence[float] = DEFAULT_COSTS
          ) -> Dict[float, Any]:
    """One :class:`~src.lifecycle.LifecycleSpec` per swept trading cost."""
    return {float(c): dataclasses.replace(spec, trading_cost=float(c))
            for c in costs}


# ---------------------------------------------------------------------------
# Measuring the trade
# ---------------------------------------------------------------------------
def schedule_turnover(weights: np.ndarray) -> np.ndarray:
    """One-way turnover the target itself demands, ignoring returns entirely.

    This is the schedule's own cost: what it would take to follow the plan if
    every asset returned the same thing every year, so nothing ever drifted.
    A constant-weight strategy scores zero here by construction.
    """
    moves = np.abs(np.diff(np.asarray(weights, dtype=float), axis=0)).sum(axis=1)
    return np.concatenate([[0.0], 0.5 * moves])


def drift_turnover(paths: Any, weights: np.ndarray) -> np.ndarray:
    """The trade a *stationary* target would still have required each year.

    At year ``h`` this asks what it would have cost to rebalance back to last
    year's weights rather than on to this year's -- the trade the investor
    could not have avoided by holding still. Subtracting it from the total
    isolates what moving the target actually added.

    It is emphatically not the turnover of a one-year-lagged schedule: that
    still carries every move the schedule makes, only a year late, and would
    make any schedule look free.
    """
    from src import lifecycle as lc

    horizon = int(np.asarray(weights).shape[0])
    stack = np.stack([paths.series(a)[:, :horizon] for a in lc.ASSETS], axis=-1)
    out = np.zeros((paths.n_paths, horizon))
    for h in range(1, horizon):
        anchor = np.asarray(weights, dtype=float)[h - 1]
        grown = anchor * (1.0 + stack[:, h - 1, :])
        total = grown.sum(axis=1, keepdims=True)
        drifted = np.divide(grown, total, out=np.zeros_like(grown),
                            where=total != 0.0)
        out[:, h] = 0.5 * np.abs(anchor - drifted).sum(axis=1)
    return out


def measure(paths: Any, strategies: Mapping[str, Any], spec: Any
            ) -> pd.DataFrame:
    """Turnover per strategy: total, the drift floor, and the schedule's own.

    ``drift`` is the counterfactual in which the target stays where it was --
    the trade the investor would have had to do anyway. The excess over it is
    what moving the target added, and it is signed: a negative excess means
    the schedule's move ran with the drift instead of against it.
    """
    from src import lifecycle as lc

    rows: List[Dict[str, Any]] = []
    for key, strat in strategies.items():
        total = lc.portfolio_turnover(paths, strat)
        drift = drift_turnover(paths, strat.weights)
        own = schedule_turnover(strat.weights)
        work = slice(0, spec.n_working)
        ret = slice(spec.n_working, spec.horizon)
        rows.append({
            "strategy": key,
            "label": str(strat.label),
            "turnover_total": float(total.mean()),
            "turnover_working": float(total[:, work].mean()),
            "turnover_retired": float(total[:, ret].mean()),
            "turnover_drift_only": float(drift.mean()),
            "turnover_schedule_only": float(own.mean()),
            "excess_over_drift": float(total.mean() - drift.mean()),
            "turnover_peak_year": float(total.mean(axis=0).max()),
            "lifetime_turnover": float(total.mean(axis=0).sum()),
        })
    return pd.DataFrame.from_records(rows).sort_values(
        "turnover_total", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Charging for it
# ---------------------------------------------------------------------------
def sweep(summarise: Callable[[Any, float], pd.DataFrame], spec: Any,
          costs: Sequence[float] = DEFAULT_COSTS) -> pd.DataFrame:
    """Re-run the comparison at each trading cost, tagged with the cost used."""
    frames: List[pd.DataFrame] = []
    for cost, tweaked in specs(spec, costs).items():
        LOGGER.info("turnover: charging %.0f bp of traded value", cost * 1e4)
        block = summarise(tweaked, cost)
        block.insert(0, "trading_cost", cost)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def best_fixed(frame: pd.DataFrame, exclude: str) -> str:
    """The best strategy other than the challenger, judged at zero cost.

    The comparison the section promises is against "the best fixed portfolio",
    and which portfolio that is should be settled before costs are introduced
    rather than re-chosen at every level -- otherwise the incumbent moves to
    meet the challenger and the break-even means nothing.
    """
    cec = _cec_column(frame)
    cheapest = float(frame["trading_cost"].min())
    block = frame[np.isclose(frame["trading_cost"], cheapest)]
    block = block[block["strategy"] != exclude]
    if not len(block):
        return ""
    return str(block.loc[block[cec].idxmax(), "strategy"])


def gap_curve(frame: pd.DataFrame, challenger: str,
              incumbent: str) -> pd.DataFrame:
    """The challenger's lead over the incumbent at each trading cost."""
    cec = _cec_column(frame)
    rows: List[Dict[str, Any]] = []
    for cost in sorted(frame["trading_cost"].unique()):
        block = frame[np.isclose(frame["trading_cost"], cost)]
        values = {r["strategy"]: float(r[cec]) for _, r in block.iterrows()}
        ordered = sorted(values, key=values.get, reverse=True)
        rows.append({
            "trading_cost": float(cost),
            "basis_points": float(cost) * 1e4,
            "gap_pct": (values[challenger] / values[incumbent] - 1.0) * 100.0
            if challenger in values and incumbent in values else float("nan"),
            f"cec_{challenger}": values.get(challenger, float("nan")),
            f"cec_{incumbent}": values.get(incumbent, float("nan")),
            "winner": ordered[0] if ordered else "",
        })
    return pd.DataFrame.from_records(rows)


def break_even(curve: pd.DataFrame, key: str = "basis_points") -> float:
    """The trading cost at which the lead first reaches zero.

    Linear interpolation between the straddling grid points. Infinity where
    the lead survives the whole grid, zero where it was never there -- both
    are answers, and both beat a missing value.
    """
    ordered = curve.sort_values(key).reset_index(drop=True)
    gaps = ordered["gap_pct"].to_numpy(dtype=float)
    levels = ordered[key].to_numpy(dtype=float)
    if not np.isfinite(gaps).any():
        return float("nan")
    if gaps[0] <= 0.0:
        return 0.0
    below = np.flatnonzero(gaps <= 0.0)
    if below.size == 0:
        return float("inf")
    i = int(below[0])
    lo_gap, hi_gap = gaps[i - 1], gaps[i]
    lo, hi = levels[i - 1], levels[i]
    if lo_gap == hi_gap:
        return float(hi)
    return float(lo + (hi - lo) * lo_gap / (lo_gap - hi_gap))


def cost_of_the_schedule(measured: pd.DataFrame, solved: str,
                         fixed: str) -> Dict[str, Any]:
    """How much more the solved schedule trades than the fixed portfolio does."""
    rows = measured.set_index("strategy")
    if solved not in rows.index or fixed not in rows.index:
        return {"measured": False}
    a, b = rows.loc[solved], rows.loc[fixed]
    fixed = float(b["turnover_total"])
    return {
        "measured": True,
        "solved_turnover": float(a["turnover_total"]),
        "fixed_turnover": fixed,
        "extra_turnover": float(a["turnover_total"]) - fixed,
        # A single-asset benchmark never rebalances at all, so the ratio is
        # undefined rather than large. Saying "it trades and this does not" is
        # the honest rendering; printing an infinity is not.
        "fixed_trades_nothing": bool(fixed <= 1e-12),
        "ratio": (float(a["turnover_total"]) / fixed if fixed > 1e-12
                  else float("inf")),
        "solved_is_self_rebalancing": bool(a["excess_over_drift"] < 0.0),
    }


def verdict(curve: pd.DataFrame, measured: pd.DataFrame, cross: Mapping[str, Any],
            challenger: str, incumbent: str) -> Dict[str, Any]:
    """What trading costs do to the solved schedule, classified from the sweep."""
    if not len(curve):
        return {"levels": 0}
    base = curve.loc[curve["basis_points"].idxmin()]
    top = curve.loc[curve["basis_points"].idxmax()]
    be = break_even(curve)
    winners = set(str(w) for w in curve["winner"])
    return {
        "levels": int(len(curve)),
        "baseline_gap_pct": float(base["gap_pct"]),
        "highest_cost_bp": float(top["basis_points"]),
        "gap_at_highest_pct": float(top["gap_pct"]),
        "change_pp": float(top["gap_pct"] - base["gap_pct"]),
        "break_even_bp": be,
        "survives_whole_grid": bool(np.isinf(be)),
        "survives_ten_bp": bool(be > 10.0),
        "winner_ever_changes": bool(len(winners) > 1),
        "winners_seen": sorted(winners),
        "busiest_strategy": str(measured["strategy"].iloc[0]) if len(measured)
        else "",
        "busiest_turnover": float(measured["turnover_total"].iloc[0])
        if len(measured) else float("nan"),
        "quietest_strategy": str(measured["strategy"].iloc[-1]) if len(measured)
        else "",
        "quietest_turnover": float(measured["turnover_total"].iloc[-1])
        if len(measured) else float("nan"),
        **{f"cross_{k}": v for k, v in cross.items()},
    }
