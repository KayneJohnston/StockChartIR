"""What if your salary is already a claim on your own stock market?

Every result before this section assumes labour income and the domestic
market are independent. That assumption is a gift to domestic equity, and an
implausible one. A worker in a small open economy whose export sector is the
index is long that index twice over: once in the portfolio and once in the
pay cheque. The textbook reason to hold *less* of your home market than the
market-cap weight is exactly this, and the model has been assuming it away.

This section switches it on and sweeps it. ``rho`` is the correlation between
the permanent innovation to labour income and the domestic equity return of
the same year -- and it is a *correlation*, not a loading, so raising it
re-labels which part of career risk is systematic without changing how much
career risk there is. That separation is what makes the sweep readable: any
movement in the certainty equivalent is the correlation, not extra income
volatility smuggled in alongside it.

**What to expect, and what would be surprising.** Correlated human capital
should push against domestic equity, so the lead of the all-international
portfolio should *widen* with ``rho``. That direction is not the interesting
part -- it is close to arithmetic. Two things would be worth knowing:

* how large ``rho`` has to be before the effect is comparable with the other
  levers in this paper, and
* whether the *ranking* ever changes, which would mean the headline was
  resting on the independence assumption rather than merely being flattered
  by it.

**What this does not model.** Income here is a hump-shaped profile with
permanent and transitory shocks; it has no unemployment spells, no
industry, and no relationship to the international sleeve. A correlation
with the *foreign* market would push the other way, and it is not zero in
reality either. The sweep should be read as a bound on one channel, not as a
calibrated model of human capital.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Correlations swept by default. Zero is the headline assumption; the top of
#: the range is deliberately past anything a labour economist would defend,
#: because the distance to a reversal is the margin of safety.
DEFAULT_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)


def specs(spec: Any, grid: Sequence[float] = DEFAULT_GRID
          ) -> Dict[float, Any]:
    """One :class:`~src.lifecycle.LifecycleSpec` per swept correlation."""
    return {float(rho): dataclasses.replace(
        spec, income_return_correlation=float(rho)) for rho in grid}


def sweep(summarise: Callable[[Any, float], pd.DataFrame], spec: Any,
          grid: Sequence[float] = DEFAULT_GRID) -> pd.DataFrame:
    """Re-run the headline at each correlation, tagged with the value used.

    ``summarise`` takes a spec and the correlation and returns the usual
    per-strategy summary frame, so the arithmetic behind these rows is the
    same arithmetic behind the headline table rather than a copy of it.
    """
    frames: List[pd.DataFrame] = []
    for rho, tweaked in specs(spec, grid).items():
        LOGGER.info("human capital: correlation %.2f with the home market", rho)
        block = summarise(tweaked, rho)
        block.insert(0, "correlation", rho)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def gap_curve(frame: pd.DataFrame, pair: Tuple[str, str]) -> pd.DataFrame:
    """The lead of one strategy over another at each correlation."""
    cec = _cec_column(frame)
    rows: List[Dict[str, Any]] = []
    for rho in sorted(frame["correlation"].unique()):
        block = frame[np.isclose(frame["correlation"], rho)]
        values = {r["strategy"]: float(r[cec]) for _, r in block.iterrows()}
        ordered = sorted(values, key=values.get, reverse=True)
        rows.append({
            "correlation": float(rho),
            "gap_pct": (values[pair[0]] / values[pair[1]] - 1.0) * 100.0
            if pair[0] in values and pair[1] in values else float("nan"),
            f"cec_{pair[0]}": values.get(pair[0], float("nan")),
            f"cec_{pair[1]}": values.get(pair[1], float("nan")),
            "winner": ordered[0] if ordered else "",
            "domestic_rank": (ordered.index("domestic_equity") + 1
                              if "domestic_equity" in ordered else -1),
        })
    return pd.DataFrame.from_records(rows)


def ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Every strategy's certainty equivalent at every correlation, wide."""
    cec = _cec_column(frame)
    wide = frame.pivot_table(index=["strategy", "label"],
                             columns="correlation", values=cec)
    return wide.reset_index()


def sensitivity(curve: pd.DataFrame) -> Dict[str, Any]:
    """How much the lead moves per 0.1 of correlation, by least squares."""
    x = curve["correlation"].to_numpy(dtype=float)
    y = curve["gap_pct"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return {"slope_per_10pp": float("nan"), "r_squared": float("nan")}
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    fitted = slope * x[ok] + intercept
    resid = y[ok] - fitted
    total = y[ok] - y[ok].mean()
    r2 = 1.0 - float(resid @ resid) / float(total @ total) if total.any() else 1.0
    return {
        "slope_per_10pp": float(slope) * 0.1,
        "intercept_pct": float(intercept),
        "r_squared": float(r2),
    }


def verdict(curve: pd.DataFrame, fitted: Mapping[str, Any],
            pair: Tuple[str, str]) -> Dict[str, Any]:
    """What the correlation does to the ranking, classified from the sweep."""
    if not len(curve):
        return {"levels": 0}
    base = curve.loc[curve["correlation"].idxmin()]
    top = curve.loc[curve["correlation"].idxmax()]
    winners = set(str(w) for w in curve["winner"])
    return {
        "levels": int(len(curve)),
        "baseline_gap_pct": float(base["gap_pct"]),
        "highest_correlation": float(top["correlation"]),
        "gap_at_highest_pct": float(top["gap_pct"]),
        "change_pp": float(top["gap_pct"] - base["gap_pct"]),
        "widens_with_correlation": bool(top["gap_pct"] > base["gap_pct"]),
        "monotone": bool(curve["gap_pct"].is_monotonic_increasing
                         or curve["gap_pct"].is_monotonic_decreasing),
        "winner_ever_changes": bool(len(winners) > 1),
        "winners_seen": sorted(winners),
        "winner_is_expected_throughout": bool(winners == {pair[0]}),
        "domestic_ever_improves_rank": bool(
            curve["domestic_rank"].iloc[-1] < curve["domestic_rank"].iloc[0]),
        "slope_per_10pp": float(fitted.get("slope_per_10pp", float("nan"))),
        # The whole sweep against one of the paper's other levers, so the
        # reader can place it rather than being handed a bare number.
        "change_is_small": bool(abs(float(top["gap_pct"] - base["gap_pct"]))
                                < 1.0),
    }
