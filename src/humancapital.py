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


#: The three readings of "human capital is correlated with the market", which
#: differ in what they assume about the *foreign* market and disagree about
#: the answer as a result.
#:
#: ``home``
#:     Load on the domestic market and let the foreign correlation fall where
#:     the two markets' own correlation puts it. This is the sweep the section
#:     was built around, and the mildest of the three.
#: ``strict``
#:     Demand a pay cheque correlated with home and *uncorrelated with abroad*.
#:     Because the markets move together this needs a negative foreign loading,
#:     so it is the most favourable case the objection allows -- the purest
#:     possible version of "your salary is a claim on your own market".
#: ``diagonal``
#:     Correlate the pay cheque equally with both markets. This is the
#:     objection: if labour income is a claim on world equity rather than on
#:     the home market specifically, there is no home market to tilt away
#:     from, and whatever the correlation was buying should largely cancel.
MODES: Tuple[str, ...] = ("home", "strict", "diagonal")

MODE_LABEL: Dict[str, str] = {
    "home": "home market only (foreign correlation left free)",
    "strict": "home market only (foreign correlation pinned to zero)",
    "diagonal": "both markets equally",
}


def specs(spec: Any, grid: Sequence[float] = DEFAULT_GRID,
          mode: str = "home") -> Dict[float, Any]:
    """One :class:`~src.lifecycle.LifecycleSpec` per swept correlation.

    ``mode`` picks which of :data:`MODES` is being asked. ``home`` leaves the
    foreign correlation unspecified, which is what every figure in the
    original sweep used and what keeps those results bit-identical.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    out: Dict[float, Any] = {}
    for rho in grid:
        rho = float(rho)
        intl = (None if mode == "home"
                else 0.0 if mode == "strict" else rho)
        out[rho] = dataclasses.replace(
            spec, income_return_correlation=rho,
            income_intl_correlation=intl)
    return out


def sweep(summarise: Callable[[Any, float], pd.DataFrame], spec: Any,
          grid: Sequence[float] = DEFAULT_GRID,
          mode: str = "home") -> pd.DataFrame:
    """Re-run the headline at each correlation, tagged with the value used.

    ``summarise`` takes a spec and the correlation and returns the usual
    per-strategy summary frame, so the arithmetic behind these rows is the
    same arithmetic behind the headline table rather than a copy of it.
    """
    frames: List[pd.DataFrame] = []
    for rho, tweaked in specs(spec, grid, mode).items():
        LOGGER.info("human capital (%s): correlation %.2f", mode, rho)
        block = summarise(tweaked, rho)
        block.insert(0, "correlation", rho)
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "mode", mode)
    return out


def sweep_modes(summarise: Callable[[Any, float, str], pd.DataFrame], spec: Any,
                grid: Sequence[float] = DEFAULT_GRID,
                modes: Sequence[str] = MODES) -> pd.DataFrame:
    """The same sweep under each reading of the correlation, stacked."""
    return pd.concat(
        [sweep(lambda s, r, m=mode: summarise(s, r, m), spec, grid, mode)
         for mode in modes], ignore_index=True)


def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def gap_curve(frame: pd.DataFrame, pair: Tuple[str, str]) -> pd.DataFrame:
    """The lead of one strategy over another at each correlation.

    Where the frame carries several readings of the correlation (see
    :data:`MODES`) each is curved separately, because comparing them is the
    point: the answer is not the same under all three.
    """
    cec = _cec_column(frame)
    modes = (list(dict.fromkeys(frame["mode"])) if "mode" in frame.columns
             else [None])
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        block_all = frame if mode is None else frame[frame["mode"] == mode]
        for rho in sorted(block_all["correlation"].unique()):
            block = block_all[np.isclose(block_all["correlation"], rho)]
            values = {r["strategy"]: float(r[cec]) for _, r in block.iterrows()}
            ordered = sorted(values, key=values.get, reverse=True)
            row: Dict[str, Any] = {
                "correlation": float(rho),
                "gap_pct": (values[pair[0]] / values[pair[1]] - 1.0) * 100.0
                if pair[0] in values and pair[1] in values else float("nan"),
                f"cec_{pair[0]}": values.get(pair[0], float("nan")),
                f"cec_{pair[1]}": values.get(pair[1], float("nan")),
                "winner": ordered[0] if ordered else "",
                "domestic_rank": (ordered.index("domestic_equity") + 1
                                  if "domestic_equity" in ordered else -1),
            }
            if mode is not None:
                row = {"mode": mode, **row}
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def mode_comparison(curve: pd.DataFrame, pair: Tuple[str, str]
                    ) -> pd.DataFrame:
    """One row per reading: where the lead starts, ends, and how far it moved.

    This is the table the objection is answered from. If the ``diagonal`` row
    moves far less than the ``home`` row, the widening reported in this section
    was a statement about the *home* market specifically, and a pay cheque that
    is a claim on world equity does not deliver it.
    """
    if "mode" not in curve.columns:
        curve = curve.assign(mode="home")
    rows: List[Dict[str, Any]] = []
    for mode in dict.fromkeys(curve["mode"]):
        block = curve[curve["mode"] == mode].sort_values("correlation")
        if not len(block):
            continue
        lo, hi = block.iloc[0], block.iloc[-1]
        rows.append({
            "mode": str(mode),
            "label": MODE_LABEL.get(str(mode), str(mode)),
            "correlation_low": float(lo["correlation"]),
            "correlation_high": float(hi["correlation"]),
            "gap_low_pct": float(lo["gap_pct"]),
            "gap_high_pct": float(hi["gap_pct"]),
            "change_pp": float(hi["gap_pct"] - lo["gap_pct"]),
            "winner_changes": bool(block["winner"].nunique() > 1),
            "winner_at_high": str(hi["winner"]),
        })
    out = pd.DataFrame.from_records(rows)
    if len(out) and "home" in set(out["mode"]):
        anchor = float(out.loc[out["mode"] == "home", "change_pp"].iloc[0])
        out["share_of_home_effect"] = (out["change_pp"] / anchor
                                       if anchor else float("nan"))
    return out


def ranking(frame: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Every strategy's certainty equivalent at every correlation, wide.

    ``mode`` selects one reading out of a stacked sweep; without it a stacked
    frame would silently average the three, which is not a number anybody
    wants.
    """
    if mode is not None and "mode" in frame.columns:
        frame = frame[frame["mode"] == mode]
    elif "mode" in frame.columns and frame["mode"].nunique() > 1:
        raise ValueError("frame carries several modes; pass one to `mode`")
    cec = _cec_column(frame)
    wide = frame.pivot_table(index=["strategy", "label"],
                             columns="correlation", values=cec)
    return wide.reset_index()


def sensitivity(curve: pd.DataFrame, mode: str | None = None) -> Dict[str, Any]:
    """How much the lead moves per 0.1 of correlation, by least squares."""
    if mode is not None and "mode" in curve.columns:
        curve = curve[curve["mode"] == mode]
    elif "mode" in curve.columns and curve["mode"].nunique() > 1:
        raise ValueError("curve carries several modes; pass one to `mode`")
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
            pair: Tuple[str, str], mode: str | None = None,
            comparison: pd.DataFrame | None = None) -> Dict[str, Any]:
    """What the correlation does to the ranking, classified from the sweep."""
    extra: Dict[str, Any] = {}
    if comparison is not None and len(comparison):
        rows = comparison.set_index("mode")
        extra["modes_run"] = sorted(rows.index)
        for name in rows.index:
            extra[f"change_{name}_pp"] = float(rows.loc[name, "change_pp"])
        if {"home", "diagonal"} <= set(rows.index):
            home = float(rows.loc["home", "change_pp"])
            diag = float(rows.loc["diagonal", "change_pp"])
            extra.update({
                "diagonal_share_of_home": (diag / home if home else float("nan")),
                # The objection, answered: if correlating the pay cheque with
                # *both* markets keeps most of the widening, the effect was
                # never about the home market's identity.  If it kills it, the
                # section is a statement about home bias and nothing wider.
                "diagonal_mostly_cancels": bool(
                    home != 0.0 and abs(diag) < 0.5 * abs(home)),
                "diagonal_same_sign": bool((diag > 0.0) == (home > 0.0)),
            })
        extra["winner_changes_in_any_mode"] = bool(
            comparison["winner_changes"].any())
    if mode is not None and "mode" in curve.columns:
        curve = curve[curve["mode"] == mode]
    if not len(curve):
        return {"levels": 0, **extra}
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
        **extra,
    }
