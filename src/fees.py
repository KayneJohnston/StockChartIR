"""What costs do to the headline, and how small a cost is enough to undo it.

Every return in this project is gross. Nobody earns a gross return. The
omission is defensible for a comparison between strategies drawn from the same
panel -- a cost common to all of them cancels -- but it is *not* defensible for
this project's one divergence from the study it re-implements, because the
strategies at stake hold different amounts of the expensive sleeve.

All-international pays the international fund's cost on the whole portfolio.
The 50/50 split pays it on half. So a fee **differential** between domestic and
international funds falls on the two strategies unequally, and it compounds
over sixty-eight years. The question this module answers is not whether fees
matter -- they obviously do -- but how large the differential has to be before
it cancels a lead of a few percent, and whether that number is inside or
outside the range a real investor has faced.

**How a fee is charged.** An expense ratio is levied on assets, not on
returns, so a gross real return ``r`` net of a fee ``f`` is
``(1 + r)(1 - f) - 1``, not ``r - f``. The difference is second-order in any
one year and not in sixty-eight of them, so the exact form is used.

**Where it is applied.** To the panel, before the bootstrap sees it. Every
downstream stage -- sampler, lifecycle, utility -- runs unchanged, and
``available`` is untouched, so a net panel draws exactly the blocks the gross
one draws and the comparison is paired rather than merely parallel.

**What this module does not model.** Trading costs, bid-ask spreads, taxes,
platform fees, and the fact that index funds did not exist for most of this
sample. The last of those is the important one: before roughly 1975 the only
way to hold a diversified foreign portfolio was expensive and often
impossible. A reader should treat the fee grid as a sensitivity, not as a
history of what was actually available.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import lifecycle as lc

LOGGER = logging.getLogger(__name__)

#: Panel series a fee can be charged on.
CHARGEABLE: Tuple[str, ...] = ("dom_eq", "intl_eq", "bond", "bill")

#: Reference expense ratios, as configured anchors rather than findings. The
#: modern pair is a large provider's total-market and total-international
#: index funds; the earlier pair is the same provider two decades before.
#: They are here to put the swept grid on a scale a reader recognises, and
#: they are assumptions -- nothing in this project's data can verify them.
ANCHORS: Dict[str, float] = {
    "modern index funds": 0.0005,
    "index funds circa 2000": 0.0019,
    "an active international fund": 0.0075,
}


def net_of_fee(gross: np.ndarray, fee: float) -> np.ndarray:
    """``(1 + r)(1 - f) - 1``: a fee on assets, not on returns.

    Charging ``r - f`` instead would understate the cost in good years and
    overstate it in bad ones. Over one year the difference is a rounding
    error; over sixty-eight it is not.
    """
    if fee < 0.0:
        raise ValueError("a negative fee is a subsidy; not modelled")
    return (1.0 + np.asarray(gross, dtype=float)) * (1.0 - float(fee)) - 1.0


def apply_fees(panel: dl.Panel, fees: Mapping[str, float]) -> dl.Panel:
    """A panel with each named series charged its fee.

    ``available`` is deliberately untouched, so every fee level admits the
    same blocks and the bootstrap draws the same calendar history for each.
    """
    unknown = set(fees) - set(CHARGEABLE)
    if unknown:
        raise ValueError(f"not chargeable series: {sorted(unknown)}")
    changed = {name: net_of_fee(getattr(panel, name), fee)
               for name, fee in fees.items() if fee}
    if not changed:
        return panel
    label = "+".join(f"{k}{v * 1e4:.0f}bp" for k, v in sorted(fees.items())
                     if v)
    return dataclasses.replace(panel, name=f"{panel.name}[{label}]", **changed)


def _tag(frame: pd.DataFrame, **columns: Any) -> pd.DataFrame:
    out = frame.copy()
    for i, (name, value) in enumerate(columns.items()):
        out.insert(i, name, value)
    return out


def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def gap(frame: pd.DataFrame, pair: Tuple[str, str]) -> float:
    """Percentage lead of the first strategy over the second."""
    cec = _cec_column(frame)
    rows = {r["strategy"]: float(r[cec]) for _, r in frame.iterrows()}
    if pair[0] not in rows or pair[1] not in rows:
        return float("nan")
    return (rows[pair[0]] / rows[pair[1]] - 1.0) * 100.0


# ---------------------------------------------------------------------------
# The two sweeps
# ---------------------------------------------------------------------------
def sweep_common(panel: dl.Panel, summarise: Callable[..., pd.DataFrame],
                 n_paths: int, grid: Sequence[float]) -> pd.DataFrame:
    """One fee, charged on every asset alike.

    This is the control. A cost common to all four sleeves is close to
    neutral between strategies that differ only in how they mix them, so a
    large movement here would mean the fee is doing something other than what
    it is supposed to.
    """
    frames: List[pd.DataFrame] = []
    for fee in grid:
        LOGGER.info("common fee: %.0f bp on every asset", float(fee) * 1e4)
        netted = apply_fees(panel, {name: float(fee) for name in CHARGEABLE})
        frames.append(_tag(summarise(netted, n_paths), fee=float(fee)))
    return pd.concat(frames, ignore_index=True)


def sweep_differential(panel: dl.Panel,
                       summarise: Callable[..., pd.DataFrame],
                       n_paths: int, grid: Sequence[float],
                       base_fee: float = 0.0) -> pd.DataFrame:
    """A fee charged on the international sleeve alone.

    This is the question. All-international pays it on everything, the 50/50
    split on half, so the differential falls on them unequally and compounds
    for a working life plus a retirement.
    """
    frames: List[pd.DataFrame] = []
    for extra in grid:
        LOGGER.info("differential: %.0f bp extra on the international sleeve",
                    float(extra) * 1e4)
        fees = {name: float(base_fee) for name in CHARGEABLE}
        fees["intl_eq"] = float(base_fee) + float(extra)
        frames.append(_tag(summarise(apply_fees(panel, fees), n_paths),
                           differential=float(extra)))
    return pd.concat(frames, ignore_index=True)


def gap_curve(frame: pd.DataFrame, key: str,
              pair: Tuple[str, str]) -> pd.DataFrame:
    """The lead of one strategy over another at each level of ``key``."""
    cec = _cec_column(frame)
    rows: List[Dict[str, Any]] = []
    for level in sorted(frame[key].unique()):
        block = frame[np.isclose(frame[key], level)]
        by_key = {r["strategy"]: float(r[cec]) for _, r in block.iterrows()}
        rows.append({
            key: float(level),
            "gap_pct": gap(block, pair),
            f"cec_{pair[0]}": by_key.get(pair[0], float("nan")),
            f"cec_{pair[1]}": by_key.get(pair[1], float("nan")),
            "leader": max(by_key, key=by_key.get) if by_key else "",
        })
    return pd.DataFrame.from_records(rows)


def break_even(curve: pd.DataFrame, key: str) -> float:
    """The level of ``key`` at which the lead first reaches zero.

    Linear interpolation between the two grid points that straddle the
    crossing. Returns infinity when the lead never closes on the grid, and
    zero when it is already gone at the bottom of it -- both are answers, and
    both are more useful than a missing value.
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


def anchor_table(curve: pd.DataFrame, key: str,
                 anchors: Mapping[str, float] = ANCHORS) -> pd.DataFrame:
    """The lead at each reference expense ratio, interpolated onto the grid."""
    ordered = curve.sort_values(key)
    levels = ordered[key].to_numpy(dtype=float)
    gaps = ordered["gap_pct"].to_numpy(dtype=float)
    rows = [{"label": label, "differential": float(value),
             "basis_points": float(value) * 1e4,
             "gap_pct": float(np.interp(float(value), levels, gaps)),
             "within_grid": bool(levels.min() <= value <= levels.max())}
            for label, value in anchors.items()]
    return pd.DataFrame.from_records(rows).sort_values("differential")


def ranking_at(frame: pd.DataFrame, key: str, level: float) -> pd.DataFrame:
    """Every strategy's certainty equivalent at one fee level."""
    cec = _cec_column(frame)
    block = frame[np.isclose(frame[key], level)]
    return block[["strategy", "label", cec, "prob_ruin"]].sort_values(
        cec, ascending=False).reset_index(drop=True)


def verdict(common: pd.DataFrame, differential: pd.DataFrame,
            pair: Tuple[str, str], anchors: pd.DataFrame) -> Dict[str, Any]:
    """What the two sweeps say, classified rather than asserted."""
    common_curve = gap_curve(common, "fee", pair)
    diff_curve = gap_curve(differential, "differential", pair)
    be_common = break_even(common_curve, "fee")
    be_diff = break_even(diff_curve, "differential")
    base = float(diff_curve.loc[diff_curve["differential"].idxmin(),
                                "gap_pct"])
    reachable = anchors[anchors["gap_pct"] <= 0.0]
    out: Dict[str, Any] = {
        "baseline_gap_pct": base,
        "break_even_common": be_common,
        "break_even_common_bp": be_common * 1e4 if np.isfinite(be_common)
        else float("inf"),
        "break_even_differential": be_diff,
        "break_even_differential_bp": be_diff * 1e4 if np.isfinite(be_diff)
        else float("inf"),
        "common_is_near_neutral": bool(not np.isfinite(be_common)),
        "differential_closes_the_gap": bool(np.isfinite(be_diff)),
        "n_anchors_that_close_it": int(len(reachable)),
        "anchors_that_close_it": [str(v) for v in reachable["label"]],
        "cheapest_anchor_that_closes_it": (str(reachable["label"].iloc[0])
                                           if len(reachable) else ""),
    }
    # Is the break-even inside the range a real investor has faced? That is
    # the whole practical question, and it is a comparison against the
    # configured anchors rather than a judgement made here.
    if np.isfinite(be_diff) and len(anchors):
        widest = float(anchors["differential"].max())
        narrowest = float(anchors["differential"].min())
        out["inside_historic_range"] = bool(be_diff <= widest)
        out["below_cheapest_anchor"] = bool(be_diff <= narrowest)
    else:
        out["inside_historic_range"] = False
        out["below_cheapest_anchor"] = False
    return out
