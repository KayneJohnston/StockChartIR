"""Foreign dividend withholding tax, and the rate at which it undoes the result.

Section #fees asks how large a cost differential between a domestic and an
international fund would have to be to cancel this paper's headline, and
answers with a number it describes as beyond any index-fund pair. That framing
has a hole in it. There is a cost differential between holding your own market
and holding everyone else's that is not a fund's fee, is not negotiable, and
is not a choice: **foreign dividend withholding tax**.

A government taxes dividends leaving its borders. A resident holding their own
market either does not pay it or receives a full credit for it; a foreigner
holding that same market pays it and, in a retail structure, usually cannot
reclaim it. The statutory US rate on dividends to non-resident individuals is
30%, most treaties cut it to 15% for portfolio investors, and a broad
developed-markets index fund bears a weighted-average of roughly 7.5% across
its constituents. None of those is a fee an investor can shop around for.

So this is the same experiment as Section #fees run with a real instrument
instead of a hypothetical one -- and with two differences that matter.

**The base is dividends, not assets.** A fee is levied on the whole portfolio;
withholding is levied only on the part of the return that arrives as a
dividend. That makes the drag proportional to the dividend yield, which in
this panel has fallen by a third since the war. The same statutory rate cost
an investor materially more in 1930 than it does now, and a flat assumption
would miss that entirely.

**It falls on exactly one leg.** All-international pays it on every dividend
it receives; the 50/50 split pays it on half; a domestic-only portfolio pays
none. That is the same asymmetry Section #fees constructs by hypothesis,
except that here it is the law.

**The arithmetic.** The panel's source data follows the convention
``1 + total = (1 + capital gain)(1 + dividend)``, with the dividend measured
against the ending price. Withholding at rate ``tau`` leaves the investor
``(1 - tau)`` of that dividend, so the after-tax gross factor is
``(1 + cg)(1 + (1 - tau) dp)`` and

    1 + r_net = (1 + r_gross) * (1 - tau * q),   q = dp / (1 + dp)

exactly -- a multiplicative haircut of precisely the form Section #fees
already uses for an expense ratio, with a time-varying rate ``tau * q_t``
instead of a constant one. ``q`` is the share of a year's ending value that
arrived as a taxable dividend, and it is the only new quantity this section
needs.

**What this does not model.** Dividend imputation, which in Australia and New
Zealand refunds corporate tax to *domestic* shareholders and would widen the
gap further in the home market's favour; the second layer of withholding a
country's investor suffers by holding a US-domiciled fund of foreign stocks
rather than the stocks themselves; reclaim procedures, which recover part of
the tax at a cost in paperwork most retail investors do not pay; and the tax
an investor's own government then levies on the dividend it has already been
withheld from. Every one of those omissions runs the same way -- against the
international sleeve -- so the rates below are a floor on the real burden,
not an estimate of it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import fees as fee

LOGGER = logging.getLogger(__name__)

#: Rates swept by default. The grid runs past anything statutory on purpose:
#: the break-even is worth locating even when it sits outside the legal range,
#: because the distance to it is the margin of safety -- the same reasoning as
#: the fee grid in Section #fees.
DEFAULT_RATES: Tuple[float, ...] = (0.0, 0.05, 0.075, 0.10, 0.15, 0.20,
                                    0.25, 0.30, 0.35, 0.40, 0.50)

#: Real rates, as labelled anchors rather than findings. They put the swept
#: grid on a scale a reader recognises. Sources are cited in `docs/28`; none
#: of them can be verified from this project's own data.
ANCHORS: Dict[str, float] = {
    "a recovering pension fund": 0.0,
    "a broad developed-markets index fund": 0.075,
    "the typical treaty rate on portfolio dividends": 0.15,
    "the US statutory rate for non-residents": 0.30,
}


def dividend_share(jst: pd.DataFrame, isos: Sequence[str],
                   years: np.ndarray) -> np.ndarray:
    """``(T, C)`` share of a year's ending value that arrived as a dividend.

    The source data compounds rather than adds: ``1 + eq_tr`` equals
    ``(1 + eq_capgain)(1 + eq_dp)`` to eight decimal places, with the dividend
    measured against the ending price. The dividend's share of the gross
    factor is therefore ``dp / (1 + dp)``, which needs no reference to the
    total return at all and so cannot be thrown off by the panel's
    hyperinflations.

    Values outside ``[0, 1]`` are dropped rather than clipped into
    plausibility: a negative dividend is a data error, and a country-year the
    tax cannot be computed for should be visible as missing.
    """
    out = np.full((np.asarray(years).size, len(isos)), np.nan)
    index = {int(y): i for i, y in enumerate(np.asarray(years))}
    for j, iso in enumerate(isos):
        block = jst[jst["iso"] == iso]
        if block.empty or "eq_dp" not in block.columns:
            continue
        for year, dp in zip(block["year"], block["eq_dp"]):
            i = index.get(int(year))
            if i is None or not np.isfinite(dp):
                continue
            share = float(dp) / (1.0 + float(dp))
            if 0.0 <= share <= 1.0:
                out[i, j] = share
    return out


def sleeve_dividend_share(domestic: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of every *other* market's dividend share.

    Mirrors the construction of the international leg itself: the sleeve holds
    equal money in each of the other markets, so the dividend share of the
    sleeve is the plain mean of its constituents' -- the same reasoning, and
    the same weighting, as :func:`src.valuation.international_yield`.
    """
    values = np.asarray(domestic, dtype=float)
    out = np.full_like(values, np.nan)
    for j in range(values.shape[1]):
        others = np.delete(values, j, axis=1)
        present = np.isfinite(others).sum(axis=1)
        totals = np.nansum(others, axis=1)
        out[:, j] = np.where(present > 0, totals / np.maximum(present, 1),
                             np.nan)
    return out


def effective_fee(rate: float, sleeve: np.ndarray) -> np.ndarray:
    """``(T, C)`` the withholding rate expressed as an equivalent expense ratio.

    This is the whole translation: a tax on dividends is a fee on assets of
    ``tau * q``, where ``q`` is the dividend's share of the gross return. It
    lets the charge reuse the multiplicative form Section #fees established,
    and it is the number to quote when comparing the two -- a statutory rate
    means nothing next to an expense ratio until it has been put in the same
    units.
    """
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("a withholding rate must lie in [0, 1]")
    return float(rate) * np.asarray(sleeve, dtype=float)


def apply_withholding(panel: dl.Panel, rate: float,
                      sleeve: np.ndarray) -> dl.Panel:
    """A panel whose international leg has been taxed and whose home leg has not.

    ``available`` is untouched, so every rate admits the same blocks and the
    bootstrap draws the same calendar history at each -- the comparison is
    paired rather than merely parallel, exactly as in Section #fees.

    Country-years with no dividend observation are charged nothing rather than
    dropped. Dropping them would change which blocks the sampler can draw and
    so confound the tax with a change of sample; charging nothing understates
    the tax on those cells, which is the conservative direction.
    """
    charge = np.nan_to_num(effective_fee(rate, sleeve), nan=0.0)
    if not charge.any():
        return panel
    netted = (1.0 + np.asarray(panel.intl_eq, dtype=float)) * (1.0 - charge) - 1.0
    return dataclasses.replace(
        panel, name=f"{panel.name}[wht{float(rate) * 100:.1f}%]",
        intl_eq=netted)


def realised_drag(rate: float, sleeve: np.ndarray, years: np.ndarray,
                  eras: Sequence[Tuple[int, int]] = ((1890, 1949),
                                                     (1950, 1989),
                                                     (1990, 2020)),
                  ) -> pd.DataFrame:
    """What one statutory rate actually costs, by era, in basis points a year.

    The point of the table. A statutory rate is a constant; the drag it
    produces is not, because it is levied on dividends and dividend yields
    have fallen by roughly a third across this panel. An investor facing the
    same law in 1930 and 2010 paid materially different amounts for it.
    """
    charge = effective_fee(rate, sleeve)
    years = np.asarray(years)
    rows: List[Dict[str, Any]] = [{
        "era": "whole panel",
        "first_year": int(years.min()), "last_year": int(years.max()),
        "mean_dividend_share": float(np.nanmean(sleeve)),
        "drag_bp": float(np.nanmean(charge)) * 1e4,
    }]
    for lo, hi in eras:
        mask = (years >= lo) & (years <= hi)
        if not mask.any():
            continue
        rows.append({
            "era": f"{lo}-{hi}",
            "first_year": int(lo), "last_year": int(hi),
            "mean_dividend_share": float(np.nanmean(sleeve[mask])),
            "drag_bp": float(np.nanmean(charge[mask])) * 1e4,
        })
    return pd.DataFrame.from_records(rows)


def sweep(panel: dl.Panel, summarise: Callable[..., pd.DataFrame],
          n_paths: int, sleeve: np.ndarray,
          grid: Sequence[float] = DEFAULT_RATES) -> pd.DataFrame:
    """Re-run the comparison at each withholding rate, tagged with the rate."""
    frames: List[pd.DataFrame] = []
    for rate in grid:
        LOGGER.info("withholding: %.1f%% on the international sleeve's "
                    "dividends", float(rate) * 100.0)
        taxed = apply_withholding(panel, float(rate), sleeve)
        block = summarise(taxed, n_paths)
        block.insert(0, "rate", float(rate))
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def gap_curve(frame: pd.DataFrame, challenger: str,
              rivals: Sequence[str]) -> pd.DataFrame:
    """The challenger's lead over each rival at every rate, one row per rate."""
    cec = _cec_column(frame)
    rows: List[Dict[str, Any]] = []
    for rate in sorted(frame["rate"].unique()):
        block = frame[np.isclose(frame["rate"], rate)]
        values = {r["strategy"]: float(r[cec]) for _, r in block.iterrows()}
        ordered = sorted(values, key=values.get, reverse=True)
        row: Dict[str, Any] = {
            "rate": float(rate),
            "rate_pct": float(rate) * 100.0,
            f"cec_{challenger}": values.get(challenger, float("nan")),
            "winner": ordered[0] if ordered else "",
        }
        for rival in rivals:
            row[f"cec_{rival}"] = values.get(rival, float("nan"))
            row[f"lead_over_{rival}_pct"] = (
                (values[challenger] / values[rival] - 1.0) * 100.0
                if challenger in values and rival in values and values[rival]
                else float("nan"))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def crossing(curve: pd.DataFrame, rival: str) -> float:
    """The rate at which ``rival`` first overtakes the challenger.

    Linear interpolation between the straddling grid points. Infinity where
    the challenger's lead survives the whole grid, zero where it was never
    there -- both are answers, and both beat a missing value.
    """
    column = f"lead_over_{rival}_pct"
    if column not in curve.columns:
        return float("nan")
    ordered = curve.sort_values("rate").reset_index(drop=True)
    leads = ordered[column].to_numpy(dtype=float)
    rates = ordered["rate"].to_numpy(dtype=float)
    if not np.isfinite(leads).any():
        return float("nan")
    if leads[0] <= 0.0:
        return 0.0
    below = np.flatnonzero(leads <= 0.0)
    if below.size == 0:
        return float("inf")
    i = int(below[0])
    lo_lead, hi_lead = leads[i - 1], leads[i]
    lo, hi = rates[i - 1], rates[i]
    if lo_lead == hi_lead:
        return float(hi)
    return float(lo + (hi - lo) * lo_lead / (lo_lead - hi_lead))


def crossings(curve: pd.DataFrame, rivals: Sequence[str],
              sleeve: np.ndarray) -> pd.DataFrame:
    """Where each rival overtakes all-international, in rate and in basis points.

    The second column is the one that makes the number comparable with
    anything else in this paper: a statutory rate converted, at the panel's
    own average dividend share, into the annual drag it represents.
    """
    mean_share = float(np.nanmean(sleeve))
    rows: List[Dict[str, Any]] = []
    for rival in rivals:
        rate = crossing(curve, rival)
        rows.append({
            "rival": rival,
            "crossing_rate": rate,
            "crossing_pct": rate * 100.0 if np.isfinite(rate) else float("inf"),
            "equivalent_drag_bp": (rate * mean_share * 1e4
                                   if np.isfinite(rate) else float("inf")),
            "reached_on_grid": bool(np.isfinite(rate)),
            "lead_at_zero_pct": float(
                curve.sort_values("rate")[f"lead_over_{rival}_pct"].iloc[0]),
        })
    return pd.DataFrame.from_records(rows)


def anchor_table(curve: pd.DataFrame, rivals: Sequence[str],
                 sleeve: np.ndarray,
                 anchors: Mapping[str, float] = ANCHORS) -> pd.DataFrame:
    """Every real rate placed on the swept curve by interpolation."""
    mean_share = float(np.nanmean(sleeve))
    ordered = curve.sort_values("rate")
    rows: List[Dict[str, Any]] = []
    for label, rate in anchors.items():
        row: Dict[str, Any] = {
            "label": label, "rate": float(rate),
            "rate_pct": float(rate) * 100.0,
            "drag_bp": float(rate) * mean_share * 1e4,
        }
        for rival in rivals:
            column = f"lead_over_{rival}_pct"
            if column in ordered.columns:
                row[column] = float(np.interp(
                    float(rate), ordered["rate"].to_numpy(dtype=float),
                    ordered[column].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("rate")


def verdict(curve: pd.DataFrame, crossed: pd.DataFrame,
            optima: pd.DataFrame, drag: pd.DataFrame,
            challenger: str, anchors: Mapping[str, float] = ANCHORS,
            ) -> Dict[str, Any]:
    """What withholding tax does to the headline, classified from the sweep."""
    if not len(curve):
        return {"levels": 0}
    ordered = curve.sort_values("rate")
    reached = crossed[crossed["reached_on_grid"]]
    statutory = float(anchors.get("the US statutory rate for non-residents",
                                  0.30))
    treaty = float(anchors.get("the typical treaty rate on portfolio dividends",
                               0.15))
    found: Dict[str, Any] = {
        "levels": int(len(ordered)),
        "highest_rate_pct": float(ordered["rate_pct"].iloc[-1]),
        "winner_at_zero": str(ordered["winner"].iloc[0]),
        "winner_at_top": str(ordered["winner"].iloc[-1]),
        "winner_ever_changes": bool(ordered["winner"].nunique() > 1),
        "winners_seen": sorted(set(str(w) for w in ordered["winner"])),
        "any_rival_overtakes": bool(len(reached)),
        "n_rivals_overtaking": int(len(reached)),
        "first_crossing_pct": (float(reached["crossing_pct"].min())
                               if len(reached) else float("inf")),
        "first_rival": (str(reached.loc[reached["crossing_pct"].idxmin(),
                                        "rival"]) if len(reached) else ""),
        # The two comparisons a reader will make against the real world.
        "crossing_within_statutory": bool(
            len(reached) and float(reached["crossing_pct"].min())
            <= statutory * 100.0),
        "crossing_within_treaty": bool(
            len(reached) and float(reached["crossing_pct"].min())
            <= treaty * 100.0),
    }
    if len(drag):
        whole = drag[drag["era"] == "whole panel"]
        found["drag_bp_at_this_rate"] = (float(whole["drag_bp"].iloc[0])
                                         if len(whole) else float("nan"))
        eras = drag[drag["era"] != "whole panel"]
        if len(eras):
            found["drag_bp_earliest_era"] = float(eras["drag_bp"].iloc[0])
            found["drag_bp_latest_era"] = float(eras["drag_bp"].iloc[-1])
            found["drag_falls_over_time"] = bool(
                float(eras["drag_bp"].iloc[-1]) < float(eras["drag_bp"].iloc[0]))
    if len(optima) and "optimal_domestic_share" in optima.columns:
        by_rate = optima.sort_values("rate")
        low = float(by_rate["optimal_domestic_share"].iloc[0])
        high = float(by_rate["optimal_domestic_share"].iloc[-1])
        found.update({
            "optimal_domestic_at_zero": low,
            "optimal_domestic_at_top": high,
            "optimal_domestic_shift": high - low,
            "optimum_moves_home": bool(high > low),
            "optimum_ever_moves": bool(
                by_rate["optimal_domestic_share"].nunique() > 1),
            "smallest_optimum_margin_pct": float(
                by_rate["margin_over_runner_up_pct"].min())
            if "margin_over_runner_up_pct" in by_rate else float("nan"),
        })
    return found


def optimum_by_rate(frame: pd.DataFrame, parameter: Mapping[str, float],
                    column: str, name: str = "domestic_share") -> pd.DataFrame:
    """The certainty-equivalent-maximising portfolio at each withholding rate.

    The second question this section exists to answer, and the more useful of
    the two. "At what rate does the 50/50 split overtake all-international"
    presumes the investor is choosing between two fixed portfolios. "What
    should they hold at this rate" does not, and the answer is a schedule an
    investor can actually read off against the rate they face.

    The arithmetic is Section #inflation's, applied to a different sweep --
    including the margin columns, which are what stop a flat maximum being
    reported as an identification.
    """
    from . import inflation as ifl
    return ifl.optimum_by_bucket(frame, parameter, column, name, group="rate")
