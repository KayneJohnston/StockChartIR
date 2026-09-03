"""Dividend imputation, and whether the credit undoes the case for going abroad.

Section #withholding prices the tax that falls on the *international* leg and
finds it large: at the US statutory rate for non-residents it is worth 115
basis points a year, which is more than the fee differential Section #fees
identifies as the break-even. That section closes by naming what it does not
model, and the first item on the list is this one.

**Dividend imputation runs the other way.** A company pays corporate tax on
its profit and distributes what is left. Under a classical system that is the
end of it and the shareholder is taxed again on the dividend. Under an
imputation system the corporate tax is treated as tax the *shareholder*
already paid: the dividend arrives with a credit attached, the shareholder
declares the grossed-up amount, and the credit is set against their own bill.
Australia has run such a system since 1987 and has made the credit fully
refundable since 2000, so a shareholder whose tax rate is below the corporate
rate receives the difference in cash. A superannuation fund in pension phase
pays no tax at all, and therefore collects the entire credit as a cheque.

Crucially the credit is available to *residents* only. Which makes it the
exact mirror of withholding: one tax falls only on the foreign leg, one credit
falls only on the home leg, and both are levied on dividends rather than on
assets. Section #withholding measures one blade of the scissors. This section
measures the other, and then closes them.

**The arithmetic.** Take the panel's convention, ``1 + total = (1 + capital
gain)(1 + dividend)``, and write ``q = dp / (1 + dp)`` for the dividend's
share of the gross factor -- the same quantity Section #withholding builds.
Let ``t_c`` be the corporate rate the credit imputes, ``t_f`` the rate the
holder's own fund pays, and ``phi`` the fraction of the dividend that carries
a credit. A franked dollar of cash dividend grosses up to ``1/(1 - t_c)``,
is taxed at ``t_f``, and returns ``(1 - t_f)/(1 - t_c)``; an unfranked dollar
simply returns ``(1 - t_f)``. So a dollar of cash dividend is worth

    1 + c,   c = (1 - t_f) * (1 + phi * t_c / (1 - t_c)) - 1

and the after-credit gross factor is ``(1 + cg)(1 + (1 + c) dp)``, giving

    1 + r_net = (1 + r_gross) * (1 + c * q)

which is Section #withholding's expression with ``-tau`` replaced by ``+c``.
That symmetry is not a coincidence and it is the reason both sections can be
quoted in the same units: a statutory rate means nothing next to a fee until
it has been converted into one.

**What the formula says at the corners.** With no imputation and no fund tax
``c`` is zero and the panel is untouched, which is this paper's baseline. With
full franking at a 30% corporate rate and a pension-phase fund, ``c`` is
``0.30/0.70 = 42.9%`` -- the whole corporate tax handed back. With the same
franking inside a fund taxed at 15% it is 21.4%. And with *no* franking inside
that fund it is ``-15%``: a drag, correctly, because a fund that pays tax on
an unfranked dividend is worse off than the untaxed baseline this paper
otherwise assumes. The formula is not built to produce a credit; it produces
whichever sign the parameters imply.

**Against which baseline.** Adding a credit on top of a pre-tax return looks
at first like counting the same relief twice. It is not. Equity total returns
are already net of corporate tax -- the company paid it before the dividend
was declared -- so the baseline here is an investor who receives the cash
dividend and pays no *personal* tax on it. Imputation refunds the *corporate*
tax on top of that, and for a pension-phase fund the refund is cash in the
bank. What the baseline does omit is the personal tax that a classical system
would then levy, which runs against the home leg; so the comparison below is
conservative in exactly the direction that matters.

**Whose law this is.** The credit is applied to whichever market a simulated
investor holds as their own, which models a world where every country operates
imputation rather than the world that exists. That is the same convention
Section #pension uses when it pays Australia's Age Pension to an investor
drawing sixteen countries' returns, and it is deliberate: the question is what
the *mechanism* is worth, not what the population-weighted average of
sixteen tax codes comes to. Nine of this panel's sixteen countries operated
some form of imputation at some point in the twentieth century and all but
Australia abolished it, so neither a universal nor a single-country reading is
the whole truth. The universal one is reported because it is the one that
isolates the effect.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import withholding as wh

LOGGER = logging.getLogger(__name__)

#: Credit rates swept by default, as a fraction of the cash dividend. The grid
#: runs past anything statutory on purpose, for the same reason Section
#: #withholding's does: the distance to a break-even is the margin of safety.
DEFAULT_CREDITS: Tuple[float, ...] = (-0.15, 0.0, 0.05, 0.10, 0.15, 0.2143,
                                      0.25, 0.30, 0.35, 0.4286, 0.50, 0.60)

#: Structural parameters, not credit rates. Each anchor names a real holder
#: and the three numbers that describe their position; ``c`` is *derived* from
#: them by :func:`credit_rate` rather than asserted, so a reader can check the
#: arithmetic rather than take the label's word for it.
#:
#: ``company``
#:     The corporate rate the credit imputes. Australia's is 30%.
#: ``fund``
#:     The rate the holder's own vehicle pays on investment income. An
#:     Australian fund pays 15% while accumulating and nothing once a member
#:     has started a pension.
#: ``franked``
#:     The share of dividends arriving with a credit attached.
ANCHORS: Dict[str, Dict[str, float]] = {
    "no imputation, no fund tax": {
        "company": 0.30, "fund": 0.00, "franked": 0.00},
    "a taxed fund, nothing franked": {
        "company": 0.30, "fund": 0.15, "franked": 0.00},
    "an Australian fund still accumulating": {
        "company": 0.30, "fund": 0.15, "franked": 1.00},
    "an Australian fund paying a pension": {
        "company": 0.30, "fund": 0.00, "franked": 1.00},
}

#: The anchor whose fund rate the partial-franking table is drawn at. Naming
#: it here rather than repeating ``0.15`` in the pipeline, the document and the
#: paper means the three cannot drift apart, and a change to :data:`ANCHORS`
#: reaches all of them.
ACCUMULATING: str = "an Australian fund still accumulating"

#: The franked shares tabulated so partial franking is visible rather than
#: assumed away. A market's franking level is not in this project's data, so
#: it is swept rather than asserted.
DEFAULT_FRANKED_SHARES: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)


# ---------------------------------------------------------------------------
# The credit
# ---------------------------------------------------------------------------
def anchor_parameters(label: str,
                      anchors: Mapping[str, Mapping[str, float]] = ANCHORS,
                      ) -> Tuple[float, float, float]:
    """``(company, fund, franked)`` for a named anchor, or a clear failure."""
    if label not in anchors:
        raise KeyError(f"unknown anchor {label!r}; expected one of "
                       f"{sorted(anchors)}")
    params = anchors[label]
    return (float(params["company"]), float(params["fund"]),
            float(params["franked"]))


def credit_rate(company: float, fund: float, franked: float = 1.0) -> float:
    """What a dollar of cash dividend is worth, less the dollar itself.

    The whole tax code of this section in one line. Returns a *signed*
    quantity: positive when the credit outweighs the fund's own tax, negative
    when it does not, and zero when neither applies.
    """
    for name, value in (("company", company), ("fund", fund),
                        ("franked", franked)):
        if not 0.0 <= float(value) < 1.0 + (name == "franked") * 1e-12:
            raise ValueError(f"{name} rate must lie in [0, 1), got {value!r}")
    if float(company) >= 1.0:
        raise ValueError("a corporate rate of 100% cannot be imputed")
    gross_up = 1.0 + float(franked) * float(company) / (1.0 - float(company))
    return (1.0 - float(fund)) * gross_up - 1.0


def anchor_credits(anchors: Mapping[str, Mapping[str, float]] = ANCHORS,
                   ) -> pd.DataFrame:
    """Each labelled holder's structural parameters and the credit they imply."""
    rows: List[Dict[str, Any]] = []
    for label, params in anchors.items():
        rows.append({
            "label": label,
            "company_tax": float(params["company"]),
            "fund_tax": float(params["fund"]),
            "franked_share": float(params["franked"]),
            "credit": credit_rate(float(params["company"]),
                                  float(params["fund"]),
                                  float(params["franked"])),
        })
    frame = pd.DataFrame.from_records(rows)
    frame["credit_pct"] = frame["credit"] * 100.0
    return frame


def franking_grid(company: float, fund: float,
                  shares: Sequence[float] = DEFAULT_FRANKED_SHARES,
                  ) -> pd.DataFrame:
    """The credit as a function of how much of the dividend is franked."""
    return pd.DataFrame.from_records([
        {"franked_share": float(s),
         "credit": credit_rate(company, fund, float(s)),
         "credit_pct": credit_rate(company, fund, float(s)) * 100.0}
        for s in shares])


def break_even_franked_share(company: float, fund: float) -> float:
    """How much of a dividend must be franked before the credit pays for itself.

    A fund that pays tax on its investment income starts behind this paper's
    untaxed baseline, and franking has to make that back before it is worth
    anything at all. Setting ``credit_rate`` to zero and solving for ``phi``
    gives the level at which it does. Returns zero for an untaxed fund, which
    is already ahead, and infinity where no amount of franking can close the
    gap.
    """
    company, fund = float(company), float(fund)
    if fund <= 0.0:
        return 0.0
    if company <= 0.0:
        return float("inf")
    return (fund / (1.0 - fund)) * (1.0 - company) / company


def effective_credit(credit: float, domestic: np.ndarray) -> np.ndarray:
    """``(T, C)`` the credit expressed as an equivalent negative expense ratio.

    The mirror of :func:`src.withholding.effective_fee`, and the reason the two
    sections are commensurable: a credit on dividends is a fee on assets of
    ``-c * q``. Quoting it in basis points is the only way to compare it
    against the withholding drag, the fee break-even, or the cost of trading.
    """
    return float(credit) * np.asarray(domestic, dtype=float)


def realised_credit(credit: float, domestic: np.ndarray, years: np.ndarray,
                    eras: Sequence[Tuple[int, int]] = ((1890, 1949),
                                                       (1950, 1989),
                                                       (1990, 2020)),
                    ) -> pd.DataFrame:
    """What one credit rate is actually worth, by era, in basis points a year.

    Same point as Section #withholding's drag table and the same cause: the
    credit is levied on dividends, and dividend yields have fallen across this
    panel, so an unchanged tax code delivers less than it used to.
    """
    value = effective_credit(credit, domestic)
    years = np.asarray(years)
    rows: List[Dict[str, Any]] = [{
        "era": "whole panel",
        "first_year": int(years.min()), "last_year": int(years.max()),
        "mean_dividend_share": float(np.nanmean(domestic)),
        "credit_bp": float(np.nanmean(value)) * 1e4,
    }]
    for lo, hi in eras:
        mask = (years >= lo) & (years <= hi)
        if not mask.any():
            continue
        rows.append({
            "era": f"{lo}-{hi}",
            "first_year": int(lo), "last_year": int(hi),
            "mean_dividend_share": float(np.nanmean(domestic[mask])),
            "credit_bp": float(np.nanmean(value[mask])) * 1e4,
        })
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------
def apply_franking(panel: dl.Panel, credit: float,
                   domestic: np.ndarray) -> dl.Panel:
    """A panel whose home leg carries the credit and whose foreign leg does not.

    ``available`` is untouched, so every credit rate admits the same blocks and
    the bootstrap draws the same calendar history at each: the comparison is
    paired rather than merely parallel, exactly as in Sections #fees and
    #withholding.

    Country-years with no dividend observation are credited nothing rather than
    dropped, for the same reason the withholding study charges them nothing.
    Here that understates the credit, which runs *against* the home leg and so
    is again the conservative direction.
    """
    value = np.nan_to_num(effective_credit(credit, domestic), nan=0.0)
    if not value.any():
        return panel
    grossed = (1.0 + np.asarray(panel.dom_eq, dtype=float)) * (1.0 + value) - 1.0
    return dataclasses.replace(
        panel, name=f"{panel.name}[frank{float(credit) * 100:.1f}%]",
        dom_eq=grossed)


def apply_wedge(panel: dl.Panel, credit: float, domestic: np.ndarray,
                rate: float, sleeve: np.ndarray) -> dl.Panel:
    """Both blades of the scissors: credit at home, withholding abroad.

    Neither section on its own describes anybody. An Australian investor in
    superannuation collects franking credits on their domestic holdings *and*
    pays unrecoverable withholding on their foreign ones, simultaneously, and
    the two push the same way. This is the only function here that models a
    real position rather than an isolated mechanism.
    """
    return wh.apply_withholding(
        apply_franking(panel, credit, domestic), rate, sleeve)


def sweep(panel: dl.Panel, summarise: Callable[..., pd.DataFrame],
          n_paths: int, domestic: np.ndarray,
          grid: Sequence[float] = DEFAULT_CREDITS,
          rate: float = 0.0,
          sleeve: np.ndarray | None = None) -> pd.DataFrame:
    """Re-run the comparison at each credit rate, tagged with the credit.

    ``rate`` and ``sleeve`` hold the foreign leg's withholding fixed while the
    credit moves, so the sweep can be run either in isolation (``rate = 0``) or
    on top of the tax Section #withholding has already established.
    """
    frames: List[pd.DataFrame] = []
    for credit in grid:
        LOGGER.info("imputation credit: %+.2f%% of the home market's "
                    "dividends, withholding %.1f%% abroad",
                    float(credit) * 100.0, float(rate) * 100.0)
        panel_c = (apply_wedge(panel, float(credit), domestic, float(rate),
                               sleeve)
                   if rate and sleeve is not None else
                   apply_franking(panel, float(credit), domestic))
        block = summarise(panel_c, n_paths)
        block.insert(0, "credit", float(credit))
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def wedge_sweep(panel: dl.Panel, summarise: Callable[..., pd.DataFrame],
                n_paths: int, domestic: np.ndarray, sleeve: np.ndarray,
                positions: Sequence[Tuple[str, float, float]],
                ) -> pd.DataFrame:
    """One row-block per named position, each a ``(label, credit, rate)`` pair.

    Unlike :func:`sweep` this does not trace a curve. It scores the handful of
    positions an actual investor can occupy, which is what makes it the table
    a reader can locate themselves in.
    """
    frames: List[pd.DataFrame] = []
    for label, credit, rate in positions:
        LOGGER.info("wedge: %s (credit %+.2f%%, withholding %.1f%%)",
                    label, float(credit) * 100.0, float(rate) * 100.0)
        block = summarise(
            apply_wedge(panel, float(credit), domestic, float(rate), sleeve),
            n_paths)
        block.insert(0, "position", str(label))
        block.insert(1, "credit", float(credit))
        block.insert(2, "rate", float(rate))
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def wedge_positions(anchors: Mapping[str, Mapping[str, float]] = ANCHORS,
                    rate: float = 0.15,
                    ) -> List[Tuple[str, float, float]]:
    """The named positions scored by :func:`wedge_sweep`, derived not asserted.

    The first is the paper's own baseline -- no credit, no withholding -- and
    the second isolates withholding, so the table reads as a decomposition
    rather than as a list.
    """
    out: List[Tuple[str, float, float]] = [
        ("neither tax", 0.0, 0.0),
        ("withholding only", 0.0, float(rate)),
    ]
    for label, params in anchors.items():
        credit = credit_rate(float(params["company"]), float(params["fund"]),
                             float(params["franked"]))
        if credit == 0.0:
            continue
        out.append((label, credit, float(rate)))
    return out


# ---------------------------------------------------------------------------
# Reading the sweep
# ---------------------------------------------------------------------------
def gap_curve(frame: pd.DataFrame, challenger: str,
              rivals: Sequence[str]) -> pd.DataFrame:
    """The challenger's lead over each rival at every credit rate."""
    return wh.gap_curve(frame.rename(columns={"credit": "rate"}), challenger,
                        rivals).rename(columns={"rate": "credit",
                                                "rate_pct": "credit_pct"})


def crossing(curve: pd.DataFrame, rival: str) -> float:
    """The credit at which ``rival`` first overtakes the challenger."""
    return wh.crossing(curve.rename(columns={"credit": "rate"}), rival)


def crossings(curve: pd.DataFrame, rivals: Sequence[str],
              domestic: np.ndarray) -> pd.DataFrame:
    """Where each rival overtakes the challenger, in credit and basis points."""
    out = wh.crossings(curve.rename(columns={"credit": "rate"}), rivals,
                       domestic)
    return out.rename(columns={"crossing_rate": "crossing_credit",
                               "equivalent_drag_bp": "equivalent_credit_bp"})


def anchor_table(curve: pd.DataFrame, rivals: Sequence[str],
                 domestic: np.ndarray,
                 anchors: Mapping[str, Mapping[str, float]] = ANCHORS,
                 ) -> pd.DataFrame:
    """Every labelled holder placed on the swept curve by interpolation."""
    credits = anchor_credits(anchors)
    placed = wh.anchor_table(
        curve.rename(columns={"credit": "rate"}), rivals, domestic,
        {str(r["label"]): float(r["credit"]) for _, r in credits.iterrows()})
    placed = placed.rename(columns={"rate": "credit", "rate_pct": "credit_pct",
                                    "drag_bp": "credit_bp"})
    return placed.merge(
        credits[["label", "company_tax", "fund_tax", "franked_share"]],
        on="label", how="left")


def optimum_by_credit(frame: pd.DataFrame, parameter: Mapping[str, float],
                      column: str, name: str = "domestic_share",
                      ) -> pd.DataFrame:
    """The certainty-equivalent-maximising portfolio at each credit rate.

    The more useful of this section's two questions, for the same reason it is
    in Section #withholding: "where does the 50/50 split overtake" presumes the
    investor is choosing between two fixed portfolios, and "what should they
    hold" does not.
    """
    from . import inflation as ifl

    out = ifl.optimum_by_bucket(frame.rename(columns={"credit": "rate"}),
                                parameter, column, name, group="rate")
    return out.rename(columns={"rate": "credit"})


def wedge_optimum(frame: pd.DataFrame, parameter: Mapping[str, float],
                  column: str, name: str = "domestic_share") -> pd.DataFrame:
    """The optimal portfolio at each named position rather than at each rate.

    The grouping column is renamed rather than reused because a wedge frame
    already carries ``rate`` -- the withholding the foreign leg pays at that
    position -- and grouping on a duplicated name silently fails.
    """
    from . import inflation as ifl

    labels_first = frame.drop_duplicates("position")
    keyed = frame.drop(columns=["rate"], errors="ignore").rename(
        columns={"position": "rate"})
    out = ifl.optimum_by_bucket(keyed, parameter, column, name, group="rate")
    out = out.rename(columns={"rate": "position"})
    labels = labels_first.set_index("position")
    for source in ("credit", "rate"):
        if source in labels.columns:
            out[source] = [float(labels.loc[p, source]) for p in out["position"]]
    return out


def wedge_comparison(frame: pd.DataFrame, column: str,
                     strategies: Sequence[str]) -> pd.DataFrame:
    """One row per position: every fixed strategy's certainty equivalent."""
    rows: List[Dict[str, Any]] = []
    for position, block in frame.groupby("position", sort=False):
        indexed = block.set_index("strategy")
        row: Dict[str, Any] = {
            "position": str(position),
            "credit": float(block["credit"].iloc[0]),
            "rate": float(block["rate"].iloc[0]),
        }
        present = [s for s in strategies if s in indexed.index]
        for key in present:
            row[f"cec_{key}"] = float(indexed.loc[key, column])
        if present:
            values = {k: row[f"cec_{k}"] for k in present}
            best = max(values, key=values.get)
            row["winner"] = best
            row["best_cec"] = values[best]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def verdict(curve: pd.DataFrame, crossed: pd.DataFrame,
            optima: pd.DataFrame, credits: pd.DataFrame,
            comparison: pd.DataFrame, challenger: str) -> Dict[str, Any]:
    """What the credit does to the headline, classified from the sweep."""
    if not len(curve):
        return {"levels": 0}
    ordered = curve.sort_values("credit")
    reached = crossed[crossed["reached_on_grid"]] if len(crossed) else crossed
    by_label = credits.set_index("label")["credit"].astype(float)
    pension = float(by_label.get("an Australian fund paying a pension",
                                 float("nan")))
    accumulating = float(by_label.get("an Australian fund still accumulating",
                                      float("nan")))
    found: Dict[str, Any] = {
        "levels": int(len(ordered)),
        "highest_credit_pct": float(ordered["credit_pct"].iloc[-1]),
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
        "pension_credit": pension,
        "accumulation_credit": accumulating,
        # The two comparisons a reader will make against a real tax code.
        "crossing_within_pension_phase": bool(
            len(reached) and np.isfinite(pension)
            and float(reached["crossing_pct"].min()) <= pension * 100.0),
        "crossing_within_accumulation": bool(
            len(reached) and np.isfinite(accumulating)
            and float(reached["crossing_pct"].min()) <= accumulating * 100.0),
    }
    if len(optima) and "optimal_domestic_share" in optima.columns:
        by_credit = optima.sort_values("credit")
        low = float(by_credit["optimal_domestic_share"].iloc[0])
        high = float(by_credit["optimal_domestic_share"].iloc[-1])
        at_zero = by_credit[np.isclose(by_credit["credit"], 0.0)]
        found.update({
            "optimal_domestic_at_bottom": low,
            # The grid starts below zero, where a taxed fund holding unfranked
            # stock is *losing*. The neutral row is the one a reader compares
            # against, so it is carried separately rather than inferred.
            "optimal_domestic_at_zero": (
                float(at_zero["optimal_domestic_share"].iloc[0])
                if len(at_zero) else float("nan")),
            "optimal_domestic_at_top": high,
            "optimal_domestic_shift": high - low,
            "lowest_credit_pct": float(by_credit["credit"].iloc[0]) * 100.0,
            "optimum_moves_home": bool(high > low),
            "optimum_ever_moves": bool(
                by_credit["optimal_domestic_share"].nunique() > 1),
            "smallest_optimum_margin_pct": (
                float(by_credit["margin_over_runner_up_pct"].min())
                if "margin_over_runner_up_pct" in by_credit
                else float("nan")),
        })
    if len(comparison):
        winners = [str(w) for w in comparison.get("winner", [])]
        found.update({
            "positions": int(len(comparison)),
            "wedge_winners": sorted(set(winners)),
            "wedge_winner_changes": bool(len(set(winners)) > 1),
            "wedge_winner_at_baseline": winners[0] if winners else "",
            "wedge_winner_at_the_end": winners[-1] if winners else "",
            # The question the section exists to answer: does closing both
            # blades of the scissors overturn what one blade alone could not?
            "wedge_overturns_the_headline": bool(
                len(winners) > 1 and winners[-1] != winners[0]),
        })
    return found
