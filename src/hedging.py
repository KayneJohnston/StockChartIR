"""Currency hedging the international equity sleeve.

The international leg in ``docs/01`` bundles two distinct exposures: the
foreign *asset* return and the foreign *currency*. An investor can separate
them -- hedged share classes exist -- so the question is whether they should,
and at what price.

Under covered interest parity a fully hedged foreign equity position earns the
foreign asset return plus the *domestic* short rate, giving up the foreign
one. This module builds that series (in :mod:`src.data_loader`), blends it
with the unhedged leg at a chosen ratio, and sweeps the ratio against an
annual hedging cost.

Two properties make the comparison clean.

**Identical history.** The hedge ratio deliberately does not affect which
country-years are usable, so the bootstrap draws exactly the same blocks at
every ratio. This module goes further and reuses one set of drawn
*(country, calendar)* indices across the whole sweep, re-gathering only the
international leg -- so two hedge ratios are compared on literally the same
simulated lives, not merely on the same distribution.

**The cost is swept, not assumed.** The honest answer to "should I hedge" is a
break-even: the annual cost above which hedging stops paying. That number is
what a reader can check against the hedged share class actually on offer.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from . import bootstrap as bs
from . import data_loader as dl
from . import lifecycle as lc
from . import utility as ut

LOGGER = logging.getLogger(__name__)


def panel_moments(panel: dl.Panel) -> Dict[str, float]:
    """Moments of the international leg, so the mechanism is visible."""
    mask = panel.available
    intl = panel.intl_eq[mask]
    dom = panel.dom_eq[mask]
    infl = panel.inflation[mask]
    log_gross = np.log1p(np.clip(intl, dl.GROSS_RETURN_FLOOR - 1.0, None))
    return {
        "intl_mean": float(intl.mean()),
        "intl_geometric_mean": float(np.expm1(log_gross.mean())),
        "intl_sd": float(intl.std(ddof=1)),
        "intl_p5": float(np.percentile(intl, 5)),
        "corr_intl_domestic_equity": float(np.corrcoef(intl, dom)[0, 1]),
        "corr_intl_inflation": float(np.corrcoef(intl, infl)[0, 1]),
    }


def _regather(paths: bs.BootstrapPaths, panel: dl.Panel) -> bs.BootstrapPaths:
    """Re-read the international leg for already-drawn blocks.

    The drawn ``(calendar, country)`` indices are held fixed and only
    ``intl_eq`` is looked up again, so every hedge ratio is evaluated on
    exactly the same simulated lives.
    """
    return dataclasses.replace(
        paths,
        intl_eq=np.ascontiguousarray(
            panel.intl_eq[paths.calendar_index, paths.domestic_country]),
    )


def sweep_hedging(
    cfg: Mapping[str, Any],
    spec: lc.LifecycleSpec,
    strategies: Mapping[str, lc.Strategy],
    ratios: Sequence[float],
    costs: Sequence[float],
    n_paths: int,
    mode: str | None = None,
    gammas: Sequence[float] | None = None,
    strategy_keys: Sequence[str] = ("balanced_all_equity",
                                    "international_equity"),
) -> pd.DataFrame:
    """Evaluate every ``(hedge ratio, annual cost)`` pair on identical lives."""
    mode = mode or cfg["bootstrap"]["panel"]
    gammas = list(gammas if gammas is not None else cfg["utility"]["risk_aversions"])
    active = {k: v for k, v in strategies.items() if k in strategy_keys}

    base_panel = dl.build_panel(cfg, mode, hedge_ratio=0.0, hedge_cost=0.0)
    sampler = bs.from_config(base_panel, cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths, chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    income = lc.simulate_income(
        spec, n_paths, shocks=lc.draw_income_shocks(n_paths, spec.n_working, rng))

    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        cost_grid = list(costs) if ratio > 0 else [0.0]
        for cost in cost_grid:
            panel = dl.build_panel(cfg, mode, hedge_ratio=float(ratio),
                                   hedge_cost=float(cost))
            if not np.array_equal(panel.available, base_panel.available):
                raise RuntimeError(
                    "hedging changed the availability mask; the paired "
                    "comparison would no longer hold")
            moments = panel_moments(panel)
            local = _regather(paths, panel)
            results = lc.simulate_all(local, active, spec, income)
            for key, outcome in results.items():
                bundle = ut.bundle_from_outcome(outcome, cfg, spec)
                row: Dict[str, Any] = {
                    "hedge_ratio": float(ratio),
                    "hedge_cost": float(cost),
                    "strategy": key,
                    "label": outcome.label,
                }
                for gamma in gammas:
                    row[f"cec_gamma{float(gamma):g}"] = \
                        ut.crra_certainty_equivalent(
                            bundle, float(gamma),
                            float(cfg["utility"]["discount_factor"]),
                            float(cfg["utility"]["bequest_weight"]),
                            bool(cfg["utility"]["bequest_enabled"]))
                row["prob_ruin"] = float(outcome.ruin.mean())
                row["median_bequest"] = float(np.median(outcome.bequest))
                row["p5_retirement_consumption"] = float(np.percentile(
                    outcome.consumption[:, spec.retirement_slice].mean(axis=1),
                    5))
                row.update(moments)
                rows.append(row)
        LOGGER.info("hedging sweep: ratio %.2f done", ratio)
    return pd.DataFrame.from_records(rows)


def break_even_costs(frame: pd.DataFrame, metric: str,
                     strategy: str = "balanced_all_equity") -> pd.DataFrame:
    """The annual cost at which hedging stops being worth it, per ratio.

    Interpolates each ratio's certainty equivalent against cost and finds
    where it crosses the *unhedged* level. ``inf`` means hedging beats not
    hedging across the entire cost grid; ``nan`` means it never wins even for
    free.
    """
    block = frame[frame["strategy"] == strategy]
    unhedged = block[block["hedge_ratio"] == 0.0]
    if unhedged.empty:
        return pd.DataFrame()
    baseline = float(unhedged[metric].iloc[0])

    rows = []
    for ratio, chunk in block[block["hedge_ratio"] > 0].groupby("hedge_ratio"):
        chunk = chunk.sort_values("hedge_cost")
        costs = chunk["hedge_cost"].to_numpy(dtype=float)
        values = chunk[metric].to_numpy(dtype=float)
        advantage = values - baseline
        if advantage.max() <= 0:
            break_even = float("nan")           # never worth it, even free
        elif advantage.min() > 0:
            break_even = float("inf")           # still worth it at the top
        else:
            # np.interp needs an increasing x-axis. Advantage falls as cost
            # rises, so sorting by advantage ascending puts cost descending,
            # which is what the interpolation needs. Sorting the other way
            # silently returns a grid endpoint instead of the crossing.
            order = np.argsort(advantage)
            break_even = float(np.interp(0.0, advantage[order], costs[order]))
        rows.append({
            "hedge_ratio": float(ratio),
            "unhedged_cec": baseline,
            "cec_at_zero_cost": float(values[0]),
            "gain_at_zero_cost_pct": (values[0] / baseline - 1.0) * 100.0,
            "break_even_annual_cost": break_even,
        })
    return pd.DataFrame.from_records(rows)


def optimal_ratio_by_cost(frame: pd.DataFrame, metric: str,
                          strategy: str = "balanced_all_equity"
                          ) -> pd.DataFrame:
    """The best hedge ratio at each assumed annual cost."""
    block = frame[frame["strategy"] == strategy]
    unhedged = block[block["hedge_ratio"] == 0.0]
    baseline = float(unhedged[metric].iloc[0]) if len(unhedged) else np.nan
    rows = []
    for cost in sorted(block.loc[block["hedge_ratio"] > 0,
                                 "hedge_cost"].unique()):
        candidates = pd.concat([
            block[(block["hedge_ratio"] > 0) & (block["hedge_cost"] == cost)],
            unhedged,
        ])
        best = candidates.loc[candidates[metric].idxmax()]
        rows.append({
            "hedge_cost": float(cost),
            "optimal_hedge_ratio": float(best["hedge_ratio"]),
            "cec_at_optimum": float(best[metric]),
            "gain_over_unhedged_pct": (float(best[metric]) / baseline - 1.0)
            * 100.0 if np.isfinite(baseline) else np.nan,
        })
    return pd.DataFrame.from_records(rows)
