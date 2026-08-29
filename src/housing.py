"""Housing as a fifth investable asset, priced by its holding cost.

The Jordà-Schularick-Taylor "Rate of Return on Everything" project measures
four asset classes. Three of them -- equity, bonds, bills -- are the investable
set everywhere else in this project. The fourth, housing, is measured just as
carefully and is left out, for two reasons that this module confronts rather
than repeats.

**Appraisal smoothing.** A house price index is built from transactions and
valuations, not from a continuous auction. Both lag: this year's index still
carries part of last year's level, so the published series understates the
volatility a holder actually bears. Comparing a smoothed series with a traded
one and concluding housing is low-risk is the classic error, so
:func:`desmoothed_panel` inverts the smoothing (Geltner) before anything here
uses the series, and reports the autocorrelation it removed.

**Holding cost.** A share certificate costs nothing to hold. A house costs
rates, insurance, maintenance and management, and it is illiquid and
undiversified. The source builds its housing total return from capital gains
plus a rental yield it describes as net of running costs, so the cost swept
here is best read as *additional* to whatever that construction already
deducts -- transaction and vacancy drag, the taxes a particular jurisdiction
adds, or a haircut for the risks a national index does not show.

That reading is the conservative one for the break-even the sweep reports. If
the published series is in fact grosser than the source describes, the true
all-in cost of holding housing is closer to the swept figure than to the
increment above it, and the break-even reported here is an overestimate of how
much *extra* cost housing can bear. The direction of that error is known even
though its size is not, which is why the whole curve is reported rather than a
single recommended weight.

**What the sweep does not model.** Housing here is a liquid, continuously
rebalanced, nationally diversified claim on the housing stock, because that is
what the index measures. A single leveraged owner-occupied house is a
different asset -- concentrated, lumpy, and bought with a mortgage -- and
nothing below speaks to it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import observed as obs

LOGGER = logging.getLogger(__name__)

#: The asset's name wherever it appears alongside :data:`src.lifecycle.ASSETS`.
HOUSING = "housing"

#: Investable set for this study: the usual four, plus housing last.
ASSETS: Tuple[str, ...] = ("dom_eq", "intl_eq", "bond", "bill", HOUSING)

#: Shortest run of housing returns worth de-smoothing. Below this the lag-one
#: autocorrelation is too noisy to invert with, and inverting with a noisy
#: coefficient injects volatility rather than restoring it.
MIN_HOUSING_YEARS = 30


def gap_respecting_autocorrelation(series: np.ndarray) -> float:
    """Lag-one autocorrelation over *consecutive* years only.

    :func:`src.observed.first_order_autocorrelation` drops missing values and
    correlates what remains, which splices the year before a gap onto the year
    after it. That is harmless for a series with no gaps and wrong for one with
    them, and the smoothing coefficient is the number the de-smoothing divides
    by, so it is worth getting right here.
    """
    values = np.asarray(series, dtype=float)
    pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    if int(pairs.sum()) < 10:
        return float("nan")
    return float(np.corrcoef(values[:-1][pairs], values[1:][pairs])[0, 1])


def desmoothed_panel(jst: pd.DataFrame, panel: dl.Panel
                     ) -> Tuple[np.ndarray, pd.DataFrame]:
    """``(T, C)`` de-smoothed real housing returns, plus a per-country audit.

    Each country is de-smoothed with its *own* estimated coefficient, because
    index construction differs by country and a pooled coefficient would
    over-correct the cleanly measured series and under-correct the rest. A
    country whose returns are not positively autocorrelated is left alone --
    there is no smoothing to undo -- and one with too short a history is
    dropped rather than de-smoothed with a coefficient that is mostly noise.
    """
    years = np.asarray(panel.years)
    raw = obs.housing_returns(jst, panel.countries, years)
    out = np.full_like(raw, np.nan)
    rows: List[Dict[str, Any]] = []
    for j, iso in enumerate(panel.countries):
        column = raw[:, j]
        finite = np.isfinite(column)
        rho = gap_respecting_autocorrelation(column)
        if int(finite.sum()) < MIN_HOUSING_YEARS:
            LOGGER.warning(
                "%s has %d housing years (<%d); dropped rather than "
                "de-smoothed with a noisy coefficient",
                iso, int(finite.sum()), MIN_HOUSING_YEARS)
            continue
        smoothed = obs.desmooth(column, rho)
        out[:, j] = smoothed
        rows.append({
            "iso": iso,
            "country": dl.ISO_TO_NAME.get(iso, iso),
            "years_raw": int(finite.sum()),
            "years_desmoothed": int(np.isfinite(smoothed).sum()),
            "autocorrelation": rho,
            "mean_raw": float(np.nanmean(column)),
            "mean_desmoothed": float(np.nanmean(smoothed)),
            "sd_raw": float(np.nanstd(column, ddof=1)),
            "sd_desmoothed": float(np.nanstd(smoothed, ddof=1)),
            "sd_ratio": float(np.nanstd(smoothed, ddof=1)
                              / np.nanstd(column, ddof=1)),
            "equity_mean": float(np.nanmean(panel.dom_eq[:, j])),
            "equity_sd": float(np.nanstd(panel.dom_eq[:, j], ddof=1)),
        })
    return out, pd.DataFrame.from_records(rows)


def restrict_to_housing(panel: dl.Panel, housing: np.ndarray) -> dl.Panel:
    """The panel with country-years lacking a housing return marked unavailable.

    Housing is recorded for fewer country-years than equity is. The alternative
    to restricting the panel would be to fill the missing cells, which would
    mean inventing returns; this instead draws every asset -- housing included
    -- from the same, smaller set of genuinely observed years. The four-asset
    control is re-solved on this same restricted panel, so the comparison is
    like for like and the restriction cancels out of it.
    """
    available = panel.available & np.isfinite(housing)
    return dataclasses.replace(panel, available=available,
                               name=f"{panel.name}+housing")


def gather(paths: Any, matrix: np.ndarray) -> np.ndarray:
    """``(N, H)`` housing returns for drawn paths, on their own calendar.

    The sampler records the calendar year and country behind every path-year,
    and gathers each asset by indexing its ``(T, C)`` matrix with them. Doing
    the same here means housing arrives on exactly the block structure the
    other four assets were drawn on -- same years, same countries, same blocks
    -- rather than as an independently sampled series that would break the
    cross-asset correlation the bootstrap exists to preserve.
    """
    cal = np.asarray(paths.calendar_index)
    ctry = np.asarray(paths.domestic_country)
    return matrix[cal, ctry]


def net_of_cost(gross: np.ndarray, holding_cost: float) -> np.ndarray:
    """Housing returns after an annual cost charged on the asset's value.

    The cost is a percentage of value, levied every year regardless of the
    return, which is how rates, insurance and maintenance actually fall. It is
    therefore subtracted from the return rather than taken out of the gain: a
    year that loses 10% before costs loses 12% after a 2% cost.
    """
    return np.asarray(gross, dtype=float) - float(holding_cost)


def moments(matrix: np.ndarray, holding_cost: float = 0.0) -> Dict[str, float]:
    """Pooled mean, volatility and Sharpe-free summary of the housing panel."""
    values = net_of_cost(matrix, holding_cost)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"observations": 0}
    return {
        "observations": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "geometric_mean": float(np.exp(np.mean(np.log1p(
            np.maximum(values, -0.99)))) - 1.0),
    }


def cost_grid(cfg: Mapping[str, Any]) -> List[float]:
    """The holding costs to sweep, from configuration."""
    house = cfg.get("housing", {})
    return [float(c) for c in house.get(
        "holding_costs", [0.0, 0.01, 0.02, 0.03, 0.04, 0.05])]


# ---------------------------------------------------------------------------
# Reading the sweep
# ---------------------------------------------------------------------------
def break_even_cost(frame: pd.DataFrame,
                    weight_column: str = "mean_housing",
                    cost_column: str = "holding_cost",
                    threshold: float = 0.01) -> float:
    """The holding cost at which housing's optimal weight falls to ``threshold``.

    Linear interpolation between the two swept costs that bracket the
    crossing. Returns NaN when the sweep never crosses -- either housing is
    still wanted at the most expensive cost tried, or it was never wanted at
    all -- because reporting an extrapolated number in either case would be
    inventing the answer the sweep failed to find.
    """
    ordered = frame.sort_values(cost_column)
    costs = ordered[cost_column].to_numpy(dtype=float)
    weights = ordered[weight_column].to_numpy(dtype=float)
    if weights.size < 2 or weights[0] <= threshold:
        return float("nan")
    below = np.flatnonzero(weights <= threshold)
    if not below.size:
        return float("nan")
    i = int(below[0])
    w0, w1 = weights[i - 1], weights[i]
    if not np.isfinite(w0) or not np.isfinite(w1) or w0 == w1:
        return float("nan")
    span = (w0 - threshold) / (w0 - w1)
    return float(costs[i - 1] + span * (costs[i] - costs[i - 1]))


def displacement(frame: pd.DataFrame, assets: Sequence[str] = ASSETS,
                 cost_column: str = "holding_cost") -> pd.DataFrame:
    """Where housing's weight comes from, asset by asset, as cost falls.

    The interesting question is not only how much housing the optimum holds but
    what it displaces: an asset that substitutes for bonds is a different
    proposition from one that substitutes for equity.
    """
    ordered = frame.sort_values(cost_column, ascending=False)
    if len(ordered) < 2:
        return pd.DataFrame()
    dearest = ordered.iloc[0]
    rows: List[Dict[str, Any]] = []
    for _, row in ordered.iloc[1:].iterrows():
        entry: Dict[str, Any] = {
            cost_column: float(row[cost_column]),
            "reference_cost": float(dearest[cost_column]),
        }
        for asset in assets:
            column = f"mean_{asset}"
            if column in frame.columns:
                entry[asset] = float(row[column]) - float(dearest[column])
        rows.append(entry)
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def constant_mix_optimum(evaluator: Any, gamma: float,
                         coarse_step: float = 0.10,
                         fine_step: float = 0.025,
                         max_batch: int = 12,
                         label: str = "") -> Tuple[np.ndarray, float, int]:
    """Best single allocation held for life, over the whole weight simplex.

    A coarse lattice sweep followed by pairwise-exchange refinement, the same
    two-stage design as :func:`src.allocation.optimise_full_simplex` but over
    one allocation rather than sixty-eight. Held constant deliberately: the
    sweep runs at every holding cost, and a single number per cost is what
    makes the costs comparable. :func:`solve_sweep` re-solves the age-varying
    schedule at selected costs to check the restriction does not drive the
    answer.
    """
    from . import allocation as al

    n_assets = len(getattr(evaluator, "assets", ASSETS))
    horizon = evaluator.spec.horizon
    lattice = al.simplex_lattice(coarse_step, n_assets)
    schedules = np.repeat(lattice[:, None, :], horizon, axis=1)
    scores = evaluator.cec_chunked(schedules, gamma, max_batch=max_batch)
    evaluations = int(len(lattice))
    best_weights = lattice[int(np.argmax(scores))].copy()
    best = float(np.max(scores))
    LOGGER.info("%scoarse lattice (%d points): CEC=%.6f at %s",
                label, len(lattice), best,
                np.round(best_weights, 3).tolist())

    while True:
        candidates = al.exchange_neighbourhood(best_weights, fine_step)
        variants = np.repeat(candidates[:, None, :], horizon, axis=1)
        scores = evaluator.cec_chunked(variants, gamma, max_batch=max_batch)
        evaluations += int(len(candidates))
        pick = int(np.argmax(scores))
        if scores[pick] <= best * (1.0 + 1e-7):
            break
        best = float(scores[pick])
        best_weights = candidates[pick].copy()
    LOGGER.info("%srefined: CEC=%.6f at %s (%d evaluations)",
                label, best, np.round(best_weights, 3).tolist(), evaluations)
    return best_weights, best, evaluations


def solve_sweep(paths: Any, spec: Any, income: np.ndarray,
                cfg: Mapping[str, Any], gross: np.ndarray,
                costs: Sequence[float], gamma: float,
                coarse_step: float = 0.10,
                fine_step: float = 0.025,
                variant: str = "de-smoothed") -> pd.DataFrame:
    """Re-solve the allocation at each annual holding cost.

    Every cost is solved against the *same* paths and the same income draws --
    common random numbers -- so differences between rows are the cost and
    nothing else. The four-asset control is solved on those same paths too,
    which is what makes "housing is worth X" a statement about housing rather
    than about the restricted panel it had to be drawn on.
    """
    from . import glidepath as gp

    rows: List[Dict[str, Any]] = []
    control = gp.BatchEvaluator(paths, spec, income, cfg)
    n_core = len(control.assets)
    core_weights, core_cec, _ = constant_mix_optimum(
        control, gamma, coarse_step, fine_step, label="[no housing] ")
    baseline = {f"mean_{a}": float(core_weights[i])
                for i, a in enumerate(control.assets)}
    rows.append({
        "holding_cost": float("nan"), "investable_set": "four assets",
        "variant": variant, "cec": core_cec, "advantage_pct": 0.0, **baseline,
        f"mean_{HOUSING}": 0.0})

    for cost in costs:
        net = net_of_cost(gross, cost)
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg, assets=ASSETS,
                                      extra={HOUSING: net})
        weights, cec, evaluations = constant_mix_optimum(
            evaluator, gamma, coarse_step, fine_step,
            label=f"[housing @ {cost:.1%}] ")
        row: Dict[str, Any] = {
            "holding_cost": float(cost),
            "investable_set": "five assets",
            "variant": variant,
            "cec": float(cec),
            "advantage_pct": (cec / core_cec - 1.0) * 100.0,
            "evaluations": int(evaluations),
        }
        row.update({f"mean_{a}": float(weights[i])
                    for i, a in enumerate(evaluator.assets)})
        row["equity"] = float(row["mean_dom_eq"] + row["mean_intl_eq"])
        row["fixed_income"] = float(row["mean_bond"] + row["mean_bill"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def age_varying_check(paths: Any, spec: Any, income: np.ndarray,
                      cfg: Mapping[str, Any], gross: np.ndarray,
                      costs: Sequence[float], gamma: float,
                      constant: pd.DataFrame,
                      coarse_step: float = 0.25,
                      fine_step: float = 0.05,
                      coarse_sweeps: int = 2,
                      fine_sweeps: int = 3) -> pd.DataFrame:
    """Does letting the allocation vary with age change the housing answer?

    The sweep holds one allocation for life so that every holding cost is
    described by a single comparable number. That is a restriction, and this
    removes it at a couple of costs: the full age-by-asset schedule is solved
    over the five-asset simplex and its mean housing weight compared with the
    constant-mix answer. If the two disagree materially, the constant-mix
    reading is the one to distrust.
    """
    from . import allocation as al
    from . import glidepath as gp

    reference = constant.set_index("holding_cost")
    rows: List[Dict[str, Any]] = []
    for cost in costs:
        net = net_of_cost(gross, cost)
        evaluator = gp.BatchEvaluator(paths, spec, income, cfg, assets=ASSETS,
                                      extra={HOUSING: net})
        schedule, cec, _ = al.optimise_full_simplex(
            evaluator, gamma, coarse_step=coarse_step, fine_step=fine_step,
            coarse_sweeps=coarse_sweeps, fine_sweeps=fine_sweeps,
            label=f"[age-varying @ {cost:.1%}] ")
        row: Dict[str, Any] = {"holding_cost": float(cost), "cec": float(cec)}
        row.update({f"mean_{a}": float(schedule[:, i].mean())
                    for i, a in enumerate(evaluator.assets)})
        housing_column = schedule[:, list(evaluator.assets).index(HOUSING)]
        row["housing_working"] = float(housing_column[:spec.n_working].mean())
        row["housing_retired"] = float(housing_column[spec.n_working:].mean())
        if float(cost) in reference.index:
            fixed = reference.loc[float(cost)]
            row["constant_mix_housing"] = float(fixed[f"mean_{HOUSING}"])
            row["constant_mix_cec"] = float(fixed["cec"])
            row["housing_difference"] = (row[f"mean_{HOUSING}"]
                                         - row["constant_mix_housing"])
            row["cec_gain_pct"] = (cec / float(fixed["cec"]) - 1.0) * 100.0
        rows.append(row)
    return pd.DataFrame.from_records(rows)
