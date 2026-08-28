"""Sensitivity analysis across allocation, preference and planning parameters.

The headline result in ``docs/04`` is a single point in a large parameter
space.  This module walks that space and asks the only question that matters
for a replication: **which assumptions, if any, overturn the ranking?**

Two design choices make the answers interpretable.

**Common random numbers.**  One set of bootstrap paths and one set of
labour-income shocks are drawn up front and reused at every sweep point.  A
difference between two settings is therefore the parameter's effect, not
Monte Carlo noise -- which matters because many of the gaps being measured are
a few percent of certainty equivalent consumption.

**Re-simulate only what changed.**  Preference parameters (``gamma``,
``psi``, ``beta``, the bequest weight) enter only through the utility
aggregator, so those sweeps re-evaluate cached consumption paths rather than
re-running the lifecycle.  Allocation and planning parameters need a fresh
lifecycle pass but reuse the cached returns.  Only bootstrap parameters force
a fresh draw.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import bootstrap as bs
from . import lifecycle as lc
from . import spending as spg
from . import utility as ut
from .data_loader import Panel

LOGGER = logging.getLogger(__name__)

#: Strategies compared in every planning-parameter sweep.
CORE_STRATEGIES: Tuple[str, ...] = (
    "balanced_all_equity", "target_date_fund", "sixty_forty",
    "domestic_equity", "international_equity", "bills_only",
)

#: The portfolio whose advantage the sweeps are measuring.
CHALLENGER = "balanced_all_equity"

#: The portfolios it has to beat for the paper's claim to hold.
INCUMBENTS: Tuple[str, ...] = ("target_date_fund", "sixty_forty")


# ---------------------------------------------------------------------------
# Strategy construction helpers
# ---------------------------------------------------------------------------
def constant_mix(key: str, label: str, weights: Mapping[str, float],
                 horizon: int) -> lc.Strategy:
    """Build a fixed-weight :class:`~src.lifecycle.Strategy`."""
    row = np.array([float(weights.get(asset, 0.0)) for asset in lc.ASSETS])
    total = row.sum()
    if not np.isclose(total, 1.0):
        raise ValueError(f"weights for {key!r} sum to {total:.4f}, not 1")
    return lc.Strategy(key=key, label=label, weights=np.tile(row, (horizon, 1)))


def all_equity_split(domestic_share: float, horizon: int) -> lc.Strategy:
    """All-equity portfolio with ``domestic_share`` in the home market."""
    share = float(np.clip(domestic_share, 0.0, 1.0))
    return constant_mix(
        key=f"dom{share:.2f}",
        label=f"{share:.0%} domestic / {1 - share:.0%} international",
        weights={"dom_eq": share, "intl_eq": 1.0 - share},
        horizon=horizon,
    )


def equity_fixed_income_mix(equity_share: float, horizon: int,
                            domestic_share: float = 0.5,
                            bond_share: float = 0.7) -> lc.Strategy:
    """Constant mix of a 50/50 equity sleeve against a bond/bill sleeve."""
    eq = float(np.clip(equity_share, 0.0, 1.0))
    fi = 1.0 - eq
    return constant_mix(
        key=f"eq{eq:.2f}",
        label=f"{eq:.0%} equity / {fi:.0%} fixed income",
        weights={
            "dom_eq": eq * domestic_share,
            "intl_eq": eq * (1.0 - domestic_share),
            "bond": fi * bond_share,
            "bill": fi * (1.0 - bond_share),
        },
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Sweep context
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SweepContext:
    """Cached draws shared by every sweep point."""

    cfg: Mapping[str, Any]
    panel: Panel
    paths: bs.BootstrapPaths
    shocks: Tuple[np.ndarray, np.ndarray]
    base_spec: lc.LifecycleSpec
    n_paths: int

    @classmethod
    def build(cls, cfg: Mapping[str, Any], panel: Panel, n_paths: int,
              max_horizon: int, max_working: int,
              seed: int | None = None) -> "SweepContext":
        """Draw the shared return paths and income shocks once."""
        sampler = bs.from_config(panel, cfg, horizon_years=max_horizon,
                                 **({"seed": seed} if seed is not None else {}))
        LOGGER.info("sweep context: drawing %s paths over %s years",
                    f"{n_paths:,}", max_horizon)
        paths = sampler.sample(n_paths, chunk_size=int(cfg["bootstrap"]["chunk_size"]))
        rng = np.random.default_rng(int(cfg["run"]["seed"]))
        shocks = lc.draw_income_shocks(n_paths, max_working, rng)
        return cls(cfg=cfg, panel=panel, paths=paths, shocks=shocks,
                   base_spec=lc.spec_from_config(cfg), n_paths=n_paths)

    # -- per-point execution ------------------------------------------------
    def spec_with(self, **overrides: Any) -> lc.LifecycleSpec:
        """Base lifecycle spec with selected fields replaced."""
        return dataclasses.replace(self.base_spec, **overrides)

    def income_for(self, spec: lc.LifecycleSpec) -> np.ndarray:
        return lc.simulate_income(spec, self.n_paths, shocks=self.shocks)

    def run(self, strategies: Mapping[str, lc.Strategy],
            spec: lc.LifecycleSpec | None = None
            ) -> Dict[str, lc.LifecycleOutcome]:
        """Simulate every strategy on the cached paths."""
        spec = spec or self.base_spec
        if spec.horizon > self.paths.horizon:
            raise ValueError(
                f"spec horizon {spec.horizon} exceeds the cached "
                f"{self.paths.horizon}-year draw; raise max_horizon")
        return lc.simulate_all(self.paths, strategies, spec,
                               self.income_for(spec))

    def strategies_from_config(self, spec: lc.LifecycleSpec | None = None
                               ) -> Dict[str, lc.Strategy]:
        return lc.build_strategies(self.cfg, spec or self.base_spec)


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------
def summarise(results: Mapping[str, lc.LifecycleOutcome],
              cfg: Mapping[str, Any], spec: lc.LifecycleSpec,
              gammas: Sequence[float] | None = None,
              ies: Sequence[float] | None = None,
              extra: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Compact metric row per strategy, optionally at custom preferences."""
    util = cfg["utility"]
    beta = float(util["discount_factor"])
    bequest_weight = float(util["bequest_weight"])
    include_bequest = bool(util["bequest_enabled"])
    gammas = list(gammas if gammas is not None else util["risk_aversions"])
    ies = list(ies if ies is not None else [float(util["baseline_ies"])])
    replacement = float(cfg["report"].get("consumption_target_replacement", 0.70))

    rows: List[Dict[str, Any]] = []
    for key, outcome in results.items():
        bundle = ut.bundle_from_outcome(outcome, cfg, spec)
        row: Dict[str, Any] = dict(extra or {})
        row.update({"strategy": key, "label": outcome.label})
        for gamma in gammas:
            row[f"cec_crra_gamma{float(gamma):g}"] = ut.crra_certainty_equivalent(
                bundle, float(gamma), beta, bequest_weight, include_bequest)
        for gamma in gammas:
            for psi in ies:
                row[f"cec_ez_gamma{float(gamma):g}_psi{float(psi):g}"] = \
                    ut.epstein_zin_certainty_equivalent(
                        bundle, float(gamma), float(psi), beta,
                        bequest_weight, include_bequest)
        row.update(ut.shortfall_metrics(
            bundle, outcome.ruin, outcome.wealth_at_retirement,
            slice(0, bundle.horizon),
            percentiles=cfg["report"]["percentiles"],
            consumption_target=replacement * outcome.career_average_income,
        ))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def advantage(frame: pd.DataFrame, metric: str,
              challenger: str = CHALLENGER,
              incumbents: Sequence[str] = INCUMBENTS,
              group: Sequence[str] = ()) -> pd.DataFrame:
    """Percentage advantage of ``challenger`` over each incumbent."""
    keys = list(group)
    rows: List[Dict[str, Any]] = []
    groups = frame.groupby(keys) if keys else [((), frame)]
    for name, block in groups:
        indexed = block.set_index("strategy")
        if challenger not in indexed.index:
            continue
        base = float(indexed.loc[challenger, metric])
        record: Dict[str, Any] = {}
        if keys:
            values = name if isinstance(name, tuple) else (name,)
            record.update(dict(zip(keys, values)))
        record[f"{challenger}"] = base
        for incumbent in incumbents:
            if incumbent not in indexed.index:
                continue
            other = float(indexed.loc[incumbent, metric])
            record[incumbent] = other
            record[f"advantage_vs_{incumbent}_pct"] = (
                (base / other - 1.0) * 100.0 if other != 0 else np.nan)
        record["challenger_wins_all"] = all(
            base > float(indexed.loc[i, metric])
            for i in incumbents if i in indexed.index)
        rows.append(record)
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Allocation sweeps  (re-simulate the lifecycle, reuse the returns)
# ---------------------------------------------------------------------------
def sweep_domestic_share(ctx: SweepContext,
                         grid: Sequence[float] | None = None,
                         gammas: Sequence[float] | None = None
                         ) -> pd.DataFrame:
    """Vary the domestic share of an all-equity portfolio from 0 to 100%."""
    grid = list(grid if grid is not None else np.linspace(0.0, 1.0, 11))
    spec = ctx.base_spec
    frames: List[pd.DataFrame] = []
    for share in grid:
        strategy = all_equity_split(float(share), spec.horizon)
        results = ctx.run({strategy.key: strategy}, spec)
        frames.append(summarise(results, ctx.cfg, spec, gammas=gammas,
                                extra={"domestic_share": float(share)}))
    return pd.concat(frames, ignore_index=True)


def sweep_equity_share(ctx: SweepContext,
                       grid: Sequence[float] | None = None,
                       gammas: Sequence[float] | None = None,
                       domestic_share: float = 0.5) -> pd.DataFrame:
    """Vary the equity share of a constant-mix portfolio from 0 to 100%."""
    grid = list(grid if grid is not None else np.linspace(0.0, 1.0, 11))
    spec = ctx.base_spec
    frames: List[pd.DataFrame] = []
    for share in grid:
        strategy = equity_fixed_income_mix(float(share), spec.horizon,
                                           domestic_share=domestic_share)
        results = ctx.run({strategy.key: strategy}, spec)
        frames.append(summarise(results, ctx.cfg, spec, gammas=gammas,
                                extra={"equity_share": float(share)}))
    return pd.concat(frames, ignore_index=True)


def optimal_allocation(frame: pd.DataFrame, parameter: str,
                       gammas: Sequence[float]) -> pd.DataFrame:
    """Argmax of certainty equivalent consumption along an allocation sweep."""
    rows = []
    for gamma in gammas:
        column = f"cec_crra_gamma{float(gamma):g}"
        if column not in frame.columns:
            continue
        best = frame.loc[frame[column].idxmax()]
        rows.append({
            "risk_aversion": float(gamma),
            f"optimal_{parameter}": float(best[parameter]),
            "cec_at_optimum": float(best[column]),
            "prob_ruin_at_optimum": float(best["prob_ruin"]),
        })
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Preference sweeps  (no re-simulation needed)
# ---------------------------------------------------------------------------
def sweep_risk_aversion(ctx: SweepContext,
                        strategies: Mapping[str, lc.Strategy],
                        grid: Sequence[float]) -> pd.DataFrame:
    """Certainty equivalents across a fine grid of CRRA risk aversion.

    Consumption paths do not depend on ``gamma``, so this runs the lifecycle
    once and re-evaluates the aggregator.
    """
    spec = ctx.base_spec
    results = ctx.run(strategies, spec)
    util = ctx.cfg["utility"]
    beta = float(util["discount_factor"])
    weight = float(util["bequest_weight"])
    include = bool(util["bequest_enabled"])
    rows: List[Dict[str, Any]] = []
    for key, outcome in results.items():
        bundle = ut.bundle_from_outcome(outcome, ctx.cfg, spec)
        for gamma in grid:
            rows.append({
                "risk_aversion": float(gamma),
                "strategy": key,
                "label": outcome.label,
                "cec": ut.crra_certainty_equivalent(
                    bundle, float(gamma), beta, weight, include),
            })
    return pd.DataFrame.from_records(rows)


def sweep_ies(ctx: SweepContext, strategies: Mapping[str, lc.Strategy],
              grid: Sequence[float], gamma: float | None = None
              ) -> pd.DataFrame:
    """Epstein-Zin certainty equivalents across a grid of IES values."""
    spec = ctx.base_spec
    util = ctx.cfg["utility"]
    gamma = float(gamma if gamma is not None else util["baseline_risk_aversion"])
    beta = float(util["discount_factor"])
    weight = float(util["bequest_weight"])
    include = bool(util["bequest_enabled"])
    results = ctx.run(strategies, spec)
    rows: List[Dict[str, Any]] = []
    for key, outcome in results.items():
        bundle = ut.bundle_from_outcome(outcome, ctx.cfg, spec)
        for psi in grid:
            rows.append({
                "ies": float(psi),
                "risk_aversion": gamma,
                "strategy": key,
                "label": outcome.label,
                "cec": ut.epstein_zin_certainty_equivalent(
                    bundle, gamma, float(psi), beta, weight, include),
            })
    return pd.DataFrame.from_records(rows)


def sweep_bequest_weight(ctx: SweepContext,
                         strategies: Mapping[str, lc.Strategy],
                         grid: Sequence[float],
                         gamma: float | None = None) -> pd.DataFrame:
    """Does the result depend on the strength of the bequest motive?"""
    spec = ctx.base_spec
    util = ctx.cfg["utility"]
    gamma = float(gamma if gamma is not None else util["baseline_risk_aversion"])
    beta = float(util["discount_factor"])
    results = ctx.run(strategies, spec)
    rows: List[Dict[str, Any]] = []
    for key, outcome in results.items():
        bundle = ut.bundle_from_outcome(outcome, ctx.cfg, spec)
        for weight in grid:
            rows.append({
                "bequest_weight": float(weight),
                "strategy": key,
                "label": outcome.label,
                "cec": ut.crra_certainty_equivalent(
                    bundle, gamma, beta, float(weight), float(weight) > 0),
            })
    return pd.DataFrame.from_records(rows)


def crossover_risk_aversion(frame: pd.DataFrame,
                            challenger: str = CHALLENGER,
                            incumbents: Sequence[str] = INCUMBENTS
                            ) -> pd.DataFrame:
    """The risk aversion at which an incumbent would overtake the challenger.

    Reports ``inf`` when the challenger still leads at the top of the grid,
    which is the interesting outcome: no amount of risk aversion in the
    tested range reverses the ranking.
    """
    wide = frame.pivot_table(index="risk_aversion", columns="strategy",
                             values="cec").sort_index()
    rows = []
    for incumbent in incumbents:
        if incumbent not in wide.columns or challenger not in wide.columns:
            continue
        gap = wide[challenger] - wide[incumbent]
        crossed = gap <= 0
        if not crossed.any():
            crossover = float("inf")
        else:
            first = int(np.flatnonzero(crossed.to_numpy())[0])
            if first == 0:
                crossover = float(wide.index[0])
            else:
                x0, x1 = wide.index[first - 1], wide.index[first]
                g0, g1 = gap.iloc[first - 1], gap.iloc[first]
                crossover = float(x0 + (x1 - x0) * g0 / (g0 - g1))
        rows.append({
            "incumbent": incumbent,
            "crossover_risk_aversion": crossover,
            "challenger_leads_at_max_gamma": bool(gap.iloc[-1] > 0),
            "gap_at_max_gamma_pct": float(
                gap.iloc[-1] / wide[incumbent].iloc[-1] * 100.0),
        })
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Planning-parameter sweeps  (re-simulate the lifecycle, reuse the returns)
# ---------------------------------------------------------------------------
#: Lifecycle fields the generic sweeper knows how to vary, with the label
#: used for the swept column in the output frame.
LIFECYCLE_FIELDS: Mapping[str, str] = {
    "age_death": "age_death",
    "age_retire": "age_retire",
    "savings_rate": "savings_rate",
    "rule_rate": "withdrawal_rate",
    "replacement_rate": "replacement_rate",
    "permanent_shock_sd": "permanent_shock_sd",
}


def sweep_lifecycle_field(ctx: SweepContext,
                          strategies: Mapping[str, lc.Strategy],
                          field: str, grid: Sequence[Any],
                          gammas: Sequence[float] | None = None
                          ) -> pd.DataFrame:
    """Vary one lifecycle parameter and re-run every strategy.

    Strategies whose weights are age-indexed (the glide path) are rebuilt at
    each point, since changing the retirement or death age changes the
    horizon the glide path has to span.
    """
    if field not in LIFECYCLE_FIELDS:
        raise ValueError(
            f"unknown lifecycle field {field!r}; expected one of "
            f"{sorted(LIFECYCLE_FIELDS)}")
    column = LIFECYCLE_FIELDS[field]
    frames: List[pd.DataFrame] = []
    for value in grid:
        spec = ctx.spec_with(**{field: type(getattr(ctx.base_spec, field))(value)})
        rebuilt = lc.build_strategies(ctx.cfg, spec)
        active = {k: v for k, v in rebuilt.items() if k in strategies}
        results = ctx.run(active, spec)
        frames.append(summarise(results, ctx.cfg, spec, gammas=gammas,
                                extra={column: value}))
    return pd.concat(frames, ignore_index=True)


def sweep_social_security(ctx: SweepContext,
                          strategies: Mapping[str, lc.Strategy],
                          gammas: Sequence[float] | None = None
                          ) -> pd.DataFrame:
    """Progressive schedule versus a flat replacement rate versus none."""
    variants = {
        "progressive (US bend points)": {"social_security_enabled": True,
                                         "social_security_formula": "progressive"},
        "flat 45% replacement": {"social_security_enabled": True,
                                 "social_security_formula": "flat",
                                 "replacement_rate": 0.45},
        "flat 25% replacement": {"social_security_enabled": True,
                                 "social_security_formula": "flat",
                                 "replacement_rate": 0.25},
        "flat 10% replacement": {"social_security_enabled": True,
                                 "social_security_formula": "flat",
                                 "replacement_rate": 0.10},
    }
    # A literal "no social security" variant is deliberately excluded: with no
    # floor at all, a ruined investor consumes zero, every certainty
    # equivalent collapses to the numerical consumption floor, and the metric
    # stops measuring anything.  The 10% variant is the low-support case.
    frames: List[pd.DataFrame] = []
    for name, overrides in variants.items():
        spec = ctx.spec_with(**overrides)
        results = ctx.run(strategies, spec)
        frames.append(summarise(results, ctx.cfg, spec, gammas=gammas,
                                extra={"social_security": name}))
    return pd.concat(frames, ignore_index=True)


def safe_withdrawal_rates(frame: pd.DataFrame,
                          target_ruin: float = 0.05,
                          rate_column: str = "withdrawal_rate"
                          ) -> pd.DataFrame:
    """Interpolate the withdrawal rate at which ruin probability hits a target.

    The classic "safe withdrawal rate" question, answered per strategy from
    the ruin curve produced by :func:`sweep_lifecycle_field` on ``rule_rate``.
    """
    rows = []
    for strategy, block in frame.groupby("strategy"):
        block = block.sort_values(rate_column)
        rates = block[rate_column].to_numpy(dtype=float)
        ruin = block["prob_ruin"].to_numpy(dtype=float)
        label = block["label"].iloc[0]
        if ruin.min() > target_ruin:
            swr = float("nan")     # never safe enough on this grid
        elif ruin.max() < target_ruin:
            swr = float(rates.max())  # safe across the whole grid
        else:
            swr = float(np.interp(target_ruin, ruin, rates))
        rows.append({
            "strategy": strategy,
            "label": label,
            f"safe_withdrawal_rate_at_{target_ruin:.0%}_ruin": swr,
            "ruin_at_4pct": float(np.interp(0.04, rates, ruin)),
        })
    return pd.DataFrame.from_records(rows).sort_values(
        f"safe_withdrawal_rate_at_{target_ruin:.0%}_ruin", ascending=False)


# ---------------------------------------------------------------------------
# Bootstrap-parameter sweeps  (fresh draws required)
# ---------------------------------------------------------------------------
def sweep_bootstrap_field(cfg: Mapping[str, Any], panel: Panel,
                          strategies: Mapping[str, lc.Strategy],
                          spec: lc.LifecycleSpec, field: str,
                          grid: Sequence[Any], n_paths: int,
                          gammas: Sequence[float] | None = None
                          ) -> pd.DataFrame:
    """Vary a sampler parameter, redrawing the paths at each point."""
    chunk = int(cfg["bootstrap"]["chunk_size"])
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    shocks = lc.draw_income_shocks(n_paths, spec.n_working, rng)
    income = lc.simulate_income(spec, n_paths, shocks=shocks)
    frames: List[pd.DataFrame] = []
    for value in grid:
        sampler = bs.from_config(panel, cfg, **{field: value},
                                 horizon_years=spec.horizon)
        paths = sampler.sample(n_paths, chunk_size=chunk)
        results = lc.simulate_all(paths, strategies, spec, income)
        frames.append(summarise(results, cfg, spec, gammas=gammas,
                                extra={field: value}))
    return pd.concat(frames, ignore_index=True)


def sweep_panels(cfg: Mapping[str, Any], panels: Mapping[str, Panel],
                 strategies: Mapping[str, lc.Strategy],
                 spec: lc.LifecycleSpec, n_paths: int,
                 gammas: Sequence[float] | None = None) -> pd.DataFrame:
    """Re-run the comparison on each available panel."""
    chunk = int(cfg["bootstrap"]["chunk_size"])
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    shocks = lc.draw_income_shocks(n_paths, spec.n_working, rng)
    income = lc.simulate_income(spec, n_paths, shocks=shocks)
    frames: List[pd.DataFrame] = []
    for name, panel in panels.items():
        sampler = bs.from_config(panel, cfg, horizon_years=spec.horizon)
        paths = sampler.sample(n_paths, chunk_size=chunk)
        results = lc.simulate_all(paths, strategies, spec, income)
        frames.append(summarise(results, cfg, spec, gammas=gammas,
                                extra={"panel": name}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Cross-sweep summary
# ---------------------------------------------------------------------------
def tornado(sweeps: Mapping[str, Tuple[pd.DataFrame, str]], metric: str,
            challenger: str = CHALLENGER,
            incumbents: Sequence[str] = INCUMBENTS) -> pd.DataFrame:
    """Range of the challenger's advantage across every swept dimension.

    ``sweeps`` maps a human-readable dimension name to
    ``(frame, parameter_column)``.  The output is the classic tornado input:
    which assumption moves the answer most, and does any setting flip it?
    """
    rows: List[Dict[str, Any]] = []
    for name, (frame, column) in sweeps.items():
        if column not in frame.columns or metric not in frame.columns:
            continue
        adv = advantage(frame, metric, challenger, incumbents, group=[column])
        for incumbent in incumbents:
            key = f"advantage_vs_{incumbent}_pct"
            if key not in adv.columns:
                continue
            values = adv[key].dropna()
            if values.empty:
                continue
            rows.append({
                "dimension": name,
                "parameter": column,
                "incumbent": incumbent,
                "n_settings": int(len(values)),
                "min_advantage_pct": float(values.min()),
                "median_advantage_pct": float(values.median()),
                "max_advantage_pct": float(values.max()),
                "range_pp": float(values.max() - values.min()),
                "challenger_always_wins": bool((values > 0).all()),
                "settings_lost": int((values <= 0).sum()),
            })
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values("range_pp", ascending=False) if len(frame) else frame


def overall_verdict(tornado_frame: pd.DataFrame) -> Dict[str, Any]:
    """One-line summary of how robust the ranking is."""
    if tornado_frame.empty:
        return {"n_settings": 0, "n_lost": 0, "always_wins": False}
    return {
        "n_settings": int(tornado_frame["n_settings"].sum()),
        "n_lost": int(tornado_frame["settings_lost"].sum()),
        "always_wins": bool(tornado_frame["challenger_always_wins"].all()),
        "worst_dimension": str(tornado_frame.iloc[0]["dimension"]),
        "min_advantage_pct": float(tornado_frame["min_advantage_pct"].min()),
        "max_advantage_pct": float(tornado_frame["max_advantage_pct"].max()),
    }


# ---------------------------------------------------------------------------
# Retirement spending rules
# ---------------------------------------------------------------------------
def spending_metrics(outcome: lc.LifecycleOutcome, spec: lc.LifecycleSpec
                     ) -> Dict[str, float]:
    """Consumption-shape diagnostics that explain a rule's certainty equivalent.

    The certainty equivalent already prices volatility; these columns say
    *where* the volatility comes from, which is what a retiree choosing a rule
    actually wants to know.  ``median_worst_spending_cut`` is the deepest
    peak-to-trough fall in real retirement consumption on a typical path --
    the number a rule's critics usually point at.
    """
    consumption = outcome.consumption[:, spec.retirement_slice]
    floor = 1e-9
    logs = np.log(np.maximum(consumption, floor))
    running_max = np.maximum.accumulate(consumption, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        drawdown = np.where(running_max > 0.0, 1.0 - consumption / running_max, 0.0)
    total = consumption.sum(axis=1)
    return {
        "consumption_volatility": float(np.mean(logs.std(axis=1, ddof=1))),
        "median_worst_spending_cut": float(np.median(drawdown.max(axis=1))),
        "p90_worst_spending_cut": float(np.percentile(drawdown.max(axis=1), 90)),
        "median_total_real_spending": float(np.median(total)),
        "median_first_year_spending": float(np.median(consumption[:, 0])),
        "median_final_year_spending": float(np.median(consumption[:, -1])),
    }


def summarise_spending(outcome: lc.LifecycleOutcome, cfg: Mapping[str, Any],
                       spec: lc.LifecycleSpec,
                       gammas: Sequence[float] | None = None,
                       extra: Mapping[str, Any] | None = None
                       ) -> Dict[str, Any]:
    """One row describing a (strategy, spending rule) pair.

    Certainty equivalents are reported twice: with the configured bequest
    weight, and with no bequest motive at all.  Horizon-based rules spend the
    portfolio down to nothing by construction, so a bequest motive is exactly
    the assumption that decides between them and the fixed-amount rules --
    reporting only one would hide the mechanism.
    """
    util = cfg["utility"]
    beta = float(util["discount_factor"])
    weight = float(util["bequest_weight"])
    gammas = list(gammas if gammas is not None else util["risk_aversions"])
    replacement = float(cfg["report"].get("consumption_target_replacement", 0.70))

    bundle = ut.bundle_from_outcome(outcome, cfg, spec)
    row: Dict[str, Any] = dict(extra or {})
    row["strategy"] = outcome.strategy
    for gamma in gammas:
        row[f"cec_gamma{float(gamma):g}"] = ut.crra_certainty_equivalent(
            bundle, float(gamma), beta, weight, True)
        row[f"cec_nobequest_gamma{float(gamma):g}"] = \
            ut.crra_certainty_equivalent(bundle, float(gamma), beta, 0.0, False)
    row["prob_ruin"] = float(outcome.ruin.mean())
    row["median_bequest"] = float(np.median(outcome.bequest))
    row["p5_bequest"] = float(np.percentile(outcome.bequest, 5))
    shortfall = ut.shortfall_metrics(
        bundle, outcome.ruin, outcome.wealth_at_retirement,
        slice(0, bundle.horizon),
        percentiles=(5, 50),
        consumption_target=replacement * outcome.career_average_income)
    row["median_retirement_consumption"] = \
        shortfall["median_retirement_consumption"]
    row["p5_retirement_consumption"] = shortfall["p5_retirement_consumption"]
    row.update(spending_metrics(outcome, spec))
    return row


def sweep_spending_rules(
    ctx: SweepContext,
    rule_specs: Sequence[Mapping[str, Any]],
    rate_grid: Sequence[float],
    strategy_key: str = CHALLENGER,
    gammas: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Evaluate every spending rule, sweeping the rate where a rule has one.

    Comparing families at a common headline rate would not be a fair test: a
    4% constant-real withdrawal and a 4% share-of-portfolio withdrawal are
    different amounts of money, and the horizon-based rules have no rate to
    set at all.  Each rule is therefore swept over its own rate grid, and the
    comparison in ``docs/06`` is made at each rule's own optimum.
    """
    spec = ctx.base_spec
    strategies = ctx.strategies_from_config(spec)
    if strategy_key not in strategies:
        raise ValueError(f"unknown strategy {strategy_key!r}")
    strategy = strategies[strategy_key]

    rows: List[Dict[str, Any]] = []
    for entry in rule_specs:
        key = str(entry["key"])
        params = dict(entry.get("params", {}) or {})
        suffix = entry.get("suffix")
        rates = (list(rate_grid) if key in spg.RATE_PARAMETERISED
                 else [float("nan")])
        for rate in rates:
            call = dict(params)
            if key in spg.RATE_PARAMETERISED:
                call["rate"] = float(rate)
            rule = spg.build(key, **call)
            label = rule.label + (f" ({suffix})" if suffix else "")
            results = lc.simulate_all(ctx.paths, {strategy_key: strategy},
                                      spec, ctx.income_for(spec), rule)
            outcome = results[strategy_key]
            rows.append(summarise_spending(
                outcome, ctx.cfg, spec, gammas,
                extra={"rule": key, "variant": label,
                       "rate": float(rate) if rate == rate else np.nan,
                       "parameters": ", ".join(f"{k}={v}" for k, v in
                                               sorted(params.items())) or "-"}))
    return pd.DataFrame.from_records(rows)


def best_spending_rules(frame: pd.DataFrame, metric: str,
                        group: str = "variant") -> pd.DataFrame:
    """Each rule variant at the rate that maximises ``metric``."""
    best = frame.loc[frame.groupby(group)[metric].idxmax()]
    return best.sort_values(metric, ascending=False).reset_index(drop=True)


def spending_by_strategy(
    ctx: SweepContext,
    rules: Mapping[str, "spg.SpendingRule"],
    strategy_keys: Sequence[str] = CORE_STRATEGIES,
    gammas: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Cross every strategy with a shortlist of already-tuned spending rules."""
    spec = ctx.base_spec
    strategies = ctx.strategies_from_config(spec)
    active = {k: v for k, v in strategies.items() if k in strategy_keys}
    income = ctx.income_for(spec)
    rows: List[Dict[str, Any]] = []
    for label, rule in rules.items():
        results = lc.simulate_all(ctx.paths, active, spec, income, rule)
        for outcome in results.values():
            rows.append(summarise_spending(
                outcome, ctx.cfg, spec, gammas,
                extra={"variant": label, "label": outcome.label}))
    return pd.DataFrame.from_records(rows)


def spending_bequest_sensitivity(
    ctx: SweepContext,
    rules: Mapping[str, "spg.SpendingRule"],
    weights: Sequence[float],
    strategy_key: str = CHALLENGER,
    gamma: float | None = None,
) -> pd.DataFrame:
    """Re-rank the spending rules across bequest motives.

    This is the pivot of the whole comparison. Horizon-based rules spend the
    portfolio down to nothing by design; fixed-amount rules die with most of
    it unspent. Which family wins is therefore mostly a question of how much
    the investor values the money they leave behind, so the ranking is
    reported across the full range rather than at one assumed weight.
    """
    spec = ctx.base_spec
    strategies = ctx.strategies_from_config(spec)
    strategy = {strategy_key: strategies[strategy_key]}
    income = ctx.income_for(spec)
    util = ctx.cfg["utility"]
    gamma = float(gamma if gamma is not None else util["baseline_risk_aversion"])
    beta = float(util["discount_factor"])

    rows: List[Dict[str, Any]] = []
    for label, rule in rules.items():
        outcome = lc.simulate_all(ctx.paths, strategy, spec, income,
                                  rule)[strategy_key]
        bundle = ut.bundle_from_outcome(outcome, ctx.cfg, spec)
        for weight in weights:
            rows.append({
                "variant": label,
                "bequest_weight": float(weight),
                "cec": ut.crra_certainty_equivalent(
                    bundle, gamma, beta, float(weight), float(weight) > 0),
                "median_bequest": float(np.median(outcome.bequest)),
            })
    return pd.DataFrame.from_records(rows)


def spending_paths(
    ctx: SweepContext,
    rules: Mapping[str, "spg.SpendingRule"],
    strategy_key: str = CHALLENGER,
    percentiles: Sequence[float] = (10, 50, 90),
) -> pd.DataFrame:
    """Real retirement consumption by age, for plotting each rule's shape."""
    spec = ctx.base_spec
    strategies = ctx.strategies_from_config(spec)
    strategy = {strategy_key: strategies[strategy_key]}
    income = ctx.income_for(spec)
    ages = np.arange(spec.age_retire, spec.age_death)
    rows: List[Dict[str, Any]] = []
    for label, rule in rules.items():
        outcome = lc.simulate_all(ctx.paths, strategy, spec, income,
                                  rule)[strategy_key]
        consumption = outcome.consumption[:, spec.retirement_slice]
        for i, age in enumerate(ages):
            record: Dict[str, Any] = {"variant": label, "age": int(age)}
            for q in percentiles:
                record[f"p{q:g}"] = float(np.percentile(consumption[:, i], q))
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def rules_from_config(cfg: Mapping[str, Any],
                      best: pd.DataFrame | None = None,
                      limit: int | None = None
                      ) -> Dict[str, "spg.SpendingRule"]:
    """Rebuild rule objects from a `best_spending_rules` table.

    The sweep stores each winning variant's rate; this turns those rows back
    into rule objects so the follow-up analyses evaluate exactly the tuned
    configurations rather than the defaults.
    """
    if best is None:
        return {}
    lookup = {str(entry["key"]) + "|" + str(entry.get("suffix") or ""): entry
              for entry in cfg["spending"]["rules"]}
    out: Dict[str, "spg.SpendingRule"] = {}
    for _, row in best.head(limit if limit is not None else len(best)).iterrows():
        key = str(row["rule"])
        params: Dict[str, Any] = {}
        for handle, entry in lookup.items():
            if handle.startswith(key + "|"):
                candidate = spg.build(key, **dict(entry.get("params", {}) or {}),
                                      **({"rate": float(row["rate"])}
                                         if key in spg.RATE_PARAMETERISED
                                         and row["rate"] == row["rate"] else {}))
                suffix = entry.get("suffix")
                label = candidate.label + (f" ({suffix})" if suffix else "")
                if label == row["variant"]:
                    params = dict(entry.get("params", {}) or {})
                    break
        if key in spg.RATE_PARAMETERISED and row["rate"] == row["rate"]:
            params["rate"] = float(row["rate"])
        out[str(row["variant"])] = spg.build(key, **params)
    return out


def spending_rank_by_risk_aversion(sweep: pd.DataFrame,
                                   gammas: Sequence[float],
                                   rule: str = "constant_real"
                                   ) -> pd.DataFrame:
    """Where one rule family places, at each risk aversion.

    Turns "the flat rule loses at every risk aversion" from an assertion into
    a table, and shows how the gap narrows as risk aversion rises -- which is
    the direction a smoothing rule should benefit from.
    """
    rows: List[Dict[str, Any]] = []
    for gamma in gammas:
        metric = f"cec_gamma{float(gamma):g}"
        if metric not in sweep.columns:
            continue
        best = sweep.loc[sweep.groupby("rule")[metric].idxmax()]
        best = best.sort_values(metric, ascending=False).reset_index(drop=True)
        if rule not in set(best["rule"]):
            continue
        position = int(best.index[best["rule"] == rule][0]) + 1
        target = float(best.loc[best["rule"] == rule, metric].iloc[0])
        winner = best.iloc[0]
        rows.append({
            "risk_aversion": float(gamma),
            "best_rule": str(winner["variant"]),
            "best_cec": float(winner[metric]),
            f"{rule}_cec": target,
            f"{rule}_rank": position,
            "n_families": int(len(best)),
            "gap_pct": (float(winner[metric]) / target - 1.0) * 100.0,
        })
    return pd.DataFrame.from_records(rows)
