"""Accumulation / decumulation lifecycle simulator.

The investor is born into the simulation at ``age_start``, works and saves a
fixed fraction of labour income until ``age_retire``, then draws down
financial wealth alongside a social-security annuity until ``age_death``.
Whatever is left at ``age_death`` is the bequest.

Timing convention (stated once, used everywhere)::

    W[0] = 0
    working year h:     W[h+1] = (W[h] + s * Y[h]) * (1 + Rp[h])
    retirement year h:  W[h+1] = (W[h] - X[h])     * (1 + Rp[h])

Contributions and withdrawals happen at the *start* of the year and are
exposed to that year's portfolio return; ``Rp[h]`` is the return on the
strategy's target weights, which are restored at every rebalancing date.
Everything is denominated in real (CPI-deflated) units, so no nominal
quantity ever appears in the recursion.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .bootstrap import BootstrapPaths

#: Asset ordering used by every weight vector in this module.
ASSETS: Tuple[str, ...] = ("dom_eq", "intl_eq", "bond", "bill")


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LifecycleSpec:
    """Ages, cash-flow rules and preference-free plumbing of one investor."""

    age_start: int = 25
    age_retire: int = 63
    age_death: int = 93
    savings_rate: float = 0.10

    initial_real_income: float = 1.0
    income_b1: float = 0.045
    income_b2: float = -0.0009
    permanent_shock_sd: float = 0.10
    transitory_shock_sd: float = 0.25
    income_shocks_enabled: bool = True

    social_security_enabled: bool = True
    replacement_rate: float = 0.45
    # "progressive" reproduces the US PIA bend-point schedule, which makes
    # the replacement rate fall with career earnings and therefore supplies a
    # genuine real consumption floor.  "flat" applies `replacement_rate`
    # uniformly.
    social_security_formula: str = "progressive"
    pia_bend1: float = 0.21      # x economy-wide average earnings
    pia_bend2: float = 1.28
    pia_rate1: float = 0.90
    pia_rate2: float = 0.32
    pia_rate3: float = 0.15

    retirement_rule: str = "fixed_real_rule"
    rule_rate: float = 0.04
    allow_ruin: bool = True

    def __post_init__(self) -> None:
        if not self.age_start < self.age_retire < self.age_death:
            raise ValueError("ages must satisfy start < retire < death")
        if not 0.0 <= self.savings_rate < 1.0:
            raise ValueError("savings_rate must lie in [0, 1)")
        if self.retirement_rule not in ("fixed_real_rule", "fixed_percentage"):
            raise ValueError(f"unknown retirement_rule {self.retirement_rule!r}")
        if self.social_security_formula not in ("progressive", "flat"):
            raise ValueError(
                f"unknown social_security_formula {self.social_security_formula!r}")

    @property
    def horizon(self) -> int:
        """Number of simulated years (68 for 25 -> 93)."""
        return self.age_death - self.age_start

    @property
    def n_working(self) -> int:
        return self.age_retire - self.age_start

    @property
    def n_retired(self) -> int:
        return self.age_death - self.age_retire

    @property
    def ages(self) -> np.ndarray:
        return np.arange(self.age_start, self.age_death)

    @property
    def retirement_slice(self) -> slice:
        return slice(self.n_working, self.horizon)

    def social_security_benefit(self, career_average: np.ndarray) -> np.ndarray:
        """Real annual retirement benefit from career-average real earnings.

        Under ``"progressive"`` this is the US primary-insurance-amount
        formula: 90% of career-average earnings up to the first bend point,
        32% between the bend points and 15% above the second, with the bend
        points expressed as multiples of economy-wide average earnings.  The
        90% first tranche is what puts a floor under retirement consumption
        for investors who drew a bad sequence of labour-income shocks.
        """
        if not self.social_security_enabled:
            return np.zeros_like(career_average)
        if self.social_security_formula == "flat":
            return self.replacement_rate * career_average
        economy_average = float(self.deterministic_income().mean())
        bend1 = self.pia_bend1 * economy_average
        bend2 = self.pia_bend2 * economy_average
        tranche1 = np.minimum(career_average, bend1)
        tranche2 = np.clip(career_average - bend1, 0.0, bend2 - bend1)
        tranche3 = np.maximum(career_average - bend2, 0.0)
        return (self.pia_rate1 * tranche1
                + self.pia_rate2 * tranche2
                + self.pia_rate3 * tranche3)

    def deterministic_income(self) -> np.ndarray:
        """Hump-shaped real labour-income profile over the working years."""
        t = np.arange(self.n_working, dtype=float)
        log_profile = self.income_b1 * t + self.income_b2 * t ** 2
        return self.initial_real_income * np.exp(log_profile)


@dataclasses.dataclass(frozen=True)
class Strategy:
    """A candidate portfolio, expanded to explicit age-by-asset weights."""

    key: str
    label: str
    weights: np.ndarray  # (H, 4) rows sum to 1

    def __post_init__(self) -> None:
        if self.weights.ndim != 2 or self.weights.shape[1] != len(ASSETS):
            raise ValueError(f"weights must be (horizon, {len(ASSETS)})")
        sums = self.weights.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-8):
            raise ValueError(f"strategy {self.key!r} weights must sum to 1")

    def equity_share(self) -> np.ndarray:
        return self.weights[:, 0] + self.weights[:, 1]


def build_strategies(cfg: Mapping[str, Any], spec: LifecycleSpec
                     ) -> Dict[str, Strategy]:
    """Expand the ``strategies`` config block into age-indexed weights."""
    horizon = spec.horizon
    ages = spec.ages
    out: Dict[str, Strategy] = {}
    for key, entry in cfg["strategies"].items():
        kind = entry.get("type", "constant")
        if kind == "constant":
            row = np.array([float(entry["weights"].get(a, 0.0)) for a in ASSETS])
            weights = np.tile(row, (horizon, 1))
        elif kind == "glide":
            equity = np.interp(ages, np.asarray(entry["glide_ages"], dtype=float),
                               np.asarray(entry["glide_equity"], dtype=float))
            eq_split = entry["equity_split"]
            fi_split = entry["fixed_income_split"]
            eq_total = sum(float(v) for v in eq_split.values())
            fi_total = sum(float(v) for v in fi_split.values())
            weights = np.zeros((horizon, len(ASSETS)))
            weights[:, 0] = equity * float(eq_split.get("dom_eq", 0.0)) / eq_total
            weights[:, 1] = equity * float(eq_split.get("intl_eq", 0.0)) / eq_total
            weights[:, 2] = (1 - equity) * float(fi_split.get("bond", 0.0)) / fi_total
            weights[:, 3] = (1 - equity) * float(fi_split.get("bill", 0.0)) / fi_total
        else:
            raise ValueError(f"unknown strategy type {kind!r} for {key!r}")
        out[key] = Strategy(key=key, label=str(entry["label"]), weights=weights)
    return out


def spec_from_config(cfg: Mapping[str, Any]) -> LifecycleSpec:
    """Read a :class:`LifecycleSpec` out of the ``lifecycle`` config block."""
    life = cfg["lifecycle"]
    income = life["income"]
    ss = life["social_security"]
    ret = life["retirement"]
    return LifecycleSpec(
        age_start=int(life["age_start"]),
        age_retire=int(life["age_retire"]),
        age_death=int(life["age_death"]),
        savings_rate=float(life["savings_rate"]),
        initial_real_income=float(income["initial_real_income"]),
        income_b1=float(income["b1"]),
        income_b2=float(income["b2"]),
        permanent_shock_sd=float(income["permanent_shock_sd"]),
        transitory_shock_sd=float(income["transitory_shock_sd"]),
        income_shocks_enabled=bool(income["shocks_enabled"]),
        social_security_enabled=bool(ss["enabled"]),
        replacement_rate=float(ss["replacement_rate"]),
        social_security_formula=str(ss.get("formula", "progressive")),
        pia_bend1=float(ss.get("pia_bend1", 0.21)),
        pia_bend2=float(ss.get("pia_bend2", 1.28)),
        pia_rate1=float(ss.get("pia_rate1", 0.90)),
        pia_rate2=float(ss.get("pia_rate2", 0.32)),
        pia_rate3=float(ss.get("pia_rate3", 0.15)),
        retirement_rule=str(ret["rule"]),
        rule_rate=float(ret["rule_rate"]),
        allow_ruin=bool(ret["allow_ruin"]),
    )


# ---------------------------------------------------------------------------
# Labour income
# ---------------------------------------------------------------------------
def simulate_income(spec: LifecycleSpec, n_paths: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Real labour income over the working years, shape ``(n_paths, n_working)``.

    The deterministic hump is multiplied by a permanent component (a random
    walk in logs) and an i.i.d. transitory component, both normalised to have
    unit mean so that the profile's *level* is unchanged by adding risk.
    """
    profile = spec.deterministic_income()[None, :]
    if not spec.income_shocks_enabled:
        return np.repeat(profile, n_paths, axis=0)
    n_work = spec.n_working
    perm = rng.normal(-0.5 * spec.permanent_shock_sd ** 2,
                      spec.permanent_shock_sd, size=(n_paths, n_work))
    tran = rng.normal(-0.5 * spec.transitory_shock_sd ** 2,
                      spec.transitory_shock_sd, size=(n_paths, n_work))
    permanent = np.exp(np.cumsum(perm, axis=1))
    transitory = np.exp(tran)
    return profile * permanent * transitory


# ---------------------------------------------------------------------------
# Simulation output
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LifecycleOutcome:
    """Per-path results for one strategy."""

    strategy: str
    label: str
    consumption: np.ndarray            # (N, H) real consumption
    wealth: np.ndarray                 # (N, H + 1) real financial wealth
    portfolio_return: np.ndarray       # (N, H)
    wealth_at_retirement: np.ndarray   # (N,)
    bequest: np.ndarray                # (N,)
    ruin: np.ndarray                   # (N,) bool
    ruin_age: np.ndarray               # (N,) int, age_death where no ruin
    social_security: np.ndarray        # (N,) real annual benefit
    career_average_income: np.ndarray  # (N,) mean real working-life income

    @property
    def n_paths(self) -> int:
        return int(self.consumption.shape[0])

    def concat(self, other: "LifecycleOutcome") -> "LifecycleOutcome":
        if other.strategy != self.strategy:
            raise ValueError("cannot concatenate different strategies")
        joined: Dict[str, Any] = {"strategy": self.strategy, "label": self.label}
        for field in dataclasses.fields(self):
            if field.name in ("strategy", "label"):
                continue
            joined[field.name] = np.concatenate(
                [getattr(self, field.name), getattr(other, field.name)], axis=0)
        return LifecycleOutcome(**joined)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
def portfolio_returns(paths: BootstrapPaths, strategy: Strategy) -> np.ndarray:
    """Rebalanced real portfolio return, shape ``(n_paths, horizon)``."""
    horizon = strategy.weights.shape[0]
    if paths.horizon < horizon:
        raise ValueError(
            f"bootstrap horizon {paths.horizon} is shorter than the "
            f"lifecycle horizon {horizon}")
    stack = np.stack([paths.series(a)[:, :horizon] for a in ASSETS], axis=-1)
    return np.einsum("nha,ha->nh", stack, strategy.weights)


def simulate(
    paths: BootstrapPaths,
    strategy: Strategy,
    spec: LifecycleSpec,
    income: np.ndarray,
) -> LifecycleOutcome:
    """Run one strategy over one chunk of bootstrapped return paths.

    ``income`` is passed in rather than drawn inside so that every strategy
    faces the *same* labour-income realisations on the same path -- without
    that, differences across strategies would be contaminated by income noise.
    """
    n_paths = paths.n_paths
    horizon = spec.horizon
    if income.shape != (n_paths, spec.n_working):
        raise ValueError("income must be (n_paths, n_working)")

    rp = portfolio_returns(paths, strategy)
    wealth = np.zeros((n_paths, horizon + 1))
    consumption = np.zeros((n_paths, horizon))

    # --- accumulation -----------------------------------------------------
    for h in range(spec.n_working):
        contribution = spec.savings_rate * income[:, h]
        consumption[:, h] = income[:, h] - contribution
        wealth[:, h + 1] = (wealth[:, h] + contribution) * (1.0 + rp[:, h])

    wealth_at_retirement = wealth[:, spec.n_working].copy()

    # --- social security --------------------------------------------------
    career_average = income.mean(axis=1)
    benefit = spec.social_security_benefit(career_average)

    # --- decumulation -----------------------------------------------------
    target_withdrawal = spec.rule_rate * wealth_at_retirement
    ruin_age = np.full(n_paths, spec.age_death, dtype=int)
    ruined = np.zeros(n_paths, dtype=bool)

    for h in range(spec.n_working, horizon):
        available = wealth[:, h]
        if spec.retirement_rule == "fixed_real_rule":
            desired = target_withdrawal
        else:  # "fixed_percentage" -- proportional rule, cannot ruin
            desired = spec.rule_rate * available
        withdrawal = np.minimum(desired, np.maximum(available, 0.0))
        newly_ruined = (~ruined) & (withdrawal < desired - 1e-12)
        ruin_age = np.where(newly_ruined, spec.age_start + h, ruin_age)
        ruined |= newly_ruined
        consumption[:, h] = benefit + withdrawal
        wealth[:, h + 1] = np.maximum(available - withdrawal, 0.0) * (1.0 + rp[:, h])

    return LifecycleOutcome(
        strategy=strategy.key,
        label=strategy.label,
        consumption=consumption,
        wealth=wealth,
        portfolio_return=rp,
        wealth_at_retirement=wealth_at_retirement,
        bequest=wealth[:, horizon].copy(),
        ruin=ruined,
        ruin_age=ruin_age,
        social_security=benefit,
        career_average_income=income.mean(axis=1),
    )


def simulate_all(
    paths: BootstrapPaths,
    strategies: Mapping[str, Strategy],
    spec: LifecycleSpec,
    income: np.ndarray,
) -> Dict[str, LifecycleOutcome]:
    """Run every strategy on the same chunk of paths and income draws."""
    return {key: simulate(paths, strat, spec, income)
            for key, strat in strategies.items()}


# ---------------------------------------------------------------------------
# Chunked driver
# ---------------------------------------------------------------------------
def run_chunked(
    sampler: Any,
    strategies: Mapping[str, Strategy],
    spec: LifecycleSpec,
    n_paths: int,
    chunk_size: int,
    income_seed: int = 12345,
) -> Dict[str, LifecycleOutcome]:
    """Stream ``n_paths`` lifetimes through the bootstrap and the simulator.

    Memory scales with ``chunk_size``, not ``n_paths``, so 100k+ lifetimes fit
    comfortably; the per-path outcomes are concatenated as they arrive.
    """
    income_root = np.random.SeedSequence(income_seed)
    n_chunks = int(np.ceil(n_paths / chunk_size))
    income_children = iter(income_root.spawn(n_chunks))
    results: Dict[str, LifecycleOutcome] = {}
    for chunk in sampler.chunks(n_paths, chunk_size):
        rng = np.random.default_rng(next(income_children))
        income = simulate_income(spec, chunk.n_paths, rng)
        outcomes = simulate_all(chunk, strategies, spec, income)
        for key, outcome in outcomes.items():
            results[key] = outcome if key not in results \
                else results[key].concat(outcome)
    return results


def glide_path_table(strategies: Mapping[str, Strategy], spec: LifecycleSpec
                     ) -> "Any":
    """Age-by-strategy equity share, for docs/03 and the glide-path figure."""
    import pandas as pd

    frame = pd.DataFrame({"age": spec.ages})
    for key, strat in strategies.items():
        frame[key] = strat.equity_share()
    return frame.set_index("age")
