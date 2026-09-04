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

from . import spending as sp
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
    # Correlation between the permanent labour-income innovation and the
    # domestic equity return of the same year.  Zero -- the default, and what
    # every headline result uses -- makes human capital an independent asset.
    # It is the textbook reason to hold *less* of your own market: if your
    # salary is already a claim on it, buying the index doubles the bet.
    # Sweeping it is `docs/23`.  Because it is a correlation and not a
    # loading, raising it re-labels which part of income risk is systematic
    # without changing how much income risk there is.
    income_return_correlation: float = 0.0
    # Correlation between the same permanent innovation and the *international*
    # return.  ``None`` -- the default -- means "unspecified": the innovation is
    # rotated toward the domestic market alone, and whatever correlation with
    # the foreign market follows from the two markets' own correlation is left
    # to fall out.  That is what every result in `docs/23` reports.  Setting a
    # number instead pins *both* correlations at once, which is the harder
    # question: if a pay cheque is a claim on world equity rather than on the
    # home market specifically, the reason to tilt away from home weakens.  The
    # two-regressor rotation that does it is in :func:`simulate_income`.
    income_intl_correlation: float | None = None
    # Proportional one-way cost of trading, charged on the value turned over at
    # each annual rebalance.  Zero -- the default -- leaves every other result
    # in the project bit-identical.  It is the price of the schedule, and the
    # schedules solved in `docs/12` never paid it while they were being solved.
    trading_cost: float = 0.0
    # A compulsory employer contribution to a separate retirement fund, on
    # top of whatever the worker saves themselves.  This is Australia's
    # Superannuation Guarantee: 12% of ordinary time earnings since 1 July
    # 2025, taxed at 15% on the way in, and invested in the member's chosen
    # strategy.  Zero -- the default -- leaves every other result untouched.
    #
    # It is *additional* to `savings_rate` rather than a substitute for it,
    # and it does not reduce take-home pay: the statutory incidence is on the
    # employer.  Working-life consumption is therefore unchanged by it, which
    # matters less than it sounds here because the utility window is
    # retirement only -- but it is an assumption, and `docs/25` says so.
    super_guarantee_rate: float = 0.0
    super_contributions_tax: float = 0.15
    # Floor on labour income, as a multiple of economy-wide average earnings,
    # standing in for unemployment insurance and in-work benefits.  Default 0
    # (no floor), which leaves every existing result unchanged.
    #
    # It matters in exactly one place.  Retirement carries a real consumption
    # floor through the progressive social-security schedule; working life, at
    # zero, carries none.  Wherever compared strategies share the same
    # working-life consumption that asymmetry cancels, so it is invisible.  It
    # does *not* cancel when the comparison is over *when to retire*, where it
    # would otherwise reward retiring early purely for reaching the floor
    # sooner.  See docs/09.
    working_income_floor: float = 0.0

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

    # "means_tested" reproduces the Australian Age Pension: a flat maximum
    # rate, withdrawn against *assessable assets* rather than against career
    # earnings, and therefore recomputed every retirement year as the portfolio
    # draws down.  The three parameters below are the rate and the free area as
    # multiples of economy-wide average earnings, and the annual pension lost
    # per unit of assets above the free area.  See `docs/25`.
    pension_full_rate: float = 0.293
    pension_free_area: float = 3.01
    pension_taper: float = 0.078

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
        if self.social_security_formula not in ("progressive", "flat",
                                                "means_tested"):
            raise ValueError(
                f"unknown social_security_formula {self.social_security_formula!r}")
        if not 0.0 <= self.pre_eligibility_benefit_share <= 1.0:
            raise ValueError(
                "pre_eligibility_benefit_share must lie in [0, 1]")
        if (self.benefit_start_age is not None
                and not self.age_start <= self.benefit_start_age
                <= self.age_death):
            raise ValueError(
                "benefit_start_age must lie between age_start and age_death")
        if not -1.0 <= self.income_return_correlation <= 1.0:
            raise ValueError("income_return_correlation must lie in [-1, 1]")
        if self.income_intl_correlation is not None and not (
                -1.0 <= self.income_intl_correlation <= 1.0):
            raise ValueError("income_intl_correlation must lie in [-1, 1]")
        if self.trading_cost < 0.0:
            raise ValueError("trading_cost must be non-negative")
        if not 0.0 <= self.super_guarantee_rate < 1.0:
            raise ValueError("super_guarantee_rate must lie in [0, 1)")
        if not 0.0 <= self.super_contributions_tax < 1.0:
            raise ValueError("super_contributions_tax must lie in [0, 1)")
        if self.pension_taper < 0.0 or self.pension_free_area < 0.0:
            raise ValueError("pension taper and free area must be non-negative")

    @property
    def super_net_rate(self) -> float:
        """The share of income that actually reaches the super fund.

        Twelve per cent of ordinary time earnings, less the fifteen per cent
        tax on concessional contributions, leaves 10.2% -- which happens to
        sit almost exactly on this paper's own 10% savings rate, a coincidence
        worth noticing and not worth leaning on.
        """
        return self.super_guarantee_rate * (1.0 - self.super_contributions_tax)

    @property
    def total_contribution_rate(self) -> float:
        """Everything going into the portfolio each working year."""
        return self.savings_rate + self.super_net_rate

    @property
    def super_share_of_contributions(self) -> float:
        """How much of the pot the compulsory fund supplies, exactly.

        Both streams are the same fraction of the same income in the same
        years and earn the same returns, so their ratio is constant through
        the accumulation phase and this is the share at every age -- no
        separate balance needs tracking to report it.
        """
        total = self.total_contribution_rate
        return self.super_net_rate / total if total > 0.0 else 0.0

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

    #: Multiplier on the retirement benefit, for studies that make the
    #: retirement date a decision variable.  Every other section fixes that
    #: date, so the benefit's start date is common to all strategies and the
    #: factor cancels; ``1.0`` is therefore the default and leaves every
    #: existing result bit-identical.  Section #leisure sets it, because once
    #: the date can move a benefit that starts whenever work stops -- with no
    #: reduction for the longer claiming period -- pays an investor who
    #: retires at fifty-five a full pension for thirty-eight years and makes
    #: early retirement look free.
    ss_claim_factor: float = 1.0
    #: Age from which the retirement benefit is actually paid.  ``None`` --
    #: the default -- means "whenever work stops", which is what every other
    #: section assumes and what leaves those results bit-identical.
    #:
    #: It is not a detail once the retirement date can move.  Australia's Age
    #: Pension is payable at 67 however early somebody stopped working, so an
    #: Australian retiring at 55 funds twelve years entirely from their own
    #: portfolio before any pension arrives.  A model that starts the benefit
    #: on the day work ends cannot see that bridge, and would price early
    #: retirement as though the state stepped in immediately.
    benefit_start_age: int | None = None
    #: Share of the means-tested payment available *before* the eligibility
    #: age, standing in for the working-age safety net.  Zero -- the default
    #: -- means somebody who exhausts their portfolio during a bridge to the
    #: pension receives literally nothing, which no country arranges and
    #: which a CRRA aggregator punishes without limit: one year at the
    #: consumption floor is enough to decide a certainty equivalent on its
    #: own.  Australia pays JobSeeker to a retiree who has run out before 67,
    #: at appreciably less than the Age Pension but not at nothing.
    pre_eligibility_benefit_share: float = 0.0

    @property
    def benefit_start_index(self) -> int:
        """Simulated year from which the benefit is paid."""
        if self.benefit_start_age is None:
            return int(self.n_working)
        return int(np.clip(int(self.benefit_start_age) - int(self.age_start),
                           0, int(self.horizon)))

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
            return (self.ss_claim_factor * self.replacement_rate
                    * career_average)
        if self.social_security_formula == "means_tested":
            # A flat pension does not depend on career earnings at all, so the
            # career-average signature has nothing to say about it.  What comes
            # back here is the *maximum* rate, which is what a retiree with no
            # assessable assets receives; :func:`simulate` overrides it year by
            # year with :meth:`means_tested_benefit` once assets are known.
            return np.full_like(career_average, self.pension_full_rate
                                * float(self.deterministic_income().mean()))
        economy_average = float(self.deterministic_income().mean())
        bend1 = self.pia_bend1 * economy_average
        bend2 = self.pia_bend2 * economy_average
        tranche1 = np.minimum(career_average, bend1)
        tranche2 = np.clip(career_average - bend1, 0.0, bend2 - bend1)
        tranche3 = np.maximum(career_average - bend2, 0.0)
        return self.ss_claim_factor * (self.pia_rate1 * tranche1
                                       + self.pia_rate2 * tranche2
                                       + self.pia_rate3 * tranche3)

    def means_tested_benefit(self, assets: np.ndarray) -> np.ndarray:
        """The Australian Age Pension: a flat rate, tapered against assets.

        The maximum rate is paid while assessable assets sit below the free
        area, then withdrawn at :attr:`pension_taper` per unit of assets above
        it, reaching zero at the cut-off.  Both thresholds and the rate are
        held as multiples of economy-wide average earnings so that the schedule
        travels between the panel's currencies unchanged.

        The economics that matters here is the taper.  Every extra dollar a
        retiree holds inside the tapered band costs them
        :attr:`pension_taper` of pension a year, which is an implicit wealth
        tax at a rate no financial asset in this paper earns reliably -- and it
        runs the other way too, so a portfolio that falls is met by a pension
        that rises.  A means test is therefore a floor and a ceiling at once,
        which is why it has to be applied to assets *as they are*, year by
        year, rather than once at retirement.
        """
        if not self.social_security_enabled:
            return np.zeros_like(np.asarray(assets, dtype=float))
        economy_average = float(self.deterministic_income().mean())
        full = self.pension_full_rate * economy_average
        excess = np.maximum(np.asarray(assets, dtype=float)
                            - self.pension_free_area * economy_average, 0.0)
        return np.clip(full - self.pension_taper * excess, 0.0, full)

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
        income_return_correlation=float(
            income.get("return_correlation", 0.0)),
        income_intl_correlation=(
            None if income.get("intl_correlation") is None
            else float(income["intl_correlation"])),
        trading_cost=float(life.get("trading_cost", 0.0)),
        working_income_floor=float(income.get("working_income_floor", 0.0)),
        social_security_enabled=bool(ss["enabled"]),
        replacement_rate=float(ss["replacement_rate"]),
        super_guarantee_rate=float(life.get("super_guarantee_rate", 0.0)),
        super_contributions_tax=float(
            life.get("super_contributions_tax", 0.15)),
        social_security_formula=str(ss.get("formula", "progressive")),
        pia_bend1=float(ss.get("pia_bend1", 0.21)),
        pia_bend2=float(ss.get("pia_bend2", 1.28)),
        pia_rate1=float(ss.get("pia_rate1", 0.90)),
        pia_rate2=float(ss.get("pia_rate2", 0.32)),
        pia_rate3=float(ss.get("pia_rate3", 0.15)),
        pension_full_rate=float(ss.get("pension_full_rate", 0.293)),
        pension_free_area=float(ss.get("pension_free_area", 3.01)),
        pension_taper=float(ss.get("pension_taper", 0.078)),
        retirement_rule=str(ret["rule"]),
        rule_rate=float(ret["rule_rate"]),
        allow_ruin=bool(ret["allow_ruin"]),
    )


# ---------------------------------------------------------------------------
# Labour income
# ---------------------------------------------------------------------------
def draw_income_shocks(n_paths: int, n_years: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Standard-normal draws for the permanent and transitory components.

    Drawing these once and reusing them across configurations gives
    :mod:`src.sensitivity` common random numbers: when a sweep changes the
    retirement age or the withdrawal rate, the difference in outcomes is the
    parameter's effect rather than Monte Carlo noise.
    """
    return (rng.standard_normal((n_paths, n_years)),
            rng.standard_normal((n_paths, n_years)))


def _standardise(matrix: np.ndarray) -> np.ndarray:
    """A matrix rescaled to zero mean and unit variance over all its entries.

    Standardising the *realised* draws rather than using a population moment
    keeps the requested correlation exact for the sample actually simulated,
    which is what makes a sweep over it interpretable.
    """
    values = np.asarray(matrix, dtype=float)
    sd = float(values.std())
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(values)
    return (values - float(values.mean())) / sd


def _rotation(market: np.ndarray, foreign: np.ndarray,
              rho_dom: float, rho_intl: float
              ) -> Tuple[float, float, float]:
    """Loadings that hit two target correlations at once, at unit variance.

    We want an innovation ``z = a*d + b*f + s*e`` whose correlation with the
    domestic series ``d`` is ``rho_dom`` and with the foreign series ``f`` is
    ``rho_intl``, holding ``var(z) = 1`` so that the sweep changes *which*
    risks a career carries and not *how much*.  With ``c = corr(d, f)`` the
    first two conditions are a 2x2 system,

    ``a + b*c = rho_dom`` and ``a*c + b = rho_intl``,

    whose solution is ``a = (rho_dom - c*rho_intl)/(1 - c^2)`` and
    ``b = (rho_intl - c*rho_dom)/(1 - c^2)``.  The variance explained by the
    two markets then collapses to ``a*rho_dom + b*rho_intl``, and the residual
    weight is the square root of what is left.

    Note what this does *not* reduce to.  Passing ``rho_intl = 0`` here is a
    stronger statement than leaving it unspecified: it demands a pay cheque
    uncorrelated with the foreign market, which -- because the two markets move
    together -- requires loading *negatively* on the foreign one.  Leaving it
    unspecified instead takes the one-regressor rotation and accepts the
    induced foreign correlation of ``c * rho_dom``.  The two readings bracket
    the honest answer, which is why `docs/23` reports both.
    """
    c = float(np.corrcoef(market.ravel(), foreign.ravel())[0, 1])
    if not np.isfinite(c) or abs(c) >= 1.0:
        raise ValueError("the two markets are collinear; no rotation exists")
    a = (rho_dom - c * rho_intl) / (1.0 - c ** 2)
    b = (rho_intl - c * rho_dom) / (1.0 - c ** 2)
    explained = a * rho_dom + b * rho_intl
    if explained > 1.0 + 1e-12:
        raise ValueError(
            f"correlations ({rho_dom:.3f}, {rho_intl:.3f}) are infeasible "
            f"against markets correlated {c:.3f}: they would need "
            f"{explained:.3f} of the innovation's variance")
    return a, b, float(np.sqrt(max(1.0 - explained, 0.0)))


def simulate_income(spec: LifecycleSpec, n_paths: int,
                    rng: np.random.Generator | None = None,
                    shocks: Tuple[np.ndarray, np.ndarray] | None = None,
                    dom_eq: np.ndarray | None = None,
                    intl_eq: np.ndarray | None = None,
                    ) -> np.ndarray:
    """Real labour income over the working years, shape ``(n_paths, n_working)``.

    The deterministic hump is multiplied by a permanent component (a random
    walk in logs) and an i.i.d. transitory component, both normalised to have
    unit mean so that the profile's *level* is unchanged by adding risk.

    ``shocks`` optionally supplies pre-drawn standard normals from
    :func:`draw_income_shocks`; they are sliced to the spec's working years,
    so a shorter career is a prefix of a longer one rather than an
    independent draw.

    ``dom_eq`` supplies the domestic equity returns of the same paths. It is
    used only when ``spec.income_return_correlation`` is non-zero, where the
    permanent innovation is rotated toward the market:
    ``rho * u + sqrt(1 - rho^2) * z``. Because that rotation preserves unit
    variance, raising ``rho`` changes how much of a career's risk is
    *systematic* without changing how much risk it carries -- so a sweep over
    it isolates the correlation and does not confound it with a level of
    income volatility. At ``rho = 0`` the arithmetic is untouched and every
    other result in the project is bit-identical.
    """
    profile = spec.deterministic_income()[None, :]
    floor = spec.working_income_floor * float(
        spec.deterministic_income().mean())
    if not spec.income_shocks_enabled:
        return np.maximum(np.repeat(profile, n_paths, axis=0), floor)
    n_work = spec.n_working
    if shocks is None:
        if rng is None:
            raise ValueError("simulate_income needs either rng or shocks")
        z_perm = rng.standard_normal((n_paths, n_work))
        z_tran = rng.standard_normal((n_paths, n_work))
    else:
        z_perm, z_tran = (arr[:n_paths, :n_work] for arr in shocks)
        if z_perm.shape != (n_paths, n_work):
            raise ValueError(
                f"pre-drawn shocks are too small: need ({n_paths}, {n_work}), "
                f"got {z_perm.shape}")
    rho = float(spec.income_return_correlation)
    rho_i = spec.income_intl_correlation
    if rho or rho_i is not None:
        if dom_eq is None:
            raise ValueError(
                "income_return_correlation is set but no domestic equity "
                "returns were supplied to correlate the shocks with")
        market = _standardise(np.asarray(dom_eq)[:n_paths, :n_work])
        if market.shape != (n_paths, n_work):
            raise ValueError(
                f"dom_eq must cover ({n_paths}, {n_work}), got {market.shape}")
        if rho_i is None:
            z_perm = rho * market + np.sqrt(1.0 - rho ** 2) * z_perm
        else:
            if intl_eq is None:
                raise ValueError(
                    "income_intl_correlation is set but no international "
                    "equity returns were supplied")
            foreign = _standardise(np.asarray(intl_eq)[:n_paths, :n_work])
            if foreign.shape != (n_paths, n_work):
                raise ValueError(
                    f"intl_eq must cover ({n_paths}, {n_work}), "
                    f"got {foreign.shape}")
            a, b, resid = _rotation(market, foreign, rho, float(rho_i))
            z_perm = a * market + b * foreign + resid * z_perm
    perm = -0.5 * spec.permanent_shock_sd ** 2 + spec.permanent_shock_sd * z_perm
    tran = -0.5 * spec.transitory_shock_sd ** 2 + spec.transitory_shock_sd * z_tran
    permanent = np.exp(np.cumsum(perm, axis=1))
    transitory = np.exp(tran)
    return np.maximum(profile * permanent * transitory, floor)


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


def portfolio_turnover(paths: BootstrapPaths, strategy: Strategy) -> np.ndarray:
    """One-way turnover at each annual rebalance, shape ``(n_paths, horizon)``.

    Two things make a portfolio trade, and only one of them is a choice.  A
    constant-weight portfolio must still trade every year, because the assets
    inside it did not return the same thing and the weights have drifted; that
    is *drift* turnover, and every strategy in this paper pays it.  A schedule
    that deliberately moves its target -- a glide path, or the age-by-asset
    surface solved in `docs/12` -- pays that and then pays again for the move
    it chose.  The difference between the two is the cost the optimiser never
    saw, which is the whole point of measuring this.

    Turnover is reported one-way (half the sum of absolute weight changes, so a
    full switch from one asset to another counts once) and is charged at the
    *start* of each year against the weights the previous year drifted to.  The
    first year is free: the opening contribution buys the target outright.
    Contributions in later years are assumed to be invested pro-rata rather
    than steered toward the underweight asset, which overstates turnover during
    accumulation -- a conservative choice, and one that goes away in retirement
    where the flows run the other way.
    """
    horizon = strategy.weights.shape[0]
    weights = strategy.weights
    stack = np.stack([paths.series(a)[:, :horizon] for a in ASSETS], axis=-1)
    out = np.zeros((paths.n_paths, horizon))
    for h in range(1, horizon):
        grown = weights[h - 1] * (1.0 + stack[:, h - 1, :])
        total = grown.sum(axis=1, keepdims=True)
        drifted = np.divide(grown, total, out=np.zeros_like(grown),
                            where=total != 0.0)
        out[:, h] = 0.5 * np.abs(weights[h] - drifted).sum(axis=1)
    return out


def net_portfolio_returns(paths: BootstrapPaths, strategy: Strategy,
                          cost: float) -> Tuple[np.ndarray, np.ndarray]:
    """Portfolio returns after trading costs, and the turnover that caused them.

    The cost is charged on the value traded at the start of the year, before
    that year's return compounds on what is left: ``(1 - k*T) * (1 + r) - 1``.
    At ``cost = 0`` this is :func:`portfolio_returns` exactly.
    """
    gross = portfolio_returns(paths, strategy)
    if not cost:
        return gross, np.zeros_like(gross)
    turnover = portfolio_turnover(paths, strategy)
    return (1.0 - cost * turnover) * (1.0 + gross) - 1.0, turnover


def simulate(
    paths: BootstrapPaths,
    strategy: Strategy,
    spec: LifecycleSpec,
    income: np.ndarray,
    spending: "sp.SpendingRule | None" = None,
) -> LifecycleOutcome:
    """Run one strategy over one chunk of bootstrapped return paths.

    ``income`` is passed in rather than drawn inside so that every strategy
    faces the *same* labour-income realisations on the same path -- without
    that, differences across strategies would be contaminated by income noise.

    ``spending`` selects the retirement withdrawal policy.  When omitted it
    is built from ``spec.retirement_rule`` and ``spec.rule_rate``, which is
    what the headline pipeline uses; ``docs/06`` compares the alternatives.
    """
    n_paths = paths.n_paths
    horizon = spec.horizon
    if income.shape != (n_paths, spec.n_working):
        raise ValueError("income must be (n_paths, n_working)")

    rp, _turnover = net_portfolio_returns(paths, strategy, spec.trading_cost)
    wealth = np.zeros((n_paths, horizon + 1))
    consumption = np.zeros((n_paths, horizon))

    # --- accumulation -----------------------------------------------------
    # Two contribution streams, not one. The voluntary share comes out of
    # take-home pay and so reduces consumption; the compulsory employer
    # contribution does not, which is why a Superannuation Guarantee is a
    # transfer into the portfolio rather than a reallocation within it.
    employer_rate = spec.super_net_rate
    for h in range(spec.n_working):
        voluntary = spec.savings_rate * income[:, h]
        employer = employer_rate * income[:, h]
        consumption[:, h] = income[:, h] - voluntary
        wealth[:, h + 1] = ((wealth[:, h] + voluntary + employer)
                            * (1.0 + rp[:, h]))

    wealth_at_retirement = wealth[:, spec.n_working].copy()

    # --- social security --------------------------------------------------
    career_average = income.mean(axis=1)
    benefit = spec.social_security_benefit(career_average)
    # A means test is assessed on assets as they stand, so it cannot be settled
    # once at retirement the way an earnings-related benefit can.  Under it the
    # benefit becomes a path through the decumulation loop, and the per-path
    # figure reported afterwards is its average over the retirement years.
    means_tested = spec.social_security_formula == "means_tested"
    benefit_paid = np.zeros((n_paths, spec.n_retired)) if means_tested else None
    # Nothing is paid before the eligibility age, which need not be the
    # retirement date: see `LifecycleSpec.benefit_start_age`.
    benefit_from = spec.benefit_start_index
    entitlement = benefit
    nothing = np.zeros_like(benefit)

    # --- decumulation -----------------------------------------------------
    rule = spending or sp.from_spec(spec.retirement_rule, spec.rule_rate)
    inflation = paths.inflation[:, :horizon]
    initial_withdrawal = rule.initial_withdrawal(
        wealth_at_retirement, spec.n_retired, spec.age_retire)
    prev_withdrawal = initial_withdrawal
    # Feedback rules condition on the year just gone; entering retirement,
    # that is the final working year.
    last_return = rp[:, spec.n_working - 1]
    last_inflation = inflation[:, spec.n_working - 1]

    ruin_age = np.full(n_paths, spec.age_death, dtype=int)
    ruined = np.zeros(n_paths, dtype=bool)

    for h in range(spec.n_working, horizon):
        available = wealth[:, h]
        state = sp.SpendingState(
            year=h - spec.n_working,
            age=spec.age_start + h,
            years_remaining=horizon - h,
            wealth=available,
            prev_withdrawal=prev_withdrawal,
            initial_withdrawal=initial_withdrawal,
            wealth_at_retirement=wealth_at_retirement,
            last_return=last_return,
            last_inflation=last_inflation,
        )
        desired = np.maximum(rule.desired(state), 0.0)
        withdrawal = np.minimum(desired, np.maximum(available, 0.0))
        eligible = h >= benefit_from
        share = 1.0 if eligible else float(spec.pre_eligibility_benefit_share)
        if means_tested:
            benefit = (share * spec.means_tested_benefit(available) if share
                       else nothing)
            benefit_paid[:, h - spec.n_working] = benefit
        else:
            benefit = share * entitlement if share else nothing
        consumption[:, h] = benefit + withdrawal
        wealth[:, h + 1] = np.maximum(available - withdrawal, 0.0) * (1.0 + rp[:, h])

        # Ruin is running out of money with retirement years still to fund.
        # Testing "could not afford the desired withdrawal" instead would
        # misclassify the horizon-based rules, which deliberately spend the
        # last of the portfolio in the final year: an amortisation rule asks
        # for slightly more than the remaining balance at n = 1, and being
        # unable to spend more than everything is not ruin.
        exhausted = (wealth[:, h + 1] <= 0.0) & (h + 1 < horizon)
        newly_ruined = (~ruined) & exhausted
        ruin_age = np.where(newly_ruined, spec.age_start + h + 1, ruin_age)
        ruined |= newly_ruined
        prev_withdrawal = withdrawal
        last_return = rp[:, h]
        last_inflation = inflation[:, h]

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
        social_security=(benefit_paid.mean(axis=1) if means_tested
                         else benefit),
        career_average_income=income.mean(axis=1),
    )


def simulate_all(
    paths: BootstrapPaths,
    strategies: Mapping[str, Strategy],
    spec: LifecycleSpec,
    income: np.ndarray,
    spending: "sp.SpendingRule | None" = None,
) -> Dict[str, LifecycleOutcome]:
    """Run every strategy on the same chunk of paths and income draws."""
    return {key: simulate(paths, strat, spec, income, spending)
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
        income = simulate_income(spec, chunk.n_paths, rng,
                                 dom_eq=chunk.dom_eq, intl_eq=chunk.intl_eq)
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
