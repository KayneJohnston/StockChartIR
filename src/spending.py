"""Retirement spending rules.

The withdrawal policy is a bigger lever on retirement outcomes than the asset
allocation, and it is the part of the problem where practice has the widest
spread of competing recommendations.  This module implements the main
families as interchangeable objects so they can be compared on one footing.

Every rule is evaluated **in real terms**, because the whole engine is.  Rules
whose textbook statement is nominal -- Guyton-Klinger's inflation rule, for
instance -- are translated explicitly and the translation is documented on
the class.

A rule sees, each retirement year, the state in :class:`SpendingState` and
returns the *desired* withdrawal.  The simulator caps that at available
wealth, so a rule never has to defend itself against overdrawing.

The families implemented here:

``constant_real``      the "4% rule": a fixed real amount set at retirement
``constant_percent``   a fixed share of the *current* portfolio each year
``guyton_klinger``     guardrails: cut after bad runs, raise after good ones
``vanguard_dynamic``   percent-of-portfolio, with year-on-year ceiling/floor
``endowment``          exponentially smoothed percent-of-portfolio
``life_expectancy``    portfolio divided by remaining planning years (RMD)
``gompertz``           the same, but with an actuarial life expectancy
``amortisation``       annuity payment on the portfolio at an assumed return
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# State handed to a rule each year
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SpendingState:
    """Everything a spending rule may condition on, all ``(n_paths,)`` arrays.

    ``year`` is the 0-based retirement year and ``age`` the investor's age at
    the start of it.  ``years_remaining`` counts the withdrawals still to
    come, including this one.
    """

    year: int
    age: int
    years_remaining: int
    wealth: np.ndarray
    prev_withdrawal: np.ndarray
    initial_withdrawal: np.ndarray
    wealth_at_retirement: np.ndarray
    last_return: np.ndarray
    last_inflation: np.ndarray
    #: Average real income over the final working years, ``(n_paths,)``. Only
    #: a rule that targets a standard of living rather than a portfolio needs
    #: it; the rest ignore it.
    pre_retirement_income: np.ndarray | None = None
    #: The pension payable this year, ``(n_paths,)``, before any withdrawal.
    #: A rule that targets total consumption has to net it off; a rule that
    #: targets the withdrawal itself does not.
    benefit: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class SpendingRule(abc.ABC):
    """A retirement withdrawal policy."""

    key: str = "rule"
    label: str = "Rule"

    @abc.abstractmethod
    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        """The first retirement year's withdrawal."""

    @abc.abstractmethod
    def desired(self, state: SpendingState) -> np.ndarray:
        """The withdrawal the rule wants, before the wealth cap is applied."""

    def describe(self) -> Dict[str, Any]:
        """Parameters, for the documentation tables."""
        fields = {f.name: getattr(self, f.name)
                  for f in dataclasses.fields(self)} \
            if dataclasses.is_dataclass(self) else {}
        return {"key": self.key, "label": self.label, **fields}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.key!r})"


# ---------------------------------------------------------------------------
# Fixed-amount and fixed-share families
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ConstantRealRule(SpendingRule):
    """Withdraw a fixed *real* amount, set as a share of wealth at retirement.

    This is the Bengen "4% rule" and the SAFEMAX literature's object of
    study.  Spending is perfectly smooth until the money runs out, which is
    the trade it makes: zero consumption volatility, all the risk pushed into
    a single catastrophic event.
    """

    rate: float = 0.04
    key: str = dataclasses.field(default="constant_real", init=False)
    label: str = dataclasses.field(
        default="Constant real (4%-rule style)", init=False)

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return self.rate * wealth_at_retirement

    def desired(self, state: SpendingState) -> np.ndarray:
        return state.initial_withdrawal


@dataclasses.dataclass(frozen=True)
class ConstantPercentRule(SpendingRule):
    """Withdraw a fixed percentage of the *current* portfolio each year.

    Cannot deplete the portfolio -- a constant share of a shrinking balance
    is always affordable -- so ruin probability is zero by construction.  The
    risk shows up instead as consumption volatility, which is what the
    certainty equivalent is there to price.
    """

    rate: float = 0.05
    key: str = dataclasses.field(default="constant_percent", init=False)
    label: str = dataclasses.field(
        default="Constant % of portfolio", init=False)

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return self.rate * wealth_at_retirement

    def desired(self, state: SpendingState) -> np.ndarray:
        return self.rate * state.wealth


# ---------------------------------------------------------------------------
# Feedback rules
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GuytonKlingerRule(SpendingRule):
    """Guyton-Klinger decision rules with capital-preservation guardrails.

    Real spending is held flat unless the *current* withdrawal rate --
    spending divided by current wealth -- drifts outside a band around the
    rate set at retirement:

    * above ``1 + guardrail`` times the initial rate, cut spending by
      ``adjustment`` (the capital-preservation rule);
    * below ``1 - guardrail`` times it, raise spending by ``adjustment``
      (the prosperity rule).

    The capital-preservation cut is suspended in the last
    ``preservation_cutoff_years`` of the plan, as in the original, on the
    grounds that there is no longer a portfolio to preserve.

    **Translation to real terms.**  Guyton-Klinger's inflation rule freezes
    the *nominal* withdrawal after a year of negative portfolio returns.  In
    a real-terms engine that is a real cut equal to the realised inflation
    rate, which is how it is implemented here -- the rule reaches into the
    bootstrap's inflation draw rather than being silently dropped.
    """

    rate: float = 0.05
    guardrail: float = 0.20
    adjustment: float = 0.10
    apply_inflation_rule: bool = True
    preservation_cutoff_years: int = 15
    key: str = dataclasses.field(default="guyton_klinger", init=False)
    label: str = dataclasses.field(
        default="Guyton-Klinger guardrails", init=False)

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return self.rate * wealth_at_retirement

    def desired(self, state: SpendingState) -> np.ndarray:
        base = state.prev_withdrawal.astype(float, copy=True)

        if self.apply_inflation_rule:
            # Freezing the nominal amount after a down year is a real cut of
            # exactly the realised inflation rate.
            deflator = 1.0 + np.maximum(state.last_inflation, 0.0)
            base = np.where(state.last_return < 0.0, base / deflator, base)

        with np.errstate(invalid="ignore", divide="ignore"):
            current_rate = np.where(state.wealth > 0.0,
                                    base / state.wealth, np.inf)

        upper = self.rate * (1.0 + self.guardrail)
        lower = self.rate * (1.0 - self.guardrail)

        preserve = (current_rate > upper) & (
            state.years_remaining > self.preservation_cutoff_years)
        prosper = current_rate < lower

        base = np.where(preserve, base * (1.0 - self.adjustment), base)
        base = np.where(prosper, base * (1.0 + self.adjustment), base)
        return base


@dataclasses.dataclass(frozen=True)
class VanguardDynamicRule(SpendingRule):
    """Percent-of-portfolio spending with a year-on-year ceiling and floor.

    Vanguard's "dynamic spending" rule: target a fixed share of the current
    balance, but never let real spending rise more than ``ceiling`` or fall
    more than ``floor`` against last year.  It is a deliberate compromise
    between the smooth-but-brittle constant-real rule and the
    unbreakable-but-volatile constant-percent rule.
    """

    rate: float = 0.05
    ceiling: float = 0.05
    floor: float = -0.025
    key: str = dataclasses.field(default="vanguard_dynamic", init=False)
    label: str = dataclasses.field(
        default="Vanguard dynamic (ceiling/floor)", init=False)

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return self.rate * wealth_at_retirement

    def desired(self, state: SpendingState) -> np.ndarray:
        target = self.rate * state.wealth
        upper = state.prev_withdrawal * (1.0 + self.ceiling)
        lower = state.prev_withdrawal * (1.0 + self.floor)
        return np.clip(target, lower, upper)


@dataclasses.dataclass(frozen=True)
class EndowmentRule(SpendingRule):
    """Exponentially smoothed percent-of-portfolio, as university endowments use.

    ``smoothing`` is the weight on last year's spending; the remainder goes
    on a fresh percent-of-portfolio target.  At ``smoothing = 0`` this is the
    constant-percent rule; at ``smoothing = 1`` it is constant real spending.
    Everything interesting happens in between, which makes it a useful bridge
    between the two extremes.
    """

    rate: float = 0.05
    smoothing: float = 0.7
    key: str = dataclasses.field(default="endowment", init=False)
    label: str = dataclasses.field(
        default="Endowment smoothing", init=False)

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return self.rate * wealth_at_retirement

    def desired(self, state: SpendingState) -> np.ndarray:
        return (self.smoothing * state.prev_withdrawal
                + (1.0 - self.smoothing) * self.rate * state.wealth)


# ---------------------------------------------------------------------------
# Horizon-based rules
# ---------------------------------------------------------------------------
def gompertz_life_expectancy(age: float, modal_age: float = 88.0,
                             dispersion: float = 10.0,
                             max_age: float = 120.0,
                             steps: int = 2000) -> float:
    """Remaining life expectancy under a Gompertz mortality law.

    Survival from age ``x`` for ``t`` more years is

    ``S(t | x) = exp( exp((x - m) / b) * (1 - exp(t / b)) )``

    and ``e(x)`` is its integral.  ``m = 88``, ``b = 10`` is the calibration
    commonly used for a healthy retiree in the actuarial-spending literature
    (Milevsky).  This is a *model* of mortality, not a life table lifted from
    data, and it is used here only as a planning divisor -- the simulated
    investor's actual death age is whatever ``LifecycleSpec`` says.
    """
    if age >= max_age:
        return 0.0
    grid = np.linspace(0.0, max_age - age, steps)
    survival = np.exp(np.exp((age - modal_age) / dispersion)
                      * (1.0 - np.exp(grid / dispersion)))
    return float(np.trapezoid(survival, grid))


@dataclasses.dataclass(frozen=True)
class LifeExpectancyRule(SpendingRule):
    """Portfolio divided by the years left in the plan -- the RMD rule.

    ``W_t / n_t`` with ``n_t`` the remaining planning years.  Like the
    constant-percent rule it cannot deplete the portfolio, but the implied
    rate rises with age instead of staying flat, which is what an investor
    with a finite horizon actually wants: there is no reason to leave a large
    balance unspent in the final year unless bequests are valued.

    ``buffer_years`` extends the planning horizon beyond the modelled death
    age, trading a lower spending rate for a larger bequest.
    """

    buffer_years: int = 0
    key: str = dataclasses.field(default="life_expectancy", init=False)
    label: str = dataclasses.field(
        default="Life expectancy / RMD", init=False)

    def _divisor(self, years_remaining: int) -> float:
        return float(max(years_remaining + self.buffer_years, 1))

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return wealth_at_retirement / self._divisor(years_remaining)

    def desired(self, state: SpendingState) -> np.ndarray:
        return state.wealth / self._divisor(state.years_remaining)


@dataclasses.dataclass(frozen=True)
class GompertzRule(SpendingRule):
    """Portfolio divided by *actuarial* remaining life expectancy.

    The same shape as :class:`LifeExpectancyRule`, but the divisor comes from
    a Gompertz survival model rather than from a fixed planning horizon.  The
    two differ sharply at the start of retirement -- at 63, expected
    remaining life under this calibration is materially shorter than the
    thirty years the model's fixed death age implies -- so the rule front-
    loads spending relative to the RMD rule and leaves less on the table.

    ``buffer_years`` lengthens the divisor for an investor who wants to plan
    beyond their own life expectancy.
    """

    modal_age: float = 88.0
    dispersion: float = 10.0
    buffer_years: float = 0.0
    key: str = dataclasses.field(default="gompertz", init=False)
    label: str = dataclasses.field(
        default="Actuarial (Gompertz life expectancy)", init=False)

    def _divisor(self, age: float) -> float:
        expectancy = gompertz_life_expectancy(
            float(age), self.modal_age, self.dispersion)
        return float(max(expectancy + self.buffer_years, 1.0))

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return wealth_at_retirement / self._divisor(age)

    def desired(self, state: SpendingState) -> np.ndarray:
        return state.wealth / self._divisor(state.age)


@dataclasses.dataclass(frozen=True)
class AmortisationRule(SpendingRule):
    """Annuity payment on the current balance at an assumed real return.

    The standard amortisation-based withdrawal (ABW):

    ``W_t * r / (1 - (1 + r)^-n)``

    with ``n`` the remaining planning years and ``r`` an assumed real return.
    It nests :class:`LifeExpectancyRule` at ``r = 0``, and a higher assumed
    return front-loads spending.  Setting ``r`` above what the portfolio
    actually earns is the classic way retirees overspend, which the sweep in
    ``docs/06`` makes visible.
    """

    assumed_return: float = 0.02
    buffer_years: int = 0
    key: str = dataclasses.field(default="amortisation", init=False)
    label: str = dataclasses.field(
        default="Amortisation (annuity payment)", init=False)

    def _factor(self, years_remaining: int) -> float:
        n = max(years_remaining + self.buffer_years, 1)
        r = self.assumed_return
        if np.isclose(r, 0.0):
            return 1.0 / n
        return r / (1.0 - (1.0 + r) ** (-n))

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        return wealth_at_retirement * self._factor(years_remaining)

    def desired(self, state: SpendingState) -> np.ndarray:
        return state.wealth * self._factor(state.years_remaining)


@dataclasses.dataclass(frozen=True)
class IncomeReplacementRule(SpendingRule):
    """Spend a fixed share of pre-retirement income, and let the portfolio
    fund whatever the pension does not.

    Every other rule here sets spending from the *portfolio*: a share of what
    was saved, or of what is left. That makes a bigger portfolio spend more
    rather than last longer, so under those rules ruin is almost
    scale-invariant -- doubling wealth doubles withdrawals and runs out at the
    same time -- and extra saving buys consumption rather than safety.

    This rule sets spending from the *standard of living* instead. The target
    is ``rate`` times income over the final working years, held in real terms
    for life; the pension is netted off first and the portfolio pays the
    remainder. That makes it the natural rule for comparing pension systems,
    because it asks the question a pension is *for*: holding the retirement a
    household wants fixed, who funds it, and who runs out.

    Two consequences follow directly from the arithmetic, and both are the
    point rather than side effects. A larger portfolio now strictly reduces
    ruin, because the target does not grow with it. And a pension now
    *displaces* withdrawals rather than adding to them, so a generous benefit
    shows up as portfolio longevity instead of extra spending.
    """

    rate: float = 0.75
    key: str = dataclasses.field(default="income_replacement", init=False)
    label: str = dataclasses.field(
        default="Replace 75% of pre-retirement income", init=False)

    def initial_withdrawal(self, wealth_at_retirement: np.ndarray,
                           years_remaining: int, age: int) -> np.ndarray:
        # The anchor for this rule is income, which arrives on the state, so
        # there is nothing to set from wealth. Feedback rules read
        # ``state.initial_withdrawal``; this one does not.
        return np.zeros_like(np.asarray(wealth_at_retirement, dtype=float))

    def target(self, state: SpendingState) -> np.ndarray:
        """Total consumption the rule is aiming at, pension included."""
        if state.pre_retirement_income is None:
            raise ValueError(
                "IncomeReplacementRule needs pre_retirement_income on the "
                "spending state; the simulator supplies it")
        return self.rate * np.asarray(state.pre_retirement_income, dtype=float)

    def desired(self, state: SpendingState) -> np.ndarray:
        benefit = (np.zeros_like(state.wealth) if state.benefit is None
                   else np.asarray(state.benefit, dtype=float))
        return np.maximum(self.target(state) - benefit, 0.0)

    def describe(self) -> Dict[str, Any]:
        found = super().describe()
        found["label"] = f"Replace {self.rate:.0%} of pre-retirement income"
        return found


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#: Factories keyed by rule name.  ``rate`` is accepted by every rule that has
#: one so that a single sweep can vary "the rate" across families.
REGISTRY: Mapping[str, Callable[..., SpendingRule]] = {
    "constant_real": ConstantRealRule,
    "constant_percent": ConstantPercentRule,
    "guyton_klinger": GuytonKlingerRule,
    "vanguard_dynamic": VanguardDynamicRule,
    "endowment": EndowmentRule,
    "life_expectancy": LifeExpectancyRule,
    "gompertz": GompertzRule,
    "amortisation": AmortisationRule,
    "income_replacement": IncomeReplacementRule,
}

#: Rules whose spending level is set by a `rate` parameter.  The remainder
#: derive their level from the planning horizon, so sweeping a rate over them
#: would do nothing.
RATE_PARAMETERISED: frozenset = frozenset(
    {"constant_real", "constant_percent", "guyton_klinger",
     "vanguard_dynamic", "endowment"})


def build(key: str, **params: Any) -> SpendingRule:
    """Instantiate a rule by name."""
    if key not in REGISTRY:
        raise ValueError(
            f"unknown spending rule {key!r}; expected one of {sorted(REGISTRY)}")
    return REGISTRY[key](**params)


def from_spec(retirement_rule: str, rule_rate: float) -> SpendingRule:
    """Map the legacy ``LifecycleSpec`` fields onto a rule object.

    ``fixed_real_rule`` and ``fixed_percentage`` are the two names the
    lifecycle config has always used; keeping them wired here means the
    headline pipeline is unchanged by the introduction of this module.
    """
    if retirement_rule == "fixed_real_rule":
        return ConstantRealRule(rate=rule_rate)
    if retirement_rule == "fixed_percentage":
        return ConstantPercentRule(rate=rule_rate)
    raise ValueError(f"unknown retirement_rule {retirement_rule!r}")
