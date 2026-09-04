"""Income tax in retirement, for the two systems Section 32 compares.

Every other document in this project is tax-free. That is a defensible
simplification while the question is asset allocation -- a tax that applies
to every strategy alike cancels out of the comparison -- but it stops being
defensible the moment two *countries* are compared, because the whole
difference between their retirement systems is which income is taxed and
when.

The gap is not symmetric, which is why it cannot be waved away:

``Australia``
    Superannuation drawn after 60 from a taxed fund is not merely tax-free,
    it is not assessable income at all.  It never touches the return, so it
    cannot drag the Age Pension into tax.  The pension itself *is*
    assessable, but the Seniors and Pensioners Tax Offset lifts the
    effective threshold above the full single rate, so a pensioner living on
    the pension pays nothing.  What Australia does tax is fund *earnings*
    during accumulation, at 15%.

``United States``
    Social security is taxable on a sliding scale, and the thresholds that
    decide it have not been indexed since 1993.  Worse for a household with
    a portfolio, withdrawals from a traditional account are ordinary income
    *and* count toward the "provisional income" that decides how much of the
    benefit is taxed -- so a dollar withdrawn can pull up to eighty-five
    cents of benefit into the tax base alongside it.  That interaction is
    the "tax torpedo", and it produces effective marginal rates well above
    the bracket the household nominally sits in.

So one system charges during accumulation, smoothly, and the other charges
during retirement, progressively, and interactively, in exactly the years a
risk-averse aggregator weights most heavily.  Reasoning about which is worse
is what this module exists to stop.

Units
-----
The lifecycle model carries normalised real income, not currency, so every
threshold here is held as a **multiple of economy-wide average earnings**
and multiplied up by the model's own average at the point of use.  That is
the same convention :mod:`src.pension` uses for the Age Pension schedule,
and it is what lets one scale travel across the panel's currencies.

Sources, all read 2 September 2026:

* Australian resident rates and thresholds, Medicare levy and the seniors
  and pensioners levy threshold, 2025-26; LITO; SAPTO single maximum and
  shade-out.  Australian Taxation Office.
* US federal single-filer brackets, standard deduction, additional standard
  deduction for 65+ and the temporary senior deduction, 2025.  Internal
  Revenue Service.
* Provisional-income thresholds for taxing social security: 26 U.S.C.
  section 86, unchanged since 1993.
* SSA national average wage index, used only to put the US thresholds on the
  same average-earnings footing as the Australian ones.

What is deliberately *not* modelled: state and local income taxes; the
Australian marginal tax on earnings in a non-super account; capital gains
tax and its discount, which would reduce the effective rate on fund
earnings below the statutory 15%; and any filing status other than single,
since the lifecycle model has one earner and no spouse.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "Scale", "Offset", "Regime", "REGIMES",
    "AU_SCALE", "US_SCALE", "AU_LITO", "AU_SAPTO",
    "taxable_social_security", "regime_from_config", "effective_rate_curve",
]


# ---------------------------------------------------------------------------
# The pieces of a tax system
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Scale:
    """A marginal rate scale plus a flat levy.

    ``brackets`` is ``(threshold, rate)`` from the bottom up, with thresholds
    as multiples of average earnings.  ``levy`` is charged on the whole of
    income once it passes ``levy_threshold`` -- which is how Australia's
    Medicare levy behaves closely enough at this resolution, and which is
    zero for a system that has no such charge.
    """

    brackets: Tuple[Tuple[float, float], ...]
    levy: float = 0.0
    levy_threshold: float = 0.0

    def __post_init__(self) -> None:
        thresholds = [b[0] for b in self.brackets]
        if thresholds != sorted(thresholds):
            raise ValueError("scale brackets must ascend by threshold")
        if thresholds and thresholds[0] != 0.0:
            raise ValueError("the first bracket must start at zero")
        if any(not 0.0 <= b[1] < 1.0 for b in self.brackets):
            raise ValueError("marginal rates must lie in [0, 1)")

    def tax(self, income: np.ndarray, average: float) -> np.ndarray:
        """Tax owed on ``income``, both in the model's own units."""
        taxable = np.maximum(np.asarray(income, dtype=float), 0.0)
        owed = np.zeros_like(taxable)
        uppers = [b[0] for b in self.brackets[1:]] + [np.inf]
        for (lower, rate), upper in zip(self.brackets, uppers):
            floor_, ceil_ = lower * average, upper * average
            owed += np.clip(taxable - floor_, 0.0,
                            np.inf if not np.isfinite(ceil_)
                            else ceil_ - floor_) * rate
        if self.levy:
            owed += np.where(taxable > self.levy_threshold * average,
                             taxable * self.levy, 0.0)
        return owed


@dataclasses.dataclass(frozen=True)
class Offset:
    """A tax offset that shades out above a threshold.

    Offsets reduce tax owed but cannot make it negative, which is what
    separates them from a refundable credit and what makes Australia's
    effective tax-free threshold for a senior so much higher than the
    statutory one.
    """

    maximum: float
    threshold: float
    taper: float

    def amount(self, income: np.ndarray, average: float) -> np.ndarray:
        excess = np.maximum(np.asarray(income, dtype=float)
                            - self.threshold * average, 0.0)
        return np.maximum(self.maximum * average - self.taper * excess, 0.0)


# ---------------------------------------------------------------------------
# Australia
# ---------------------------------------------------------------------------
#: Resident rates 2025-26, as multiples of AWOTE (A$106,657.20).
AU_SCALE = Scale(
    brackets=((0.0, 0.0),
              (18_200.0 / 106_657.20, 0.16),
              (45_000.0 / 106_657.20, 0.30),
              (135_000.0 / 106_657.20, 0.37),
              (190_000.0 / 106_657.20, 0.45)),
    levy=0.02,
    # Seniors and pensioners keep the Medicare levy off until well above the
    # ordinary threshold, which is part of why a pensioner pays nothing.
    levy_threshold=41_089.0 / 106_657.20,
)

#: Low income tax offset: A$700, shading out from A$37,500.  The second,
#: gentler taper above A$45,000 is not modelled; it lies above the income
#: range a retiree on the Age Pension occupies, which is what this is for.
AU_LITO = Offset(maximum=700.0 / 106_657.20,
                 threshold=37_500.0 / 106_657.20, taper=0.05)

#: Seniors and pensioners tax offset, single: A$2,230, shading out at 12.5
#: cents in the dollar from A$32,279 of rebate income.  This is the offset
#: that makes the Age Pension untaxed in practice.
AU_SAPTO = Offset(maximum=2_230.0 / 106_657.20,
                  threshold=32_279.0 / 106_657.20, taper=0.125)

#: Statutory rate on superannuation fund earnings during accumulation.  It
#: is not the rate a fund pays: see :class:`FundTax`, which is what should be
#: used.  Retained only as the schedule's headline.
AU_FUND_EARNINGS_TAX: float = 0.15

#: One-third discount on gains held beyond twelve months, so a realised gain
#: is taxed at 10% rather than 15%.
AU_CGT_DISCOUNT: float = 1.0 / 3.0

#: Australian corporate rate, which sets the imputation credit.
AU_COMPANY_RATE: float = 0.30

#: Mean dividend yield on this panel, measured rather than assumed:
#: ``src.valuation.trailing_yield`` over every country and year.  The income
#: component of the fund's tax base is this, not the total return.
PANEL_DIVIDEND_YIELD: float = 0.0399


@dataclasses.dataclass(frozen=True)
class FundTax:
    """What a superannuation fund actually pays while it accumulates.

    Not the statutory rate on the return, and not close to it.  Three things
    stand between the headline and the charge, and together they very nearly
    cancel it:

    **Unrealised gains are not income.**  A fund that holds pays nothing on
    appreciation; only what it *realises* is assessable.  And because
    earnings in the retirement phase are exempt outright, a gain carried
    across that boundary is never taxed at all -- so a member who does not
    realise before the pension starts pays capital gains tax of zero, not of
    fifteen per cent.

    **Realised gains held beyond a year carry a one-third discount**, so the
    rate on them is ten per cent rather than fifteen.

    **Franked dividends carry an imputation credit worth more than the
    liability.**  At a 30% company rate against a 15% fund rate the credit is
    worth +21.4% of the cash dividend -- Section #franking derives this on
    the same panel -- so a domestic dividend is a *refund*, and it subsidises
    the tax on the international sleeve rather than adding to it.

    What survives is 15% on unfranked dividends and 10% on whatever
    rebalancing forces the fund to sell.  On this panel and this portfolio
    those two very nearly offset, which is the point of the class.

    ``realisation`` is the share of the portfolio turned over each year, and
    ``embedded_gain`` the fraction of a sold parcel that is gain rather than
    cost base.  Setting ``realisation`` to zero is the case the reader should
    hold in mind: hold until the pension phase, and only the dividends are
    ever taxed.
    """

    rate: float = AU_FUND_EARNINGS_TAX
    cgt_discount: float = AU_CGT_DISCOUNT
    company_rate: float = AU_COMPANY_RATE
    franked_share: float = 1.0
    dividend_yield: float = PANEL_DIVIDEND_YIELD
    realisation: float = 0.0335
    embedded_gain: float = 0.60

    def __post_init__(self) -> None:
        for name in ("rate", "cgt_discount", "company_rate", "franked_share",
                     "embedded_gain"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
        if self.dividend_yield < 0.0 or self.realisation < 0.0:
            raise ValueError("dividend_yield and realisation must be >= 0")

    @property
    def capital_gains_rate(self) -> float:
        """The rate a discounted long-held gain actually attracts."""
        return self.rate * (1.0 - self.cgt_discount)

    def dividend_value(self, franked: float) -> float:
        """What a dollar of dividend is worth to the fund, less the dollar.

        Positive where the imputation credit exceeds the fund's own tax.
        Delegated to :mod:`src.franking` so one arithmetic serves both
        sections.
        """
        from . import franking as fk

        return fk.credit_rate(self.company_rate, self.rate, float(franked))

    def income_drag(self, domestic_weight: float) -> float:
        """Annual drag from dividends. *Negative* is a charge."""
        w = float(domestic_weight)
        per_dollar = (w * self.dividend_value(self.franked_share)
                      + (1.0 - w) * self.dividend_value(0.0))
        return self.dividend_yield * per_dollar

    def gains_drag(self) -> float:
        """Annual drag from gains rebalancing forces the fund to realise."""
        return -(self.realisation * self.embedded_gain
                 * self.capital_gains_rate)

    def drag(self, domestic_weight: float) -> float:
        """Total annual proportional drag on the fund sleeve's return.

        Signed: negative is a cost, and a portfolio franked enough can come
        out positive.
        """
        return self.income_drag(domestic_weight) + self.gains_drag()

    def components(self, domestic_weight: float) -> Dict[str, float]:
        """The drag with its parts, for a table that has to show its working."""
        income = self.income_drag(domestic_weight)
        gains = self.gains_drag()
        return {
            "domestic_weight": float(domestic_weight),
            "dividend_yield": self.dividend_yield,
            "franked_credit": self.dividend_value(self.franked_share),
            "unfranked_credit": self.dividend_value(0.0),
            "income_drag": income,
            "capital_gains_rate": self.capital_gains_rate,
            "gains_drag": gains,
            "total_drag": income + gains,
            "naive_drag": -(self.rate * (self.dividend_yield + 0.04)),
        }



# ---------------------------------------------------------------------------
# United States
# ---------------------------------------------------------------------------
#: SSA national average wage index, used only to put the US thresholds on an
#: average-earnings footing.  A later vintage would move every US threshold
#: in the same direction, so the study sweeps it rather than relying on it.
US_AVERAGE_WAGE: float = 66_621.80

#: Federal single-filer brackets, 2025.
US_SCALE = Scale(
    brackets=((0.0, 0.10),
              (11_925.0 / US_AVERAGE_WAGE, 0.12),
              (48_475.0 / US_AVERAGE_WAGE, 0.22),
              (103_350.0 / US_AVERAGE_WAGE, 0.24),
              (197_300.0 / US_AVERAGE_WAGE, 0.32),
              (250_525.0 / US_AVERAGE_WAGE, 0.35),
              (626_350.0 / US_AVERAGE_WAGE, 0.37)),
)

#: Standard deduction, single, 2025, plus the additional deduction for a
#: filer aged 65 or over, plus the temporary senior deduction enacted in
#: 2025.  The last is legislated to expire after 2028; carrying it makes the
#: US arm's tax *lower*, so including it is the conservative choice for a
#: study that finds against the US arm.
US_STANDARD_DEDUCTION: float = 15_000.0 / US_AVERAGE_WAGE
US_SENIOR_DEDUCTION: float = (2_000.0 + 6_000.0) / US_AVERAGE_WAGE

#: Provisional-income thresholds from 26 U.S.C. section 86, single filer.
#: Unindexed since 1993, which is why they catch a household this model
#: would once have left alone.
US_PROVISIONAL_BASE: float = 25_000.0 / US_AVERAGE_WAGE
US_PROVISIONAL_SECOND: float = 34_000.0 / US_AVERAGE_WAGE

#: Average federal rate a single filer pays on average earnings, net of the
#: working-age standard deduction. Used only to gross a pre-tax contribution
#: onto the model's take-home base -- the same wedge
#: :func:`src.leisure.guarantee_overrides` applies to the Superannuation
#: Guarantee, so that a 401(k) and a super fund are compared on one footing.
US_AVERAGE_TAX_RATE: float = float(
    US_SCALE.tax(np.array([US_AVERAGE_WAGE * (1.0 - US_STANDARD_DEDUCTION)]),
                 US_AVERAGE_WAGE)[0] / US_AVERAGE_WAGE)


def taxable_social_security(benefit: np.ndarray, other_income: np.ndarray,
                            average: float,
                            base: float = US_PROVISIONAL_BASE,
                            second: float = US_PROVISIONAL_SECOND,
                            ) -> np.ndarray:
    """How much of a social security benefit enters taxable income.

    Provisional income is other income plus *half* the benefit.  Below the
    first threshold none of the benefit is taxed; between the thresholds up
    to half of it is; above the second up to 85% is.  The 85% is an
    inclusion share and not a rate: the included part is then taxed at the
    filer's ordinary marginal rate.

    The reason this matters more than a flat tax on benefits would is the
    *interaction*.  Every extra dollar of other income raises provisional
    income dollar for dollar, so in the phase-in it drags up to 85 cents of
    benefit into the tax base with it, and the effective marginal rate is
    the statutory one multiplied by as much as 1.85.
    """
    benefit = np.maximum(np.asarray(benefit, dtype=float), 0.0)
    other = np.maximum(np.asarray(other_income, dtype=float), 0.0)
    provisional = other + 0.5 * benefit
    first, second_ = base * average, second * average

    tier_one = np.minimum(0.5 * np.maximum(provisional - first, 0.0),
                          0.5 * benefit)
    carried = np.minimum(0.5 * (second_ - first), 0.5 * benefit)
    tier_two = 0.85 * np.maximum(provisional - second_, 0.0) + carried
    included = np.where(provisional <= second_, tier_one,
                        np.minimum(tier_two, 0.85 * benefit))
    return np.minimum(np.maximum(included, 0.0), 0.85 * benefit)


# ---------------------------------------------------------------------------
# A country's retirement tax, as one object
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Regime:
    """What a retiree in one system owes on a pension and a withdrawal.

    ``benefit_taxable`` is how the public pension is treated: ``"none"``
    leaves it out of assessable income, ``"full"`` includes all of it, and
    ``"provisional"`` runs the US sliding scale.  ``withdrawal_taxable`` is
    the share of a portfolio withdrawal that is ordinary income -- one for a
    traditional account, zero for a Roth or for Australian super after 60.
    """

    key: str
    label: str
    scale: Scale
    benefit_taxable: str = "none"
    withdrawal_taxable: float = 0.0
    deduction: float = 0.0
    offsets: Tuple[Offset, ...] = ()
    #: The fund's own tax while accumulating, as a model rather than a rate.
    #: ``None`` for a system that does not tax a retirement fund's earnings.
    fund_tax: "FundTax | None" = None

    def __post_init__(self) -> None:
        if self.benefit_taxable not in ("none", "full", "provisional"):
            raise ValueError(
                f"unknown benefit_taxable {self.benefit_taxable!r}; expected "
                f"'none', 'full' or 'provisional'")
        if not 0.0 <= self.withdrawal_taxable <= 1.0:
            raise ValueError("withdrawal_taxable must lie in [0, 1]")


    def assessable(self, benefit: np.ndarray, withdrawal: np.ndarray,
                   average: float) -> np.ndarray:
        """Income that reaches the tax return."""
        other = self.withdrawal_taxable * np.maximum(
            np.asarray(withdrawal, dtype=float), 0.0)
        benefit = np.maximum(np.asarray(benefit, dtype=float), 0.0)
        if self.benefit_taxable == "none":
            return other
        if self.benefit_taxable == "full":
            return other + benefit
        return other + taxable_social_security(benefit, other, average)

    def tax(self, benefit: np.ndarray, withdrawal: np.ndarray,
            average: float) -> np.ndarray:
        """Tax owed, floored at zero after offsets."""
        assessable = self.assessable(benefit, withdrawal, average)
        owed = self.scale.tax(
            np.maximum(assessable - self.deduction * average, 0.0), average)
        for offset in self.offsets:
            # Offsets reduce tax owed and cannot refund it, so each is
            # applied against what is left rather than summed first.
            owed = np.maximum(owed - offset.amount(assessable, average), 0.0)
        return owed


#: The regimes Section 32's arms are run under.
#:
#: The two US arms are the honest pair.  Modelling US saving as a Roth --
#: contributions from take-home pay, nothing taxed thereafter -- is what
#: every earlier section of this project implicitly assumed, and it taxes
#: only the benefit.  Modelling it as a traditional account is the
#: comparison Australia's superannuation actually deserves, since both then
#: take contributions from pre-tax earnings; it is also what switches the
#: torpedo on.
REGIMES: Mapping[str, Regime] = {
    "none": Regime(key="none", label="No tax (every other section)",
                   scale=Scale(brackets=((0.0, 0.0),))),
    "us_roth": Regime(
        key="us_roth", label="US, saving in a Roth",
        scale=US_SCALE, benefit_taxable="provisional",
        withdrawal_taxable=0.0,
        deduction=US_STANDARD_DEDUCTION + US_SENIOR_DEDUCTION),
    "us_traditional": Regime(
        key="us_traditional", label="US, saving in a traditional account",
        scale=US_SCALE, benefit_taxable="provisional",
        withdrawal_taxable=1.0,
        deduction=US_STANDARD_DEDUCTION + US_SENIOR_DEDUCTION),
    "au": Regime(
        key="au", label="Australia, superannuation after 60",
        scale=AU_SCALE, benefit_taxable="full",
        # Super drawn after 60 from a taxed fund is not assessable income at
        # all, which is why it cannot drag the pension into tax.
        withdrawal_taxable=0.0,
        offsets=(AU_SAPTO, AU_LITO),
        fund_tax=FundTax()),
}


def regime_from_config(key: str, cfg: Mapping[str, Any] | None = None
                       ) -> Regime:
    """A regime by name, with any config overrides applied."""
    if key not in REGIMES:
        raise ValueError(f"unknown tax regime {key!r}; expected one of "
                         f"{tuple(REGIMES)}")
    regime = REGIMES[key]
    block = (cfg or {}).get("tax", {}) if cfg else {}
    override = block.get(key, {}) if isinstance(block, Mapping) else {}
    if not override:
        return regime
    allowed = {"withdrawal_taxable", "deduction", "realisation",
               "franked_share", "dividend_yield", "embedded_gain"}
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"unknown tax override(s) {sorted(unknown)} for "
                         f"regime {key!r}; expected {sorted(allowed)}")
    fund_keys = {"realisation", "franked_share", "dividend_yield",
                 "embedded_gain"}
    fund_over = {k: float(v) for k, v in override.items() if k in fund_keys}
    plain = {k: float(v) for k, v in override.items() if k not in fund_keys}
    if fund_over:
        base = regime.fund_tax or FundTax()
        plain["fund_tax"] = dataclasses.replace(base, **fund_over)
    return dataclasses.replace(regime, **plain)


#: What each pension system's arm is paired with. The two US rows are the
#: honest pair: modelling American saving as a Roth is what every earlier
#: section implicitly assumed, and modelling it as a traditional account is
#: the comparison Australian superannuation actually deserves, since both
#: then take contributions from pre-tax earnings.
SYSTEM_REGIMES: Mapping[str, Tuple[str, ...]] = {
    "us": ("none", "us_roth", "us_traditional"),
    "au_pension_only": ("none", "au"),
    "au_as_legislated": ("none", "au"),
}


def arms(systems: Sequence[str],
         regimes: Mapping[str, Sequence[str]] | None = None,
         ) -> Tuple[Tuple[str, str], ...]:
    """``(system, regime)`` pairs to run, untaxed arm first for each."""
    table = regimes or SYSTEM_REGIMES
    out = []
    for system in systems:
        for key in table.get(system, ("none",)):
            out.append((str(system), str(key)))
    return tuple(out)


def tax_verdict(frame: "Any") -> Dict[str, Any]:
    """What putting real tax in does to the ranking.

    The question is not whether tax lowers consumption -- it must -- but
    whether it lowers it *unevenly enough to change the answer*. A tax that
    costs both systems the same is a level effect and can be ignored; one
    that closes or reverses the gap cannot.
    """
    import pandas as pd

    if frame is None or not len(frame):
        return {"measured": False}
    wide = frame.set_index(["system", "regime"])

    def cell(system: str, regime: str, column: str) -> float:
        try:
            return float(wide.loc[(system, regime), column])
        except KeyError:
            return float("nan")

    found: Dict[str, Any] = {"measured": True}
    rows = []
    for system in frame["system"].unique():
        free = cell(system, "none", "cec")
        for regime in frame[frame["system"] == system]["regime"].unique():
            if regime == "none":
                continue
            taxed = cell(system, regime, "cec")
            rows.append({
                "system": system, "regime": regime,
                "cec_untaxed": free, "cec_taxed": taxed,
                "cost_pct": 100.0 * (taxed / free - 1.0) if free else np.nan,
            })
    if not rows:
        return found
    found["rows"] = rows
    by = {(r["system"], r["regime"]): r["cost_pct"] for r in rows}
    au = by.get(("au_as_legislated", "au"), float("nan"))
    for label, key in (("us_roth", ("us", "us_roth")),
                       ("us_traditional", ("us", "us_traditional"))):
        us = by.get(key, float("nan"))
        if np.isfinite(us) and np.isfinite(au):
            found[f"{label}_cost_pct"] = us
            found[f"{label}_gap_pp"] = us - au
            # The only comparison that matters: does tax cost the American
            # arm more than the Australian one, and by enough to matter?
            found[f"{label}_costs_more_than_au"] = bool(us < au)
    found["au_cost_pct"] = au
    # Does the untaxed ranking survive?
    us_free = cell("us", "none", "cec")
    au_free = cell("au_as_legislated", "none", "cec")
    us_trad = cell("us", "us_traditional", "cec")
    au_taxed = cell("au_as_legislated", "au", "cec")
    if all(np.isfinite(x) for x in (us_free, au_free, us_trad, au_taxed)):
        found["us_led_untaxed"] = bool(us_free > au_free)
        found["us_leads_taxed"] = bool(us_trad > au_taxed)
        found["ranking_survives"] = bool(
            found["us_led_untaxed"] == found["us_leads_taxed"])
        found["gap_untaxed_pct"] = 100.0 * (au_free / us_free - 1.0)
        found["gap_taxed_pct"] = 100.0 * (au_taxed / us_trad - 1.0)
        found["gap_narrowed"] = bool(
            abs(found["gap_taxed_pct"]) < abs(found["gap_untaxed_pct"]))
        found["gap_change_pp"] = (found["gap_taxed_pct"]
                                  - found["gap_untaxed_pct"])
        # Which side moved it. A gap can widen because one arm lost or
        # because the other gained, and those are different findings: the
        # prose has to name the one that happened rather than assume the
        # taxed-more arm did the work.
        us_move = abs(by.get(("us", "us_traditional"), float("nan")))
        au_move = abs(au)
        if np.isfinite(us_move) and np.isfinite(au_move):
            total = us_move + au_move
            found["driver"] = "us" if us_move > au_move else "au"
            found["driver_share"] = (max(us_move, au_move) / total
                                     if total else float("nan"))
    return found


def effective_rate_curve(regime: Regime, benefit: float, average: float,
                         withdrawals: Sequence[float]) -> Dict[str, Any]:
    """Marginal and average rates across a range of withdrawals.

    The point of reporting the *marginal* rate rather than the bracket is
    the torpedo: where each withdrawn dollar also drags benefit into the tax
    base, the marginal rate exceeds every rate in the statute, and no
    reader would find it by looking one up.
    """
    draws = np.asarray(list(withdrawals), dtype=float)
    ben = np.full_like(draws, float(benefit))
    owed = regime.tax(ben, draws, average)
    gross = ben + draws
    # A forward difference, not a central one: the quantity a retiree cares
    # about is the tax on the *next* dollar drawn, and a central difference
    # averages across the kink the torpedo creates, which is exactly the
    # feature being measured. The last point repeats its predecessor so the
    # array lines up with `draws`.
    if len(draws) > 1:
        step = np.diff(owed) / np.diff(draws)
        step = np.append(step, step[-1])
    else:
        step = np.zeros_like(owed)
    # The statutory rate the household would look up, against the rate it
    # actually faces. Their difference is the torpedo, and it is the only
    # honest way to say the effect is present: a high marginal rate at a
    # high income is just a high bracket.
    assessable = regime.assessable(ben, draws, average)
    taxable = np.maximum(assessable - regime.deduction * average, 0.0)
    statutory = np.zeros_like(taxable)
    for lower, rate in regime.scale.brackets:
        statutory = np.where(taxable > lower * average, rate, statutory)
    # A filer whose taxable income is nil is not "in the bottom bracket" in
    # any sense that matters here: the next dollar they draw is untaxed. So
    # the bracket is zero there, and the excess over it is zero too, which
    # is what a system with no torpedo has to report.
    statutory = np.where(taxable > 0.0, statutory, 0.0)
    excess = step - statutory
    # Only where the filer is actually *in* a bracket. At the edge of the
    # standard deduction the rate jumps from nothing to something, and that
    # is the deduction ending rather than a torpedo: a torpedo is being
    # charged more than the bracket you are in, which needs a bracket.
    inside = statutory > 0.0
    searchable = np.where(inside, excess, -np.inf)
    hit = int(np.argmax(searchable)) if inside.any() else 0
    return {
        "withdrawal": draws, "tax": owed,
        "average_rate": np.divide(owed, gross, out=np.zeros_like(owed),
                                  where=gross > 0.0),
        "marginal_rate": step,
        "statutory_rate": statutory,
        "excess_over_statutory": excess,
        "peak_marginal": float(step.max()) if len(step) else float("nan"),
        "top_statutory": float(max(r for _, r in regime.scale.brackets)
                               + regime.scale.levy),
        # Where the torpedo is worst: the rate faced, the bracket the filer
        # is nominally in, and the withdrawal that puts them there.
        "torpedo_marginal": float(step[hit]) if len(step) else float("nan"),
        "torpedo_statutory": (float(statutory[hit]) if len(statutory)
                              else float("nan")),
        "torpedo_excess": (float(excess[hit]) if len(excess) and inside.any()
                           else 0.0),
        "torpedo_at": float(draws[hit]) if len(draws) else float("nan"),
        "torpedo_multiple": (float(step[hit] / statutory[hit])
                             if len(step) and statutory[hit] > 0.0
                             else float("nan")),
    }
