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

#: Tax on superannuation fund earnings during accumulation.  The statutory
#: rate is 15%; the rate a growth fund actually pays is lower, because
#: franking credits (Section #franking measures them on this panel) and the
#: one-third discount on gains held beyond a year both reduce it.  The
#: statutory rate is the default and the study sweeps below it.
AU_FUND_EARNINGS_TAX: float = 0.15



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
    fund_earnings_tax: float = 0.0

    def __post_init__(self) -> None:
        if self.benefit_taxable not in ("none", "full", "provisional"):
            raise ValueError(
                f"unknown benefit_taxable {self.benefit_taxable!r}; expected "
                f"'none', 'full' or 'provisional'")
        if not 0.0 <= self.withdrawal_taxable <= 1.0:
            raise ValueError("withdrawal_taxable must lie in [0, 1]")
        if not 0.0 <= self.fund_earnings_tax < 1.0:
            raise ValueError("fund_earnings_tax must lie in [0, 1)")

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
        fund_earnings_tax=AU_FUND_EARNINGS_TAX),
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
    allowed = {"withdrawal_taxable", "fund_earnings_tax", "deduction"}
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"unknown tax override(s) {sorted(unknown)} for "
                         f"regime {key!r}; expected {sorted(allowed)}")
    return dataclasses.replace(regime, **{k: float(v)
                                          for k, v in override.items()})


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
