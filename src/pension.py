"""What if the state pension is not the American one?

Every result in this paper puts the same public pension behind all sixteen
countries: the US primary-insurance-amount formula, progressive in career
earnings, paid in full regardless of what else the retiree owns. That is a
convenient choice and a questionable one. It is the schedule of exactly one
country in the panel, and the panel's whole point is that countries differ.

It also happens to be the *shape* that is kindest to the paper's argument.
An earnings-related pension is a bond-like endowment: it arrives whatever the
portfolio does, so it crowds fixed income out of the financial portfolio and
pushes the optimal equity share up. If the public pension were instead
withdrawn as private wealth rose, the same logic would run backwards.

Australia is the natural test, because it is the developed-world system built
on the opposite principle. The Age Pension is flat rather than earnings-
related, and it is *means-tested*: assessable assets above a free area reduce
it, at a taper steep enough to act as a wealth tax, until it cuts out
entirely. Retirement saving is separately compulsory through the
Superannuation Guarantee, now 12% of ordinary time earnings.

The two features pull in opposite directions and both bear on the headline:

* The **taper insures and claws back at once.** A retiree whose portfolio
  falls is met by a pension that rises; one whose portfolio does well loses
  pension for the privilege. That compresses the distribution of retirement
  consumption from both ends, which should shrink the measured distance
  between *any* two portfolios -- including the one this paper is about.
* The **contribution rate is higher**, which pushes wealth at retirement up
  and therefore pushes more retirees past the cut-off, where the taper stops
  biting and the portfolio decision is once again the whole story.

So the interesting question is not whether the numbers move. It is whether
they move enough to reorder anything, and which of the two features is doing
the moving -- which is why the sweep runs the Australian pension at the
paper's own savings rate as well as at the statutory one.

**Calibration.** Rates are held as multiples of economy-wide average
earnings so they travel across the panel's currencies. Against Australian
Average Weekly Ordinary Time Earnings for full-time adults of $2,051.10 a
week (ABS, November 2025; $106,657 a year):

===========================  ==============  ========================
Quantity                     Value           In average earnings
===========================  ==============  ========================
Maximum single rate          $1,200.90/ft    0.293
Assets-test free area        $321,500        3.014
Cut-out                      $722,000        6.769
Taper                        $3/ft/$1,000    0.078 a year per unit
Superannuation Guarantee     12% of OTE      --
===========================  ==============  ========================

Rates are the single homeowner's, at the March 2026 indexation, with the
assets-test thresholds from July 2026.

**What the Superannuation Guarantee is, in this model.** It is a *second*
contribution stream, not a larger first one. The worker still saves the
paper's own 10% out of take-home pay; on top of that the employer pays 12% of
ordinary time earnings into a separate fund, 15% of it is taken as
contributions tax, and the remaining 10.2% is invested in the same strategy
as everything else and assessed by the same means test. Total contributions
are therefore 20.2% of income against the American saver's 10%. Because both
pots hold the same strategy and face no differential tax on earnings here,
they are financially one pot and are simulated as one; the split is reported
from the contribution ratio, which is exact.

That design has a consequence worth stating before any number is read:
working-life consumption is unchanged by the guarantee, because its statutory
incidence is on the employer. If the true incidence is on workers through
lower wages — which is what most of the empirical literature finds — an
Australian worker is paying for their guarantee in forgone pay, and the
comparison below is generous to them by exactly that amount. It does not
touch the certainty equivalents, because the utility window in this paper is
retirement only and working-life consumption never enters it. It does mean
the comparison is between *systems as legislated* rather than between two
workers with the same lifetime resources, and the matched-contribution rows
exist so that a reader can have the second comparison too.

**What this leaves out.** The income test (deeming) is not modelled: for a
retiree whose assets are mostly superannuation the assets test is the binding
one, which is the case here, but that is an assumption and not a theorem. The
family home's exemption from the assets test is the largest single feature of
the real system and it is absent, as housing is absent from the baseline
model. Superannuation's tax treatment -- 15% going in, 15% on earnings in
accumulation, nothing in the pension phase -- is not modelled either, in a
paper that models no taxes anywhere; the sweep instead carries the statutory
12% and the 10.2% that survives the contributions tax, which is a way of
bracketing it rather than a substitute for modelling it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

#: Multiples of economy-wide average earnings. See the module docstring for
#: the Australian figures these come from and the date they were read.
AWOTE_ANNUAL_AUD: float = 106_657.20
FULL_RATE_AUD: float = 1_200.90 * 26.0
FREE_AREA_AUD: float = 321_500.0
#: The assets test's free area is higher for a retiree who does not own a
#: home, because the family home is exempt from the test. The model's
#: investor owns no home and pays no rent, so neither case fits exactly and
#: both are carried.
FREE_AREA_NON_HOMEOWNER_AUD: float = 600_000.0
TAPER_PER_YEAR: float = 3.0 * 26.0 / 1_000.0

#: The Superannuation Guarantee, and what is left of it after the 15% tax on
#: concessional contributions. The paper's own 10% savings rate sits between
#: them, which is worth noticing before reading anything into the difference.
SG_RATE: float = 0.12
SG_CONTRIBUTIONS_TAX: float = 0.15

#: Resident income tax scale, ``(threshold, marginal rate)`` from the bottom
#: up, plus the Medicare levy. 2025-26 rates, read 2 September 2026.
#:
#: This is here for one purpose: the two contribution streams in this model
#: are quoted on different bases. A household's voluntary saving is a share
#: of what reaches their bank account; the Superannuation Guarantee is a
#: share of pre-tax ordinary time earnings. Comparing them without the wedge
#: understates the guarantee by the average tax rate, so the wedge is
#: computed here from the scale rather than guessed.
TAX_BRACKETS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (18_200.0, 0.16),
    (45_000.0, 0.30),
    (135_000.0, 0.37),
    (190_000.0, 0.45),
)
MEDICARE_LEVY: float = 0.02


def income_tax(taxable: float,
               brackets: Tuple[Tuple[float, float], ...] = TAX_BRACKETS,
               levy: float = MEDICARE_LEVY) -> float:
    """Tax on ``taxable`` under a marginal scale, plus a flat levy."""
    taxable = max(float(taxable), 0.0)
    owed = 0.0
    for (lower, rate), upper in zip(
            brackets, [b[0] for b in brackets[1:]] + [float("inf")]):
        if taxable <= lower:
            break
        owed += (min(taxable, upper) - lower) * rate
    return owed + taxable * levy


def average_tax_rate(income: float = AWOTE_ANNUAL_AUD, **kwargs: Any) -> float:
    """The average, not marginal, rate at ``income``.

    Average because the whole of the guarantee is grossed up, not its top
    slice: the question is what share of a salary reaches the bank, and that
    is the average rate by definition.
    """
    return income_tax(income, **kwargs) / income if income > 0.0 else 0.0


@dataclasses.dataclass(frozen=True)
class System:
    """One public-pension regime, as a set of overrides on the baseline spec."""

    key: str
    label: str
    overrides: Mapping[str, Any]
    note: str = ""


def australian_parameters() -> Dict[str, float]:
    """The Age Pension schedule in multiples of economy-wide average earnings."""
    return {
        "pension_full_rate": FULL_RATE_AUD / AWOTE_ANNUAL_AUD,
        "pension_free_area": FREE_AREA_AUD / AWOTE_ANNUAL_AUD,
        "pension_free_area_non_homeowner":
            FREE_AREA_NON_HOMEOWNER_AUD / AWOTE_ANNUAL_AUD,
        "pension_taper": TAPER_PER_YEAR,
    }


def from_config(cfg: Mapping[str, Any]) -> Dict[str, float]:
    """The Age Pension schedule read out of the ``pension`` config block.

    Keeping the statutory dollars in the config and the division here means a
    reader can check the arithmetic against a payment summary without reading
    any Python, and an indexation update is a one-line edit rather than a
    recomputed constant.
    """
    block = cfg.get("pension", {})
    awote = float(block.get("awote_annual_aud", AWOTE_ANNUAL_AUD))
    return {
        "pension_full_rate": float(
            block.get("full_rate_annual_aud", FULL_RATE_AUD)) / awote,
        "pension_free_area": float(
            block.get("free_area_aud", FREE_AREA_AUD)) / awote,
        "pension_free_area_non_homeowner": float(
            block.get("free_area_non_homeowner_aud",
                      FREE_AREA_NON_HOMEOWNER_AUD)) / awote,
        "pension_taper": float(
            block.get("taper_per_1000_fortnight", 3.0)) * 26.0 / 1_000.0,
    }


def default_systems(savings_rate: float,
                    au: Mapping[str, float] | None = None,
                    sg_rate: float = SG_RATE,
                    sg_tax: float = SG_CONTRIBUTIONS_TAX) -> Tuple[System, ...]:
    """A factorial design over the two things that differ between the systems.

    Australia differs from the United States in two ways at once, and they
    push in opposite directions:

    * the **pension** is flat and means-tested rather than earnings-related
      and universal, which should compress outcomes and could reorder them;
    * the **contribution** is higher, because the Superannuation Guarantee is
      compulsory and sits on top of whatever the worker saves voluntarily,
      which should raise every outcome.

    A single Australia-versus-America row would confound the two, so the
    sweep crosses them:

    ==================  ===========================  =========================
    Contribution        US schedule (earnings-       Australian Age Pension
                        related, universal)          (flat, means-tested)
    ==================  ===========================  =========================
    Voluntary only      ``us_social_security``       ``age_pension_matched``
    Voluntary + SG      ``us_matched_saving``        ``australia_as_legislated``
    ==================  ===========================  =========================

    plus ``age_pension_untested``, which pays the Age Pension's own rate to
    everybody and so separates the means test from the smaller cheque.

    The corner that matters for a reader is ``australia_as_legislated``: 10%
    voluntary saving *and* 12% of ordinary time earnings paid by the employer
    into a super fund, less the 15% contributions tax, invested in the same
    strategy as everything else and assessed by the same means test.
    """
    au = dict(au if au is not None else australian_parameters())
    renter = au.pop("pension_free_area_non_homeowner", None)
    untested = dict(au, pension_taper=0.0)
    net = sg_rate * (1.0 - sg_tax)
    return (
        System("us_social_security",
               "US Social Security (the paper's baseline)",
               {"social_security_formula": "progressive",
                "savings_rate": savings_rate, "super_guarantee_rate": 0.0},
               "Progressive in career earnings, paid regardless of wealth."),
        System("us_matched_saving",
               f"US Social Security, saving {savings_rate + net:.1%}",
               {"social_security_formula": "progressive",
                "savings_rate": savings_rate,
                "super_guarantee_rate": sg_rate,
                "super_contributions_tax": sg_tax},
               "The American pension at the Australian contribution rate: "
               "what the extra saving is worth on its own."),
        System("age_pension_untested",
               "Age Pension rate, paid to everyone (no means test)",
               {"social_security_formula": "means_tested",
                "savings_rate": savings_rate, "super_guarantee_rate": 0.0,
                **untested},
               "The level change alone, with the assets test switched off."),
        System("age_pension_matched",
               f"Australian Age Pension, saving {savings_rate:.0%}",
               {"social_security_formula": "means_tested",
                "savings_rate": savings_rate, "super_guarantee_rate": 0.0,
                **au},
               "The means test alone, at the baseline contribution rate."),
        System("australia_as_legislated",
               f"Australia as legislated: {savings_rate:.0%} voluntary "
               f"+ {sg_rate:.0%} SG, means-tested pension",
               {"social_security_formula": "means_tested",
                "savings_rate": savings_rate,
                "super_guarantee_rate": sg_rate,
                "super_contributions_tax": sg_tax, **au},
               f"Both features together: {savings_rate + net:.1%} reaching "
               f"the portfolio, and a pension withdrawn against it."),
    ) + ((
        System("australia_non_homeowner",
               "Australia as legislated, non-homeowner thresholds",
               {"social_security_formula": "means_tested",
                "savings_rate": savings_rate,
                "super_guarantee_rate": sg_rate,
                "super_contributions_tax": sg_tax, **au,
                "pension_free_area": float(renter)},
               "The higher free area a retiree without a house is allowed, "
               "which is the closer fit to a model with no housing in it."),
    ) if renter is not None else ())


def specs(spec: Any, systems: Sequence[System]) -> Dict[str, Any]:
    """One :class:`~src.lifecycle.LifecycleSpec` per regime."""
    return {s.key: dataclasses.replace(spec, **dict(s.overrides))
            for s in systems}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def entitlement(outcome: Any, spec: Any) -> pd.DataFrame:
    """Where each retirement year falls on the taper, year by year.

    A means test that never binds is a flat pension wearing a disguise, and a
    means test that always binds to zero is no pension at all. Neither would
    tell a reader anything, so the shares below are the first thing to look
    at: they say whether the schedule is doing any work on this population.
    """
    wealth = outcome.wealth[:, spec.n_working:spec.horizon]
    paid = spec.means_tested_benefit(wealth)
    economy_average = float(spec.deterministic_income().mean())
    full = spec.pension_full_rate * economy_average
    rows: List[Dict[str, Any]] = []
    for j in range(wealth.shape[1]):
        column = paid[:, j]
        rows.append({
            "age": int(spec.age_retire + j),
            "share_full_rate": float((column >= full - 1e-12).mean()),
            "share_part_rate": float(((column > 1e-12)
                                      & (column < full - 1e-12)).mean()),
            "share_no_pension": float((column <= 1e-12).mean()),
            "mean_pension_x_earnings": float(column.mean() / economy_average),
        })
    return pd.DataFrame.from_records(rows)


def replacement(outcome: Any, spec: Any) -> Dict[str, float]:
    """How much of retirement consumption the public pension supplies."""
    consumption = outcome.consumption[:, spec.retirement_slice]
    benefit = np.asarray(outcome.social_security, dtype=float)
    mean_consumption = float(consumption.mean())
    return {
        "mean_pension": float(benefit.mean()),
        "mean_retirement_consumption": mean_consumption,
        "pension_share_of_consumption": (float(benefit.mean()) / mean_consumption
                                         if mean_consumption > 0 else float("nan")),
    }


# ---------------------------------------------------------------------------
# Reading the sweep
# ---------------------------------------------------------------------------
def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def gap_table(frame: pd.DataFrame, pair: Tuple[str, str],
              baseline: str = "us_social_security") -> pd.DataFrame:
    """The headline lead, the ranking, and the *level*, under each regime.

    Two different questions live in this table and they have different
    answers. The ``gap_pct`` column asks whether the pension system changes
    which portfolio an investor should hold. The ``cec_lift_pct`` column asks
    whether it changes how much retirement consumption they end up with. A
    system can compress the first while raising the second, and Australia's
    does exactly that: the means test narrows the distance between portfolios
    while the Superannuation Guarantee raises all of them.
    """
    cec = _cec_column(frame)
    rows: List[Dict[str, Any]] = []
    for key in frame["system"].unique():
        block = frame[frame["system"] == key]
        values = {r["strategy"]: float(r[cec]) for _, r in block.iterrows()}
        ordered = sorted(values, key=values.get, reverse=True)
        rows.append({
            "system": key,
            "label": str(block["system_label"].iloc[0]),
            "gap_pct": (values[pair[0]] / values[pair[1]] - 1.0) * 100.0
            if pair[0] in values and pair[1] in values else float("nan"),
            f"cec_{pair[0]}": values.get(pair[0], float("nan")),
            f"cec_{pair[1]}": values.get(pair[1], float("nan")),
            "winner": ordered[0] if ordered else "",
            "ranking": "|".join(ordered),
            "prob_ruin": float(block[block["strategy"] == pair[0]]
                               ["prob_ruin"].iloc[0])
            if (block["strategy"] == pair[0]).any() else float("nan"),
            "mean_consumption": float(
                block[block["strategy"] == pair[1]]
                ["mean_retirement_consumption"].iloc[0])
            if ("mean_retirement_consumption" in block.columns
                and (block["strategy"] == pair[1]).any()) else float("nan"),
            "p5_consumption": float(
                block[block["strategy"] == pair[1]]
                ["p5_retirement_consumption"].iloc[0])
            if ("p5_retirement_consumption" in block.columns
                and (block["strategy"] == pair[1]).any()) else float("nan"),
        })
    out = pd.DataFrame.from_records(rows)
    if baseline in set(out["system"]):
        anchor = float(out.loc[out["system"] == baseline, "gap_pct"].iloc[0])
        out["shift_pp"] = out["gap_pct"] - anchor
        # The level question, asked of the strategy the paper is about.
        level = f"cec_{pair[1]}"
        base_level = float(out.loc[out["system"] == baseline, level].iloc[0])
        out["cec_lift_pct"] = (out[level] / base_level - 1.0) * 100.0
        # And of whichever strategy each system's own investor would pick,
        # since a system that reorders the menu should be judged on what it
        # leads its investor to hold rather than on what another system's
        # investor holds.
        best = [float(frame[(frame["system"] == k)][_cec_column(frame)].max())
                for k in out["system"]]
        out["cec_best_available"] = best
        # Anchor on the baseline row by name. Relying on it being row zero
        # would silently rescale every lift in this table if the sweep were
        # ever reordered.
        at_baseline = out["system"] == baseline
        anchor_best = float(out.loc[at_baseline, "cec_best_available"].iloc[0])
        out["best_lift_pct"] = (out["cec_best_available"]
                                / max(anchor_best, 1e-12) - 1.0) * 100.0
        # A system can raise the average and lower the certainty equivalent at
        # the same time, by trading a guaranteed floor for a bigger portfolio.
        # Reporting only one of the two would hide the trade rather than
        # describe it.
        for source, target in (("mean_consumption", "mean_lift_pct"),
                               ("p5_consumption", "p5_lift_pct")):
            if source not in out.columns:
                continue
            anchor_level = float(out.loc[at_baseline, source].iloc[0])
            if np.isfinite(anchor_level) and anchor_level != 0.0:
                out[target] = (out[source] / anchor_level - 1.0) * 100.0
    return out


def verdict(gaps: pd.DataFrame, baseline: str = "us_social_security"
            ) -> Dict[str, Any]:
    """What swapping the pension system does, classified from the sweep."""
    if not len(gaps):
        return {"systems": 0}
    base = gaps[gaps["system"] == baseline]
    others = gaps[gaps["system"] != baseline]
    base_gap = float(base["gap_pct"].iloc[0]) if len(base) else float("nan")
    rankings = set(gaps["ranking"])
    winners = set(gaps["winner"])
    rows = gaps.set_index("system")
    tapered = gaps[gaps["system"].isin(
        ("age_pension_matched", "australia_as_legislated",
         "australia_non_homeowner"))]

    def _lift(key: str, column: str = "best_lift_pct") -> float:
        return (float(rows.loc[key, column])
                if key in rows.index and column in gaps.columns
                else float("nan"))

    au_lift = _lift("australia_as_legislated")
    saving_lift = _lift("us_matched_saving")
    pension_lift = _lift("age_pension_matched")
    return {
        "systems": int(len(gaps)),
        "baseline_gap_pct": base_gap,
        "min_gap_pct": float(gaps["gap_pct"].min()),
        "max_gap_pct": float(gaps["gap_pct"].max()),
        "largest_shift_pp": float(others["shift_pp"].abs().max())
        if "shift_pp" in others and len(others) else float("nan"),
        # Only the rows that actually carry a taper. Comparing against every
        # other row would fold in the matched-contribution control, which is
        # not a means test and moves the gap the other way.
        "means_test_narrows": bool(len(tapered) and float(
            tapered["gap_pct"].max()) < base_gap),
        "gap_positive_everywhere": bool((gaps["gap_pct"] > 0.0).all()),
        "winner_ever_changes": bool(len(winners) > 1),
        "winners_seen": sorted(winners),
        "ranking_identical_everywhere": bool(len(rankings) == 1),
        "n_rankings": int(len(rankings)),
        # The level question. A saver under the Australian system reaches
        # retirement with two pots instead of one, and the means test claws
        # some of that back; this is the net of the two.
        "australia_lift_pct": au_lift,
        "australia_delivers_more": bool(np.isfinite(au_lift) and au_lift > 0.0),
        "extra_saving_lift_pct": saving_lift,
        "means_test_lift_pct": pension_lift,
        # And the decomposition of it: what the compulsory contribution buys,
        # against what the pension design costs.
        "saving_explains_more_than_pension_costs": bool(
            np.isfinite(saving_lift) and np.isfinite(pension_lift)
            and saving_lift > abs(pension_lift)),
        # The average and the certainty equivalent are different questions and
        # here they have different answers, which is the finding rather than a
        # complication to be smoothed over.
        "australia_mean_lift_pct": _lift("australia_as_legislated",
                                         "mean_lift_pct"),
        "australia_p5_lift_pct": _lift("australia_as_legislated",
                                       "p5_lift_pct"),
        "australia_raises_the_mean": bool(
            _lift("australia_as_legislated", "mean_lift_pct") > 0.0),
        "mean_and_cec_disagree": bool(
            np.isfinite(au_lift)
            and np.isfinite(_lift("australia_as_legislated", "mean_lift_pct"))
            and (au_lift > 0.0)
            != (_lift("australia_as_legislated", "mean_lift_pct") > 0.0)),
        "non_homeowner_lift_pct": _lift("australia_non_homeowner"),
        "thresholds_change_the_answer": bool(
            np.isfinite(au_lift)
            and np.isfinite(_lift("australia_non_homeowner"))
            and abs(_lift("australia_non_homeowner") - au_lift) > 1.0),
    }
