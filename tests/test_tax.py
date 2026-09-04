"""Tests for retirement income tax under the two systems Section 33 compares.

The point of most of these is that a tax system is easy to get subtly wrong
and impossible to notice: a threshold off by a factor, an offset applied as
a deduction, an inclusion share applied as a rate. Each of those produces a
plausible-looking number, so the checks below are against worked cases and
against structural properties that must hold whatever the rates are.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import tax as tx


AWOTE = 106_657.20
AWI = tx.US_AVERAGE_WAGE


class TestScale:
    def test_no_tax_below_the_first_rated_bracket(self) -> None:
        below = tx.AU_SCALE.brackets[1][0] * AWOTE * 0.5
        assert tx.AU_SCALE.tax(np.array([below]), AWOTE)[0] == \
            pytest.approx(0.0)

    def test_the_australian_scale_matches_a_hand_calculation(self) -> None:
        """$106,657.20: 16% of the second bracket, 30% of the rest, plus 2%."""
        expected = ((45_000 - 18_200) * 0.16
                    + (AWOTE - 45_000) * 0.30 + AWOTE * 0.02)
        assert tx.AU_SCALE.tax(np.array([AWOTE]), AWOTE)[0] == \
            pytest.approx(expected, rel=1e-9)

    def test_tax_is_monotone_in_income(self) -> None:
        income = np.linspace(0.0, 4.0, 200) * AWOTE
        owed = tx.AU_SCALE.tax(income, AWOTE)
        assert np.all(np.diff(owed) >= -1e-9)

    def test_the_average_rate_never_reaches_the_top_marginal(self) -> None:
        """A progressive scale cannot charge its top rate on the whole of an
        income, however large."""
        top = max(r for _, r in tx.AU_SCALE.brackets) + tx.AU_SCALE.levy
        income = np.array([10.0]) * AWOTE
        assert tx.AU_SCALE.tax(income, AWOTE)[0] / income[0] < top

    def test_the_levy_waits_for_its_threshold(self) -> None:
        just_under = tx.AU_SCALE.levy_threshold * AWOTE * 0.99
        just_over = tx.AU_SCALE.levy_threshold * AWOTE * 1.01
        owed = tx.AU_SCALE.tax(np.array([just_under, just_over]), AWOTE)
        # The levy is charged on the whole income once due, so crossing the
        # threshold costs more than the extra income earned.
        assert owed[1] - owed[0] > (just_over - just_under)

    def test_scaling_average_earnings_scales_the_tax(self) -> None:
        """Thresholds are multiples, so the same relative income is taxed the
        same however the model's units are normalised."""
        a = tx.AU_SCALE.tax(np.array([1.5 * AWOTE]), AWOTE)[0] / AWOTE
        b = tx.AU_SCALE.tax(np.array([1.5]), 1.0)[0]
        assert a == pytest.approx(b)

    def test_negative_income_is_not_taxed(self) -> None:
        assert tx.AU_SCALE.tax(np.array([-5.0]), AWOTE)[0] == \
            pytest.approx(0.0)

    def test_brackets_must_ascend(self) -> None:
        with pytest.raises(ValueError, match="ascend"):
            tx.Scale(brackets=((0.0, 0.1), (2.0, 0.2), (1.0, 0.3)))

    def test_the_scale_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="start at zero"):
            tx.Scale(brackets=((0.5, 0.1),))

    def test_a_rate_of_one_or_more_is_refused(self) -> None:
        with pytest.raises(ValueError, match="marginal rates"):
            tx.Scale(brackets=((0.0, 1.5),))


class TestOffset:
    def test_it_pays_its_maximum_below_the_threshold(self) -> None:
        got = tx.AU_SAPTO.amount(np.array([0.0]), AWOTE)[0]
        assert got == pytest.approx(tx.AU_SAPTO.maximum * AWOTE)

    def test_it_shades_out_and_stops_at_zero(self) -> None:
        far_above = np.array([5.0 * AWOTE])
        assert tx.AU_SAPTO.amount(far_above, AWOTE)[0] == pytest.approx(0.0)

    def test_it_shades_at_the_stated_taper(self) -> None:
        base = tx.AU_SAPTO.threshold * AWOTE
        step = 1_000.0
        pair = tx.AU_SAPTO.amount(np.array([base, base + step]), AWOTE)
        assert pair[0] - pair[1] == pytest.approx(tx.AU_SAPTO.taper * step)


class TestSocialSecurityInclusion:
    """26 U.S.C. 86, checked at both tier boundaries and the cap."""

    @staticmethod
    def _included(benefit: float, other: float) -> float:
        return float(tx.taxable_social_security(
            np.array([benefit]), np.array([other]), AWI)[0])

    def test_a_benefit_alone_below_the_base_is_untaxed(self) -> None:
        assert self._included(20_000, 0) == pytest.approx(0.0)

    def test_the_first_tier_includes_half_the_excess(self) -> None:
        # provisional 30,000 -> half of (30,000 - 25,000)
        assert self._included(20_000, 20_000) == pytest.approx(2_500.0)

    def test_the_first_tier_is_capped_at_half_the_benefit(self) -> None:
        assert self._included(2_000, 30_000) == pytest.approx(1_000.0)

    def test_the_worked_case_across_the_second_threshold(self) -> None:
        # provisional 40,000: 0.85*(40,000-34,000) + min(4,500, 10,000)
        assert self._included(20_000, 30_000) == pytest.approx(9_600.0)

    def test_it_never_exceeds_eighty_five_per_cent(self) -> None:
        for benefit in (5_000, 20_000, 40_000):
            for other in (0, 25_000, 60_000, 500_000):
                assert self._included(benefit, other) <= 0.85 * benefit + 1e-6

    def test_it_is_continuous_at_the_tier_boundary(self) -> None:
        """A jump here would be a real cliff for a real retiree, so a jump in
        the code is a bug rather than a feature of the statute."""
        base = 34_000.0 - 0.5 * 20_000
        pair = [self._included(20_000, base - 1.0),
                self._included(20_000, base + 1.0)]
        assert abs(pair[1] - pair[0]) < 2.0

    def test_more_other_income_never_reduces_the_included_share(self) -> None:
        other = np.linspace(0.0, 200_000.0, 400)
        included = tx.taxable_social_security(
            np.full_like(other, 20_000.0), other, AWI)
        assert np.all(np.diff(included) >= -1e-6)

    def test_no_benefit_means_nothing_to_include(self) -> None:
        assert self._included(0.0, 80_000) == pytest.approx(0.0)


class TestAustralianRetiree:
    """The result that matters: a pensioner pays nothing, and drawing super
    does not change that."""

    AVG = 1.5347
    FULL_PENSION = 0.2927 * 1.5347

    def _tax(self, benefit: float, withdrawal: float) -> float:
        return float(tx.REGIMES["au"].tax(
            np.array([benefit]), np.array([withdrawal]), self.AVG)[0])

    def test_the_full_pension_alone_is_untaxed(self) -> None:
        assert self._tax(self.FULL_PENSION, 0.0) == pytest.approx(0.0)

    def test_drawing_super_does_not_tax_the_pension(self) -> None:
        """Super after 60 is not assessable income, so it cannot raise the
        tax on anything else. This is the exact opposite of the US torpedo
        and is the reason the two systems cannot be compared on rates."""
        for draw in (0.5, 1.0, 5.0):
            assert self._tax(self.FULL_PENSION, draw) == pytest.approx(0.0)

    def test_a_large_assessable_income_is_taxed(self) -> None:
        """The control: SAPTO shades out, so the regime is not simply
        returning zero for everything."""
        assert self._tax(3.0, 0.0) > 0.0

    def test_the_regime_carries_a_fund_tax_model_not_a_rate(self) -> None:
        """The statutory 15% is not what a fund pays, and the difference is
        more than an order of magnitude, so the regime holds the model."""
        fund = tx.REGIMES["au"].fund_tax
        assert fund is not None
        assert fund.rate == pytest.approx(0.15)
        assert abs(fund.drag(0.5)) < 0.15 * 0.05


class TestUnitedStatesRetiree:
    AVG = 1.5347
    BENEFIT = 0.43 * 1.5347

    def _tax(self, key: str, withdrawal: float) -> float:
        return float(tx.REGIMES[key].tax(
            np.array([self.BENEFIT]), np.array([withdrawal]), self.AVG)[0])

    def test_a_roth_saver_pays_nothing_on_this_benefit(self) -> None:
        """With no other income the benefit stays under the frozen base, and
        a Roth withdrawal adds nothing to provisional income."""
        for draw in (0.0, 1.0, 5.0):
            assert self._tax("us_roth", draw) == pytest.approx(0.0)

    def test_a_traditional_saver_pays_and_pays_more_as_they_draw(self
                                                                 ) -> None:
        owed = [self._tax("us_traditional", d) for d in (0.5, 1.0, 2.0)]
        assert owed == sorted(owed)
        assert owed[0] > 0.0

    def test_the_torpedo_beats_the_bracket(self) -> None:
        """The finding this module exists for: the marginal rate the retiree
        faces exceeds the bracket they are nominally in, because withdrawing
        also drags benefit into the tax base."""
        curve = tx.effective_rate_curve(
            tx.REGIMES["us_traditional"], self.BENEFIT, self.AVG,
            np.linspace(0.0, 2.5, 2001))
        assert curve["torpedo_excess"] > 0.0
        assert curve["torpedo_marginal"] > curve["torpedo_statutory"]
        # 1.85 is the arithmetic ceiling: one dollar drawn, 85 cents of
        # benefit dragged in behind it.
        assert curve["torpedo_multiple"] <= 1.85 + 1e-6

    def test_a_roth_saver_faces_no_torpedo(self) -> None:
        curve = tx.effective_rate_curve(
            tx.REGIMES["us_roth"], self.BENEFIT, self.AVG,
            np.linspace(0.0, 2.5, 501))
        assert curve["torpedo_excess"] == pytest.approx(0.0, abs=1e-9)


class TestRegime:
    def test_the_null_regime_taxes_nothing(self) -> None:
        """Every section before Section 33 assumes this, so it has to be
        exactly zero rather than nearly."""
        owed = tx.REGIMES["none"].tax(
            np.array([1.0, 5.0]), np.array([2.0, 9.0]), 1.5)
        assert np.all(owed == 0.0)

    def test_an_unknown_benefit_treatment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="benefit_taxable"):
            tx.Regime(key="x", label="x", scale=tx.AU_SCALE,
                      benefit_taxable="sometimes")

    def test_a_withdrawal_share_outside_zero_to_one_is_refused(self) -> None:
        for share in (-0.5, 1.5):
            with pytest.raises(ValueError, match="withdrawal_taxable"):
                tx.Regime(key="x", label="x", scale=tx.AU_SCALE,
                          withdrawal_taxable=share)

    def test_tax_is_never_negative(self) -> None:
        """Offsets reduce tax owed; they do not refund it."""
        for key in tx.REGIMES:
            owed = tx.REGIMES[key].tax(
                np.array([0.0, 0.1, 1.0]), np.array([0.0, 0.05, 0.5]), 1.5)
            assert np.all(owed >= 0.0)

    def test_config_overrides_reach_the_fund_model(self) -> None:
        got = tx.regime_from_config(
            "au", {"tax": {"au": {"realisation": 0.0}}})
        assert got.fund_tax.realisation == pytest.approx(0.0)
        assert got.scale is tx.AU_SCALE
        # Everything else on the fund model survives the override.
        assert got.fund_tax.rate == pytest.approx(tx.REGIMES["au"].fund_tax.rate)

    def test_an_unknown_override_is_refused_rather_than_ignored(self) -> None:
        with pytest.raises(ValueError, match="unknown tax override"):
            tx.regime_from_config("au", {"tax": {"au": {"rate": 0.1}}})

    def test_an_unknown_regime_names_the_real_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown tax regime"):
            tx.regime_from_config("canada")

    def test_no_config_returns_the_regime_unchanged(self) -> None:
        assert tx.regime_from_config("au") is tx.REGIMES["au"]


class TestFundTax:
    """What a superannuation fund pays while accumulating, which is not the
    statutory rate on the return and is not close to it.

    An earlier version of this project charged 15% of the nominal return and
    reported that it cost an Australian household 16.8% of lifetime
    certainty-equivalent consumption. That was wrong by more than an order of
    magnitude, for three separate reasons, and each gets a test here.
    """

    def test_unrealised_gains_are_not_taxed(self) -> None:
        """The first reason, and the largest. A fund that holds owes nothing
        on appreciation; only what it sells is assessable."""
        held = tx.FundTax(realisation=0.0)
        assert held.gains_drag() == pytest.approx(0.0)

    def test_holding_to_the_pension_phase_leaves_only_dividends(self) -> None:
        """Which is the case a member should hold in mind: earnings in the
        retirement phase are exempt, so a gain carried across that boundary
        is never taxed at all."""
        held = tx.FundTax(realisation=0.0)
        assert held.drag(0.5) == pytest.approx(held.income_drag(0.5))

    def test_the_discount_makes_the_gains_rate_ten_per_cent(self) -> None:
        """The second reason: a gain held beyond a year is discounted by a
        third before the 15% touches it."""
        assert tx.FundTax().capital_gains_rate == pytest.approx(0.10)

    def test_a_franked_dividend_is_a_refund_not_a_charge(self) -> None:
        """The third reason, and the one most easily missed: at a 30% company
        rate against a 15% fund rate the imputation credit exceeds the
        liability, so a domestic dividend *adds* to the fund."""
        assert tx.FundTax().dividend_value(1.0) > 0.0

    def test_an_unfranked_dividend_is_charged_at_the_fund_rate(self) -> None:
        fund = tx.FundTax()
        assert fund.dividend_value(0.0) == pytest.approx(-fund.rate)

    def test_franking_subsidises_the_international_sleeve(self) -> None:
        """A portfolio franked enough comes out ahead on income overall,
        which is why the sign of the whole thing depends on the allocation."""
        fund = tx.FundTax()
        assert fund.income_drag(1.0) > 0.0        # all domestic
        assert fund.income_drag(0.0) < 0.0        # all international
        assert fund.income_drag(0.5) > fund.income_drag(0.0)

    def test_the_drag_is_a_small_fraction_of_the_statutory_rate(self) -> None:
        """The headline check. Charging 15% against a return of roughly 8%
        would cost about 1.2% a year; the truth is an order of magnitude
        smaller, and this is what the earlier version got wrong."""
        fund = tx.FundTax()
        naive = abs(fund.components(0.5)["naive_drag"])
        assert abs(fund.drag(0.5)) < naive / 10.0

    def test_more_realisation_costs_more(self) -> None:
        fund = tx.FundTax()
        drags = [dataclasses.replace(fund, realisation=r).drag(0.5)
                 for r in (0.0, 0.1, 0.25, 0.5)]
        assert drags == sorted(drags, reverse=True)

    def test_the_components_add_to_the_total(self) -> None:
        parts = tx.FundTax().components(0.5)
        assert parts["income_drag"] + parts["gains_drag"] == \
            pytest.approx(parts["total_drag"])

    def test_a_rate_outside_zero_to_one_is_refused(self) -> None:
        for field in ("rate", "cgt_discount", "company_rate", "franked_share",
                      "embedded_gain"):
            with pytest.raises(ValueError, match=field):
                tx.FundTax(**{field: 1.5})

    def test_negative_realisation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="realisation"):
            tx.FundTax(realisation=-0.1)

    def test_a_zero_rate_fund_pays_and_receives_nothing(self) -> None:
        """The control: with no fund tax there is no liability and no credit,
        so the drag is exactly zero however the portfolio is split."""
        free = tx.FundTax(rate=0.0, company_rate=0.0)
        for weight in (0.0, 0.5, 1.0):
            assert free.drag(weight) == pytest.approx(0.0)


class TestWhichSideMovesTheGap:
    """A gap between two systems can widen because one lost or because the
    other gained, and those are different findings. The prose names one, so
    the code has to decide which rather than assume the taxed-more arm did
    the work -- which is the mistake an earlier version made."""

    @staticmethod
    def _frame(us_traditional: float, au: float,
               us_free: float = 1.0409, au_free: float = 0.7223):
        return pd.DataFrame([
            {"system": "us", "regime": "none", "cec": us_free},
            {"system": "us", "regime": "us_roth", "cec": us_free},
            {"system": "us", "regime": "us_traditional",
             "cec": us_free * (1.0 + us_traditional)},
            {"system": "au_as_legislated", "regime": "none", "cec": au_free},
            {"system": "au_as_legislated", "regime": "au",
             "cec": au_free * (1.0 + au)},
        ])

    def test_a_gain_on_one_side_is_credited_to_that_side(self) -> None:
        found = tx.tax_verdict(self._frame(us_traditional=0.02, au=-0.006))
        assert found["driver"] == "us"
        assert found["driver_share"] > 0.5

    def test_a_large_loss_on_the_other_side_flips_the_driver(self) -> None:
        found = tx.tax_verdict(self._frame(us_traditional=0.02, au=-0.168))
        assert found["driver"] == "au"

    def test_the_gap_change_is_signed_and_consistent(self) -> None:
        found = tx.tax_verdict(self._frame(us_traditional=0.02, au=-0.006))
        assert found["gap_change_pp"] == pytest.approx(
            found["gap_taxed_pct"] - found["gap_untaxed_pct"])
        assert not found["gap_narrowed"]

    def test_a_shared_move_gives_neither_side_a_majority(self) -> None:
        found = tx.tax_verdict(self._frame(us_traditional=0.02, au=-0.02))
        assert found["driver_share"] == pytest.approx(0.5)
