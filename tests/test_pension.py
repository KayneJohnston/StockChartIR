"""The Australian Age Pension, and what a means test does to a ranking."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import lifecycle as lc
from src import pension as pn


def _spec(**kw) -> lc.LifecycleSpec:
    return dataclasses.replace(lc.LifecycleSpec(), **kw)


class TestCalibration:
    def test_rates_are_multiples_of_average_earnings(self) -> None:
        p = pn.australian_parameters()
        # $31,223 a year against AWOTE of $106,657 is a shade under 30%.
        assert 0.28 < p["pension_full_rate"] < 0.31
        # $321,500 against the same is a shade over three years' earnings.
        assert 2.9 < p["pension_free_area"] < 3.1
        # $3 a fortnight per $1,000 is 7.8% a year.
        assert p["pension_taper"] == pytest.approx(0.078)

    def test_cut_out_matches_the_published_threshold(self) -> None:
        """Rate, free area and taper have to be mutually consistent.

        Services Australia publishes all three, and they over-determine each
        other: the pension has to reach zero at the published cut-off. If an
        indexation update moved one without the others this fails.
        """
        p = pn.australian_parameters()
        cut_out = (p["pension_free_area"]
                   + p["pension_full_rate"] / p["pension_taper"])
        assert cut_out * pn.AWOTE_ANNUAL_AUD == pytest.approx(722_000, rel=0.01)

    def test_from_config_divides_the_statutory_dollars(self) -> None:
        cfg = {"pension": {"awote_annual_aud": 100_000.0,
                           "full_rate_annual_aud": 30_000.0,
                           "free_area_aud": 300_000.0,
                           "taper_per_1000_fortnight": 3.0}}
        p = pn.from_config(cfg)
        assert p["pension_full_rate"] == pytest.approx(0.30)
        assert p["pension_free_area"] == pytest.approx(3.0)
        assert p["pension_taper"] == pytest.approx(0.078)

    def test_sg_net_of_contributions_tax(self) -> None:
        assert pn.SG_RATE * (1 - pn.SG_CONTRIBUTIONS_TAX) == pytest.approx(0.102)

    def test_non_homeowner_cut_out_matches_the_published_threshold(self) -> None:
        p = pn.australian_parameters()
        cut_out = (p["pension_free_area_non_homeowner"]
                   + p["pension_full_rate"] / p["pension_taper"])
        assert cut_out * pn.AWOTE_ANNUAL_AUD == pytest.approx(1_000_500,
                                                              rel=0.01)


class TestMeansTest:
    def test_full_rate_below_the_free_area(self) -> None:
        spec = _spec(social_security_formula="means_tested")
        ea = float(spec.deterministic_income().mean())
        paid = spec.means_tested_benefit(np.array([0.0, ea]))
        assert np.allclose(paid, spec.pension_full_rate * ea)

    def test_zero_above_the_cut_out(self) -> None:
        spec = _spec(social_security_formula="means_tested")
        ea = float(spec.deterministic_income().mean())
        cut_out = (spec.pension_free_area
                   + spec.pension_full_rate / spec.pension_taper) * ea
        assert spec.means_tested_benefit(np.array([cut_out * 1.5])) == 0.0

    def test_tapers_linearly_in_between(self) -> None:
        spec = _spec(social_security_formula="means_tested")
        ea = float(spec.deterministic_income().mean())
        base = spec.pension_free_area * ea
        step = 0.5 * ea
        a, b, c = spec.means_tested_benefit(
            np.array([base, base + step, base + 2 * step]))
        assert (a - b) == pytest.approx(b - c)
        assert (a - b) == pytest.approx(spec.pension_taper * step)

    def test_zero_taper_is_a_universal_flat_pension(self) -> None:
        """The control the study leans on: same rate, no assets test."""
        spec = _spec(social_security_formula="means_tested", pension_taper=0.0)
        ea = float(spec.deterministic_income().mean())
        paid = spec.means_tested_benefit(np.array([0.0, 100.0 * ea]))
        assert np.allclose(paid, spec.pension_full_rate * ea)

    def test_disabled_pays_nothing(self) -> None:
        spec = _spec(social_security_formula="means_tested",
                     social_security_enabled=False)
        assert np.all(spec.means_tested_benefit(np.array([0.0, 1.0])) == 0.0)

    def test_career_average_signature_returns_the_maximum(self) -> None:
        """A flat pension has nothing to say about career earnings.

        Callers that solve schedules ask for the benefit from career average
        income; under a means test the only well-defined answer there is the
        maximum rate, and it must not silently depend on earnings.
        """
        spec = _spec(social_security_formula="means_tested")
        paid = spec.social_security_benefit(np.array([0.5, 1.0, 4.0]))
        assert len(set(np.round(paid, 12))) == 1


class TestSystems:
    def test_the_control_isolates_the_means_test(self) -> None:
        systems = {s.key: s for s in pn.default_systems(0.10)}
        untested = systems["age_pension_untested"].overrides
        tested = systems["age_pension_matched"].overrides
        assert untested["pension_taper"] == 0.0
        assert tested["pension_taper"] > 0.0
        # Everything except the taper is held: same rate, same saving.
        assert untested["savings_rate"] == tested["savings_rate"]
        assert untested["pension_full_rate"] == tested["pension_full_rate"]
        assert untested["super_guarantee_rate"] == \
            tested["super_guarantee_rate"]

    def test_the_design_crosses_pension_against_contribution(self) -> None:
        """Both features move between America and Australia, so the sweep has
        to hold each one still while the other varies."""
        built = pn.specs(lc.LifecycleSpec(),
                         pn.default_systems(0.10)).items()
        cells = {k: (v.social_security_formula, v.super_guarantee_rate > 0.0)
                 for k, v in built}
        assert cells["us_social_security"] == ("progressive", False)
        assert cells["us_matched_saving"] == ("progressive", True)
        assert cells["age_pension_matched"] == ("means_tested", False)
        assert cells["australia_as_legislated"] == ("means_tested", True)

    def test_the_guarantee_sits_on_top_of_voluntary_saving(self) -> None:
        """The correction that matters: SG is additional, not instead of.

        An employer contribution does not replace what the worker saves, so
        the Australian saver contributes both. Modelling it as a larger
        savings rate would understate the portfolio by the whole voluntary
        share.
        """
        built = pn.specs(lc.LifecycleSpec(), pn.default_systems(0.10))
        au = built["australia_as_legislated"]
        assert au.savings_rate == pytest.approx(0.10)
        assert au.super_guarantee_rate == pytest.approx(0.12)
        assert au.super_net_rate == pytest.approx(0.102)
        assert au.total_contribution_rate == pytest.approx(0.202)

    def test_specs_apply_the_overrides(self) -> None:
        systems = pn.default_systems(0.10)
        built = pn.specs(lc.LifecycleSpec(), systems)
        assert set(built) == {s.key for s in systems}
        assert built["us_social_security"].social_security_formula == "progressive"
        assert built["us_social_security"].super_guarantee_rate == 0.0

    def test_the_non_homeowner_row_only_moves_the_free_area(self) -> None:
        built = pn.specs(lc.LifecycleSpec(), pn.default_systems(0.10))
        a, b = built["australia_as_legislated"], built["australia_non_homeowner"]
        assert b.pension_free_area > a.pension_free_area
        assert b.pension_full_rate == a.pension_full_rate
        assert b.pension_taper == a.pension_taper
        assert b.total_contribution_rate == a.total_contribution_rate

    def test_every_system_builds_a_valid_spec(self) -> None:
        for spec in pn.specs(lc.LifecycleSpec(), pn.default_systems(0.10)).values():
            assert 0.0 <= spec.savings_rate < 1.0
            assert 0.0 <= spec.super_guarantee_rate < 1.0


class TestReading:
    def _frame(self, gaps) -> pd.DataFrame:
        rows = []
        for system, (a, b) in gaps.items():
            rows.append({"system": system, "system_label": system,
                         "strategy": "international_equity", "label": "i",
                         "cec_crra_gamma5": a, "prob_ruin": 0.1})
            rows.append({"system": system, "system_label": system,
                         "strategy": "balanced_all_equity", "label": "b",
                         "cec_crra_gamma5": b, "prob_ruin": 0.1})
        return pd.DataFrame.from_records(rows)

    def test_gap_table_measures_against_the_baseline(self) -> None:
        frame = self._frame({"us_social_security": (1.10, 1.00),
                             "age_pension_matched": (1.00, 1.00)})
        out = pn.gap_table(frame, ("international_equity",
                                   "balanced_all_equity"))
        assert float(out.loc[out["system"] == "us_social_security",
                             "gap_pct"].iloc[0]) == pytest.approx(10.0)
        assert float(out.loc[out["system"] == "age_pension_matched",
                             "shift_pp"].iloc[0]) == pytest.approx(-10.0)

    def test_verdict_measures_narrowing_on_the_tapered_rows_only(self) -> None:
        """The matched-contribution control is not a means test.

        It widens the gap, so folding it into the comparison would report
        "the means test does not narrow" whenever the extra saving is in the
        sweep -- which is always.
        """
        frame = self._frame({"us_social_security": (1.10, 1.00),
                             "us_matched_saving": (1.20, 1.00),
                             "age_pension_matched": (1.02, 1.00)})
        found = pn.verdict(pn.gap_table(
            frame, ("international_equity", "balanced_all_equity")))
        assert found["means_test_narrows"]
        assert found["gap_positive_everywhere"]
        assert not found["winner_ever_changes"]

    def test_verdict_sees_a_reversal(self) -> None:
        frame = self._frame({"us_social_security": (1.10, 1.00),
                             "age_pension_matched": (0.95, 1.00)})
        found = pn.verdict(pn.gap_table(
            frame, ("international_equity", "balanced_all_equity")))
        assert not found["gap_positive_everywhere"]
        assert found["winner_ever_changes"]
        assert not found["ranking_identical_everywhere"]

    def test_verdict_on_an_empty_frame(self) -> None:
        assert pn.verdict(pd.DataFrame())["systems"] == 0


class TestLiftsAnchorOnTheBaseline:
    """Every "vs US" column has to be measured against the US row by name.

    Anchoring on row zero would rescale the whole table if the sweep were
    ever reordered, and the reordering would look like a result.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        rows = []
        levels = {"age_pension_matched": (0.5, 0.5, 1.0, 0.4),
                  "us_social_security": (1.0, 1.0, 2.0, 0.8)}
        for system, (a, b, mean, p5) in levels.items():
            for strategy, cec in (("international_equity", a),
                                  ("balanced_all_equity", b)):
                rows.append({"system": system, "system_label": system,
                             "strategy": strategy, "label": strategy,
                             "cec_crra_gamma5": cec, "prob_ruin": 0.1,
                             "mean_retirement_consumption": mean,
                             "p5_retirement_consumption": p5})
        return pd.DataFrame.from_records(rows)

    def test_baseline_row_is_zero_even_when_listed_second(self) -> None:
        out = pn.gap_table(self._frame(),
                           ("international_equity", "balanced_all_equity"))
        at_us = out["system"] == "us_social_security"
        assert float(out.loc[at_us, "best_lift_pct"].iloc[0]) == pytest.approx(0.0)
        assert float(out.loc[at_us, "mean_lift_pct"].iloc[0]) == pytest.approx(0.0)
        assert float(out.loc[at_us, "p5_lift_pct"].iloc[0]) == pytest.approx(0.0)

    def test_the_other_row_is_measured_against_it(self) -> None:
        out = pn.gap_table(self._frame(),
                           ("international_equity", "balanced_all_equity"))
        at_au = out["system"] == "age_pension_matched"
        assert float(out.loc[at_au, "best_lift_pct"].iloc[0]) == pytest.approx(-50.0)
        assert float(out.loc[at_au, "mean_lift_pct"].iloc[0]) == pytest.approx(-50.0)


class TestIncomeTaxScale:
    """One ratio, not a tax system. The scale is here only to reconcile the
    two contribution bases: super is a share of pre-tax earnings, voluntary
    saving a share of take-home pay."""

    def test_no_tax_below_the_free_threshold(self) -> None:
        first = pn.TAX_BRACKETS[1][0]
        assert pn.income_tax(first, levy=0.0) == pytest.approx(0.0)

    def test_tax_rises_with_income(self) -> None:
        owed = [pn.income_tax(x) for x in (30_000, 60_000, 120_000, 250_000)]
        assert owed == sorted(owed)

    def test_the_average_rate_stays_below_the_top_marginal(self) -> None:
        """A progressive scale cannot charge its top rate on the whole of an
        income, which is exactly why the *average* rate is the right one to
        gross a contribution up by."""
        top = pn.TAX_BRACKETS[-1][1] + pn.MEDICARE_LEVY
        for income in (50_000, 106_657.20, 300_000):
            assert 0.0 < pn.average_tax_rate(income) < top

    def test_the_rate_on_average_earnings_is_plausible(self) -> None:
        rate = pn.average_tax_rate(pn.AWOTE_ANNUAL_AUD)
        assert 0.20 < rate < 0.28

    def test_the_levy_adds_flatly(self) -> None:
        income = 80_000.0
        assert pn.income_tax(income) - pn.income_tax(income, levy=0.0) == \
            pytest.approx(income * pn.MEDICARE_LEVY)

    def test_zero_income_is_not_a_division_by_zero(self) -> None:
        assert pn.average_tax_rate(0.0) == 0.0
