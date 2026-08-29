"""Tests for the mortgage on the housing sleeve.

The load-bearing claims are that the levered return is the textbook one, that
leverage touches housing and nothing else, and that the solved schedule is not
described beyond what a deviation profile supports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import bootstrap as bs
from src import glidepath as gp
from src import housing as hs
from src import leverage as lev
from src import lifecycle as lc
from src import mortgage as mg


@pytest.fixture()
def levered_setup(toy_panel, toy_config):
    """A five-asset panel with a known, constant housing return."""
    housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.08)
    restricted = hs.restrict_to_housing(toy_panel, housing)
    spec = lc.spec_from_config(toy_config)
    sampler = bs.from_config(restricted, toy_config, horizon_years=spec.horizon)
    paths = sampler.sample(200, chunk_size=100)
    income = lc.simulate_income(spec, paths.n_paths, np.random.default_rng(1))
    return paths, spec, income, hs.gather(paths, housing)


def _evaluator(setup, cfg, **kwargs):
    paths, spec, income, gross = setup
    return mg.MortgageEvaluator(paths, spec, income, cfg,
                                extra={hs.HOUSING: gross}, **kwargs)


class TestLeverageMultiple:
    def test_maps_lvr_to_gross_exposure(self) -> None:
        assert mg.leverage_multiple(0.0) == pytest.approx(1.0)
        assert mg.leverage_multiple(0.5) == pytest.approx(2.0)
        assert mg.leverage_multiple(0.8) == pytest.approx(5.0)

    def test_rejects_an_impossible_ratio(self) -> None:
        with pytest.raises(ValueError, match="loan-to-value"):
            mg.leverage_multiple(1.0)
        with pytest.raises(ValueError, match="loan-to-value"):
            mg.leverage_multiple(-0.1)


class TestLeveredReturn:
    def test_matches_the_closed_form(self, levered_setup, toy_config) -> None:
        """``r_E = (r_H - lambda*i) / (1 - lambda)``."""
        ev = _evaluator(levered_setup, toy_config, spread=0.02, lvr=0.6)
        expected = np.maximum(
            (ev._housing - 0.6 * (ev._base_rate + 0.02)) / 0.4, -1.0)
        np.testing.assert_allclose(ev.levered_housing()[:, :, 0], expected)

    def test_zero_lvr_is_the_unlevered_return(self, levered_setup,
                                              toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config, spread=0.03, lvr=0.0)
        np.testing.assert_allclose(ev.levered_housing()[:, :, 0], ev._housing)

    def test_a_free_loan_still_costs_the_short_rate(self, levered_setup,
                                                    toy_config) -> None:
        """Spread zero is not a zero interest rate: the bill rate is real."""
        ev = _evaluator(levered_setup, toy_config, spread=0.0, lvr=0.5)
        expected = np.maximum((ev._housing - 0.5 * ev._base_rate) / 0.5, -1.0)
        np.testing.assert_allclose(ev.levered_housing()[:, :, 0], expected)

    def test_equity_is_wiped_out_not_driven_negative(self, toy_panel,
                                                     toy_config) -> None:
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), -0.90)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(100, chunk_size=100)
        income = lc.simulate_income(spec, 100, np.random.default_rng(1))
        ev = mg.MortgageEvaluator(paths, spec, income, toy_config,
                                  extra={hs.HOUSING: hs.gather(paths, housing)},
                                  spread=0.02, lvr=0.8)
        assert (ev.levered_housing() >= lev.MIN_RETURN - 1e-12).all()
        assert ev.negative_equity_frequency()[0] > 0.5, (
            "a 90% fall at 80% LVR must wipe the equity out"
        )


class TestLeverageTouchesHousingOnly:
    def test_a_portfolio_without_housing_is_unaffected_by_lvr(
            self, levered_setup, toy_config) -> None:
        """The decisive isolation test."""
        ev = _evaluator(levered_setup, toy_config, spread=0.02)
        spec = ev.spec
        weights = np.tile(np.array([0.25, 0.25, 0.25, 0.25, 0.0]),
                          (1, spec.horizon, 1))
        ev.set_lvr(0.0)
        low = ev.cec(weights, 5.0)[0]
        ev.set_lvr(0.8)
        high = ev.cec(weights, 5.0)[0]
        assert low == pytest.approx(high), (
            "with no housing there is nothing to mortgage"
        )

    def test_a_pure_housing_portfolio_earns_the_levered_return(
            self, levered_setup, toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config, spread=0.02, lvr=0.5)
        spec = ev.spec
        weights = np.tile(np.array([0., 0., 0., 0., 1.]),
                          (1, spec.horizon, 1))
        np.testing.assert_allclose(ev._portfolio_returns(weights)[:, :, 0],
                                   ev.levered_housing()[:, :, 0])

    def test_matches_the_unlevered_evaluator_at_zero_lvr(self, levered_setup,
                                                          toy_config) -> None:
        """At LVR zero the mortgage machinery must be a no-op."""
        paths, spec, income, gross = levered_setup
        plain = gp.BatchEvaluator(paths, spec, income, toy_config,
                                  assets=hs.ASSETS,
                                  extra={hs.HOUSING: gross})
        ev = _evaluator(levered_setup, toy_config, spread=0.02, lvr=0.0)
        weights = np.tile(np.array([0.2] * 5), (1, spec.horizon, 1))
        np.testing.assert_allclose(ev.cec(weights, 5.0),
                                   plain.cec(weights, 5.0), rtol=1e-12)


class TestTheCap:
    def test_rejects_a_ratio_above_the_cap(self, levered_setup,
                                           toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config)
        with pytest.raises(ValueError, match="cap"):
            ev.set_lvr(0.85)

    def test_rejects_a_negative_ratio(self, levered_setup,
                                      toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config)
        with pytest.raises(ValueError, match="deposit"):
            ev.set_lvr(-0.1)

    def test_accepts_a_per_age_schedule(self, levered_setup,
                                        toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config)
        schedule = np.linspace(0.8, 0.0, ev.spec.horizon)
        ev.set_lvr(schedule)
        np.testing.assert_allclose(ev.lvr, schedule)

    def test_rejects_a_schedule_of_the_wrong_length(self, levered_setup,
                                                    toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config)
        with pytest.raises(ValueError, match="lvr must be"):
            ev.set_lvr(np.zeros(3))


class TestRateBase:
    def test_refuses_to_price_a_loan_off_a_bond_total_return(
            self, levered_setup, toy_config) -> None:
        """The mistake this option used to make.

        A bond total return adds the capital gain a *holder* makes when yields
        fall. Charging a borrower that would make their loan cheap in exactly
        the years bonds rallied, which is backwards.
        """
        with pytest.raises(ValueError, match="total"):
            _evaluator(levered_setup, toy_config, rate_base="bond")

    def test_the_long_yield_variant_needs_the_yield_supplied(
            self, levered_setup, toy_config) -> None:
        with pytest.raises(ValueError, match="base_rate"):
            _evaluator(levered_setup, toy_config, rate_base="long_yield")

    def test_an_explicit_base_rate_is_what_the_borrower_pays(
            self, levered_setup, toy_config) -> None:
        paths, spec, income, gross = levered_setup
        rate = np.full((paths.n_paths, paths.horizon), 0.03)
        ev = mg.MortgageEvaluator(
            paths, spec, income, toy_config, extra={hs.HOUSING: gross},
            spread=0.01, lvr=0.5, rate_base="long_yield", base_rate=rate)
        expected = np.maximum((ev._housing - 0.5 * (0.03 + 0.01)) / 0.5, -1.0)
        np.testing.assert_allclose(ev.levered_housing()[:, :, 0], expected)

    def test_rejects_a_base_rate_with_holes(self, levered_setup,
                                            toy_config) -> None:
        paths, spec, income, gross = levered_setup
        rate = np.full((paths.n_paths, paths.horizon), 0.03)
        rate[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            mg.MortgageEvaluator(
                paths, spec, income, toy_config, extra={hs.HOUSING: gross},
                rate_base="long_yield", base_rate=rate)

    def test_rejects_an_unknown_base(self, levered_setup,
                                     toy_config) -> None:
        with pytest.raises(ValueError, match="rate_base"):
            _evaluator(levered_setup, toy_config, rate_base="gold")

    def test_the_default_spread_is_inside_the_evidence_range(self) -> None:
        """150-200bp is what a competitive borrower pays; the default is 200."""
        assert 0.015 <= mg.DEFAULT_SPREAD <= 0.020


class TestSearch:
    def test_finds_the_corner_when_borrowing_is_free_money(
            self, toy_panel, toy_config) -> None:
        """Housing returns far more than the loan costs: lever to the cap."""
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.40)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, 200, np.random.default_rng(1))
        ev = mg.MortgageEvaluator(paths, spec, income, toy_config,
                                  extra={hs.HOUSING: hs.gather(paths, housing)},
                                  spread=0.0)
        weights = np.tile(np.array([0., 0., 0., 0., 1.]), (spec.horizon, 1))
        best, _, curve = mg.best_constant_lvr(ev, weights, 5.0)
        assert best == pytest.approx(mg.LVR_CAP)
        assert curve["cec"].is_monotonic_increasing

    def test_refuses_to_borrow_when_the_loan_costs_more_than_the_house_earns(
            self, toy_panel, toy_config) -> None:
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.005)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, 200, np.random.default_rng(1))
        ev = mg.MortgageEvaluator(paths, spec, income, toy_config,
                                  extra={hs.HOUSING: hs.gather(paths, housing)},
                                  spread=0.10)
        weights = np.tile(np.array([0., 0., 0., 0., 1.]), (spec.horizon, 1))
        best, _, _ = mg.best_constant_lvr(ev, weights, 5.0)
        assert best == pytest.approx(0.0)

    def test_the_schedule_search_never_loses_to_the_flat_one(
            self, levered_setup, toy_config) -> None:
        """Seeded at the best constant ratio, so it cannot report a loss."""
        ev = _evaluator(levered_setup, toy_config, spread=0.01)
        spec = ev.spec
        weights = np.tile(np.array([0., 0.5, 0., 0., 0.5]), (spec.horizon, 1))
        flat, flat_cec, _ = mg.best_constant_lvr(ev, weights, 5.0)
        _, solved_cec, _ = mg.optimise_lvr_schedule(ev, weights, 5.0, sweeps=1)
        assert solved_cec >= flat_cec - 1e-12

    def test_cec_over_lvr_agrees_with_setting_each_schedule_directly(
            self, levered_setup, toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config, spread=0.02)
        spec = ev.spec
        weights = np.tile(np.array([0., 0.5, 0., 0., 0.5]), (spec.horizon, 1))
        schedules = np.array([np.full(spec.horizon, v)
                              for v in (0.0, 0.3, 0.6)])
        batched = ev.cec_over_lvr(weights, schedules, 5.0)
        one_at_a_time = []
        for row in schedules:
            ev.set_lvr(row)
            one_at_a_time.append(ev.cec(weights[None], 5.0)[0])
        np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-12)


class TestDiagnostics:
    def test_the_deviation_profile_flags_a_flat_surface(self, levered_setup,
                                                         toy_config) -> None:
        """A schedule that is already flat costs nothing to reset."""
        ev = _evaluator(levered_setup, toy_config, spread=0.02)
        spec = ev.spec
        weights = np.tile(np.array([0., 0.5, 0., 0., 0.5]), (spec.horizon, 1))
        flat = np.full(spec.horizon, 0.4)
        profile = mg.lvr_deviation_profile(ev, weights, flat, 5.0, spec)
        assert profile["cost_of_resetting_bp"].abs().max() < 1e-6
        assert mg.profile_summary(profile)["material_ages"] == 0

    def test_the_deviation_profile_prices_a_real_choice(self, levered_setup,
                                                         toy_config) -> None:
        ev = _evaluator(levered_setup, toy_config, spread=0.02)
        spec = ev.spec
        weights = np.tile(np.array([0., 0., 0., 0., 1.]), (spec.horizon, 1))
        schedule = np.full(spec.horizon, 0.4)
        schedule[0] = 0.0
        profile = mg.lvr_deviation_profile(ev, weights, schedule, 5.0, spec)
        assert profile["cost_of_resetting_bp"].abs().max() > 0.0

    def test_the_terminal_option_check_catches_a_late_spike(self) -> None:
        class Spec:
            n_working = 40
            horizon = 68
        schedule = np.concatenate([np.full(63, 0.2), np.full(5, 0.8)])
        out = mg.terminal_option_check(schedule, Spec(), tail_years=5)
        assert out["looks_like_the_option"]
        assert out["terminal_lift"] == pytest.approx(0.6)

    def test_it_stays_quiet_on_a_declining_schedule(self) -> None:
        class Spec:
            n_working = 40
            horizon = 68
        schedule = np.linspace(0.8, 0.0, 68)
        out = mg.terminal_option_check(schedule, Spec(), tail_years=5)
        assert not out["looks_like_the_option"]
        assert out["terminal_lift"] < 0.0


class TestBreakEven:
    def test_finds_the_spread_where_borrowing_stops(self) -> None:
        frame = pd.DataFrame({"spread": [0.0, 0.02, 0.04],
                              "mean_lvr": [0.60, 0.30, 0.00]})
        assert mg.break_even_spread(frame) == pytest.approx(0.0393, abs=1e-4)

    def test_undefined_when_the_sweep_never_crosses(self) -> None:
        frame = pd.DataFrame({"spread": [0.0, 0.02],
                              "mean_lvr": [0.60, 0.40]})
        assert np.isnan(mg.break_even_spread(frame))
