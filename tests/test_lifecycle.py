"""Tests for the accumulation/decumulation simulator."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src import bootstrap as bs
from src import lifecycle as lc


def constant_paths(n_paths: int, horizon: int, dom_eq: float = 0.05,
                   intl_eq: float = 0.05, bond: float = 0.02,
                   bill: float = 0.01) -> bs.BootstrapPaths:
    """Deterministic return paths, so wealth can be checked by hand."""
    def block(value: float) -> np.ndarray:
        return np.full((n_paths, horizon), value)
    return bs.BootstrapPaths(
        dom_eq=block(dom_eq), intl_eq=block(intl_eq), bond=block(bond),
        bill=block(bill), inflation=block(0.0),
        domestic_country=np.zeros((n_paths, horizon), dtype=np.int32),
        calendar_index=np.zeros((n_paths, horizon), dtype=np.int32),
        block_id=np.zeros((n_paths, horizon), dtype=np.int32),
    )


class TestLifecycleSpec:
    def test_derived_lengths(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=63, age_death=93)
        assert (spec.horizon, spec.n_working, spec.n_retired) == (68, 38, 30)
        assert spec.retirement_slice == slice(38, 68)
        assert spec.ages[0] == 25 and spec.ages[-1] == 92

    def test_rejects_out_of_order_ages(self) -> None:
        with pytest.raises(ValueError, match="start < retire < death"):
            lc.LifecycleSpec(age_start=65, age_retire=63, age_death=93)

    def test_rejects_impossible_savings_rate(self) -> None:
        with pytest.raises(ValueError, match="savings_rate"):
            lc.LifecycleSpec(savings_rate=1.5)

    def test_rejects_unknown_rules(self) -> None:
        with pytest.raises(ValueError, match="retirement_rule"):
            lc.LifecycleSpec(retirement_rule="spend_it_all")
        with pytest.raises(ValueError, match="social_security_formula"):
            lc.LifecycleSpec(social_security_formula="whatever_they_promise")

    def test_income_profile_is_hump_shaped(self) -> None:
        profile = lc.LifecycleSpec().deterministic_income()
        peak = int(np.argmax(profile))
        assert 0 < peak < profile.size - 1
        assert profile[0] == pytest.approx(1.0)
        assert profile.max() / profile[0] > 1.4


class TestSocialSecurity:
    def test_progressive_schedule_is_concave_and_monotone(self) -> None:
        spec = lc.LifecycleSpec()
        earnings = np.linspace(0.05, 5.0, 60)
        benefit = spec.social_security_benefit(earnings)
        assert np.all(np.diff(benefit) > 0)
        assert np.all(np.diff(np.diff(benefit)) <= 1e-12)

    def test_replacement_rate_falls_with_earnings(self) -> None:
        spec = lc.LifecycleSpec()
        earnings = np.array([0.1, 0.5, 1.0, 3.0])
        rates = spec.social_security_benefit(earnings) / earnings
        assert np.all(np.diff(rates) < 0)

    def test_low_earners_get_the_ninety_percent_tranche(self) -> None:
        spec = lc.LifecycleSpec()
        low = np.array([0.01])
        assert spec.social_security_benefit(low)[0] == pytest.approx(0.009)

    def test_flat_formula_uses_the_replacement_rate(self) -> None:
        spec = lc.LifecycleSpec(social_security_formula="flat",
                                replacement_rate=0.4)
        earnings = np.array([2.0])
        assert spec.social_security_benefit(earnings)[0] == pytest.approx(0.8)

    def test_disabled_returns_zero(self) -> None:
        spec = lc.LifecycleSpec(social_security_enabled=False)
        assert spec.social_security_benefit(np.array([1.0]))[0] == 0.0


class TestStrategies:
    def test_weights_sum_to_one(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        for strat in lc.build_strategies(toy_config, spec).values():
            np.testing.assert_allclose(strat.weights.sum(axis=1), 1.0)

    def test_glide_path_interpolates_the_configured_knots(self, toy_config
                                                          ) -> None:
        spec = lc.spec_from_config(toy_config)
        glide = lc.build_strategies(toy_config, spec)["glide"]
        equity = glide.equity_share()
        assert equity[0] == pytest.approx(0.9)
        assert equity[-1] == pytest.approx(0.3 + (0.5 - 0.3) * 1 / 6, abs=0.05)
        assert np.all(np.diff(equity) <= 1e-12)

    def test_glide_splits_equity_and_fixed_income(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        glide = lc.build_strategies(toy_config, spec)["glide"]
        equity = glide.equity_share()[0]
        assert glide.weights[0, 0] == pytest.approx(equity * 0.6)
        assert glide.weights[0, 1] == pytest.approx(equity * 0.4)
        assert glide.weights[0, 2] == pytest.approx((1 - equity) * 0.7)

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            lc.Strategy("bad", "Bad", np.tile([0.5, 0.2, 0.1, 0.1], (3, 1)))

    def test_rejects_unknown_strategy_type(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        cfg = dict(toy_config)
        cfg["strategies"] = {"x": {"label": "X", "type": "momentum"}}
        with pytest.raises(ValueError, match="unknown strategy type"):
            lc.build_strategies(cfg, spec)


class TestIncome:
    def test_shocks_off_gives_the_deterministic_profile(self) -> None:
        spec = lc.LifecycleSpec(income_shocks_enabled=False)
        income = lc.simulate_income(spec, 5, np.random.default_rng(0))
        np.testing.assert_allclose(income,
                                   np.tile(spec.deterministic_income(), (5, 1)))

    def test_shocks_preserve_the_profile_level_on_average(self) -> None:
        spec = lc.LifecycleSpec()
        income = lc.simulate_income(spec, 200000, np.random.default_rng(1))
        ratio = income.mean(axis=0) / spec.deterministic_income()
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=0.03)

    def test_permanent_shocks_widen_the_cross_section_with_age(self) -> None:
        spec = lc.LifecycleSpec()
        income = lc.simulate_income(spec, 20000, np.random.default_rng(2))
        dispersion = np.log(income).std(axis=0)
        assert dispersion[-1] > dispersion[0]


class TestSimulation:
    def test_accumulation_matches_a_hand_computation(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=28, age_death=31,
                                savings_rate=0.10,
                                income_shocks_enabled=False,
                                social_security_enabled=False)
        strategy = lc.Strategy("bill", "Bills",
                               np.tile([0.0, 0.0, 0.0, 1.0], (spec.horizon, 1)))
        paths = constant_paths(1, spec.horizon, bill=0.10)
        income = np.tile(spec.deterministic_income(), (1, 1))
        out = lc.simulate(paths, strategy, spec, income)

        wealth = 0.0
        for h in range(spec.n_working):
            wealth = (wealth + 0.10 * income[0, h]) * 1.10
        assert out.wealth_at_retirement[0] == pytest.approx(wealth)

    def test_consumption_during_work_is_income_net_of_saving(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=30, age_death=35,
                                savings_rate=0.2, income_shocks_enabled=False)
        strategy = lc.Strategy("eq", "Eq",
                               np.tile([1.0, 0.0, 0.0, 0.0], (spec.horizon, 1)))
        income = np.tile(spec.deterministic_income(), (3, 1))
        out = lc.simulate(constant_paths(3, spec.horizon), strategy, spec,
                          income)
        np.testing.assert_allclose(out.consumption[:, :spec.n_working],
                                   income * 0.8)

    def test_portfolio_return_is_the_weighted_average(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=28, age_death=31)
        strategy = lc.Strategy(
            "mix", "Mix", np.tile([0.5, 0.25, 0.15, 0.10], (spec.horizon, 1)))
        paths = constant_paths(2, spec.horizon, 0.10, 0.20, 0.04, 0.02)
        rp = lc.portfolio_returns(paths, strategy)
        expected = 0.5 * 0.10 + 0.25 * 0.20 + 0.15 * 0.04 + 0.10 * 0.02
        np.testing.assert_allclose(rp, expected)

    def test_ruin_is_detected_and_dated(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=30, age_death=60,
                                savings_rate=0.01, rule_rate=0.50,
                                income_shocks_enabled=False,
                                social_security_enabled=True)
        strategy = lc.Strategy("bill", "Bills",
                               np.tile([0.0, 0.0, 0.0, 1.0], (spec.horizon, 1)))
        paths = constant_paths(4, spec.horizon, bill=0.0)
        income = np.tile(spec.deterministic_income(), (4, 1))
        out = lc.simulate(paths, strategy, spec, income)
        assert out.ruin.all()
        assert (out.ruin_age < spec.age_death).all()
        # Consumption never drops to zero: social security is the floor.
        assert (out.consumption[:, spec.retirement_slice] > 0).all()

    def test_no_ruin_when_returns_are_generous(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=63, age_death=93,
                                income_shocks_enabled=False)
        strategy = lc.Strategy("eq", "Eq",
                               np.tile([1.0, 0.0, 0.0, 0.0], (spec.horizon, 1)))
        paths = constant_paths(2, spec.horizon, dom_eq=0.07)
        income = np.tile(spec.deterministic_income(), (2, 1))
        out = lc.simulate(paths, strategy, spec, income)
        assert not out.ruin.any()
        assert (out.bequest > 0).all()

    def test_fixed_percentage_rule_never_ruins(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=30, age_death=60,
                                retirement_rule="fixed_percentage",
                                rule_rate=0.5, income_shocks_enabled=False)
        strategy = lc.Strategy("bill", "Bills",
                               np.tile([0.0, 0.0, 0.0, 1.0], (spec.horizon, 1)))
        income = np.tile(spec.deterministic_income(), (3, 1))
        out = lc.simulate(constant_paths(3, spec.horizon, bill=0.0), strategy,
                          spec, income)
        assert not out.ruin.any()

    def test_wealth_is_never_negative(self, toy_panel, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        strategies = lc.build_strategies(toy_config, spec)
        sampler = bs.from_config(toy_panel, toy_config)
        results = lc.run_chunked(sampler, strategies, spec, 300, 150)
        for outcome in results.values():
            assert (outcome.wealth >= -1e-12).all()

    def test_rejects_mismatched_income_shape(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=30, age_death=35)
        strategy = lc.Strategy("eq", "Eq",
                               np.tile([1.0, 0.0, 0.0, 0.0], (spec.horizon, 1)))
        with pytest.raises(ValueError, match="income must be"):
            lc.simulate(constant_paths(3, spec.horizon), strategy, spec,
                        np.ones((2, spec.n_working)))

    def test_rejects_a_bootstrap_horizon_that_is_too_short(self) -> None:
        spec = lc.LifecycleSpec(age_start=25, age_retire=30, age_death=40)
        strategy = lc.Strategy("eq", "Eq",
                               np.tile([1.0, 0.0, 0.0, 0.0], (spec.horizon, 1)))
        with pytest.raises(ValueError, match="shorter than"):
            lc.portfolio_returns(constant_paths(2, 5), strategy)


class TestDriver:
    def test_every_strategy_sees_identical_income(self, toy_panel, toy_config
                                                  ) -> None:
        cfg = dict(toy_config)
        cfg["lifecycle"] = dict(toy_config["lifecycle"])
        cfg["lifecycle"]["income"] = dict(toy_config["lifecycle"]["income"],
                                          shocks_enabled=True)
        spec = lc.spec_from_config(cfg)
        strategies = lc.build_strategies(cfg, spec)
        sampler = bs.from_config(toy_panel, cfg)
        results = lc.run_chunked(sampler, strategies, spec, 200, 100)
        outcomes = list(results.values())
        np.testing.assert_allclose(
            outcomes[0].consumption[:, :spec.n_working],
            outcomes[1].consumption[:, :spec.n_working])
        np.testing.assert_allclose(outcomes[0].career_average_income,
                                   outcomes[1].career_average_income)

    def test_run_chunked_returns_the_requested_number_of_paths(
            self, toy_panel, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        strategies = lc.build_strategies(toy_config, spec)
        sampler = bs.from_config(toy_panel, toy_config)
        results = lc.run_chunked(sampler, strategies, spec, 250, 100)
        for outcome in results.values():
            assert outcome.n_paths == 250

    def test_glide_path_table_covers_every_age(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        strategies = lc.build_strategies(toy_config, spec)
        table = lc.glide_path_table(strategies, spec)
        assert list(table.index) == list(spec.ages)
        assert set(table.columns) == set(strategies)


class TestMeansTestedSimulation:
    """The Age Pension has to be assessed year by year, not once at retirement."""

    def test_benefit_rises_as_the_portfolio_falls(self) -> None:
        """The mechanism the whole section turns on.

        An earnings-related benefit is a constant through retirement. A
        means-tested one is a function of assets, so as the portfolio draws
        down the pension has to come back. A model that settled it once at
        retirement would miss exactly this.
        """
        spec = lc.LifecycleSpec(social_security_formula="means_tested")
        ea = float(spec.deterministic_income().mean())
        falling = np.array([[10.0 * ea, 5.0 * ea, 2.0 * ea]])
        paid = spec.means_tested_benefit(falling)
        assert paid[0, 0] < paid[0, 1] < paid[0, 2]
        assert paid[0, 2] == pytest.approx(spec.pension_full_rate * ea)


class TestUntouchedDefaults:
    """Every new dial must be off by default.

    Three were added at once -- a foreign income correlation, a trading cost
    and a means-tested pension. Each is a change to the simulator's arithmetic,
    and each has to be inert unless it is asked for, or the results elsewhere
    in the project silently move.
    """

    def test_new_fields_default_to_no_op(self) -> None:
        spec = lc.LifecycleSpec()
        assert spec.income_intl_correlation is None
        assert spec.trading_cost == 0.0
        assert spec.social_security_formula == "progressive"

    def test_income_is_unchanged_when_no_correlation_is_asked_for(self) -> None:
        spec = lc.LifecycleSpec()
        rng = np.random.default_rng(3)
        shocks = lc.draw_income_shocks(50, spec.n_working,
                                       np.random.default_rng(4))
        market = rng.standard_normal((50, spec.n_working))
        with_market = lc.simulate_income(spec, 50, shocks=shocks,
                                         dom_eq=market, intl_eq=market)
        without = lc.simulate_income(spec, 50, shocks=shocks)
        assert np.array_equal(with_market, without)

    def test_intl_correlation_requires_the_foreign_series(self) -> None:
        spec = lc.LifecycleSpec(income_return_correlation=0.3,
                                income_intl_correlation=0.3)
        shocks = lc.draw_income_shocks(20, spec.n_working,
                                       np.random.default_rng(5))
        market = np.random.default_rng(6).standard_normal((20, spec.n_working))
        with pytest.raises(ValueError, match="international"):
            lc.simulate_income(spec, 20, shocks=shocks, dom_eq=market)

    def test_out_of_range_correlations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="income_intl_correlation"):
            lc.LifecycleSpec(income_intl_correlation=1.5)


class TestSuperannuationGuarantee:
    """A compulsory employer contribution alongside voluntary saving."""

    def test_off_by_default(self) -> None:
        spec = lc.LifecycleSpec()
        assert spec.super_guarantee_rate == 0.0
        assert spec.super_net_rate == 0.0
        assert spec.total_contribution_rate == spec.savings_rate

    def test_contributions_tax_is_taken_on_the_way_in(self) -> None:
        spec = lc.LifecycleSpec(savings_rate=0.10, super_guarantee_rate=0.12,
                                super_contributions_tax=0.15)
        assert spec.super_net_rate == pytest.approx(0.102)
        assert spec.total_contribution_rate == pytest.approx(0.202)
        assert spec.super_share_of_contributions == pytest.approx(0.102 / 0.202)

    def test_it_adds_to_voluntary_saving_rather_than_replacing_it(self) -> None:
        """The distinction the whole comparison turns on.

        An employer contribution is additional. Modelling it as a bigger
        savings rate would leave the Australian saver contributing 12% where
        they actually contribute 22.2% gross.
        """
        spec = lc.LifecycleSpec(savings_rate=0.10, super_guarantee_rate=0.12)
        assert spec.total_contribution_rate > spec.savings_rate
        assert spec.total_contribution_rate > spec.super_guarantee_rate

    def test_it_does_not_reduce_working_life_consumption(self) -> None:
        """Statutory incidence is on the employer, so take-home pay is
        unchanged and only the portfolio grows."""
        from tests.test_turnover import _Paths, _flat_paths  # type: ignore

        base = lc.LifecycleSpec()
        withsg = dataclasses.replace(base, super_guarantee_rate=0.12)
        paths = _flat_paths(base.horizon, dom_eq=0.05, intl_eq=0.05,
                            bond=0.02, bill=0.01)
        strat = lc.Strategy(key="s", label="s", weights=np.tile(
            [0.5, 0.5, 0.0, 0.0], (base.horizon, 1)))
        income = lc.simulate_income(base, paths.n_paths,
                                    np.random.default_rng(1))
        a = lc.simulate(paths, strat, base, income)
        b = lc.simulate(paths, strat, withsg, income)
        work = slice(0, base.n_working)
        assert np.allclose(a.consumption[:, work], b.consumption[:, work])
        assert (b.wealth_at_retirement > a.wealth_at_retirement).all()

    def test_zero_guarantee_is_the_original_simulation(self) -> None:
        from tests.test_turnover import _flat_paths  # type: ignore

        spec = lc.LifecycleSpec()
        paths = _flat_paths(spec.horizon, dom_eq=0.06, bond=0.02)
        strat = lc.Strategy(key="s", label="s", weights=np.tile(
            [1.0, 0.0, 0.0, 0.0], (spec.horizon, 1)))
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(2))
        a = lc.simulate(paths, strat, spec, income)
        b = lc.simulate(paths, strat,
                        dataclasses.replace(spec, super_guarantee_rate=0.0),
                        income)
        assert np.array_equal(a.wealth, b.wealth)
        assert np.array_equal(a.consumption, b.consumption)

    def test_out_of_range_rates_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="super_guarantee_rate"):
            lc.LifecycleSpec(super_guarantee_rate=1.2)
        with pytest.raises(ValueError, match="super_contributions_tax"):
            lc.LifecycleSpec(super_contributions_tax=1.0)
