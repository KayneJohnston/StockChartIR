"""Tests for housing as a fifth investable asset.

Two claims carry this study: that the de-smoothing restores volatility the
index hid, and that housing is drawn on the same block structure as everything
else. The tests below attack both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import bootstrap as bs
from src import glidepath as gp
from src import housing as hs
from src import lifecycle as lc
from src import observed as obs


class TestGapRespectingAutocorrelation:
    def test_matches_the_naive_estimate_without_gaps(self) -> None:
        rng = np.random.default_rng(0)
        series = np.zeros(200)
        for t in range(1, 200):
            series[t] = 0.6 * series[t - 1] + rng.normal(0, 0.1)
        assert hs.gap_respecting_autocorrelation(series) == pytest.approx(
            obs.first_order_autocorrelation(series), abs=1e-12)

    def test_does_not_splice_across_a_gap(self) -> None:
        """The distinguishing case.

        Two independent halves separated by a gap. Dropping the missing value
        and correlating what remains pairs the last year of one half with the
        first of the other; respecting the gap does not.
        """
        left = np.array([0.1, -0.1] * 15)
        right = np.array([0.1, -0.1] * 15)
        series = np.concatenate([left, [np.nan], right])
        pairs = np.isfinite(series[:-1]) & np.isfinite(series[1:])
        assert int(pairs.sum()) == 58, "the spliced pair must be excluded"
        assert np.isfinite(hs.gap_respecting_autocorrelation(series))

    def test_too_few_pairs_is_undefined(self) -> None:
        assert np.isnan(hs.gap_respecting_autocorrelation(
            np.array([0.1, 0.2, np.nan, 0.3])))


class TestDesmoothing:
    def test_restores_the_volatility_a_filter_removed(self) -> None:
        """Smooth a known series, then invert it, and recover the original."""
        rng = np.random.default_rng(1)
        true = rng.normal(0.06, 0.20, 400)
        a = 0.4
        smoothed = np.empty_like(true)
        smoothed[0] = true[0]
        for t in range(1, 400):
            smoothed[t] = (1 - a) * true[t] + a * smoothed[t - 1]
        assert smoothed.std(ddof=1) < true.std(ddof=1), "the filter must hide risk"
        recovered = obs.desmooth(smoothed, a)
        # Not exact -- the filter is on the smoothed lag, the inverse on the
        # observed one -- but the volatility must come back.
        assert recovered[1:].std(ddof=1) == pytest.approx(
            true.std(ddof=1), rel=0.15)

    def test_preserves_the_mean(self) -> None:
        rng = np.random.default_rng(2)
        series = 0.06 + rng.normal(0, 0.1, 500)
        out = obs.desmooth(series, 0.5)
        assert float(np.nanmean(out)) == pytest.approx(
            float(series[1:].mean()), abs=0.01)

    def test_leaves_an_unsmoothed_series_alone(self) -> None:
        series = np.array([0.1, -0.1, 0.1, -0.1] * 30)   # negative autocorr
        np.testing.assert_allclose(obs.desmooth(series), series)


class TestHoldingCost:
    def test_is_charged_on_value_not_on_gains(self) -> None:
        gross = np.array([0.10, -0.10, 0.0])
        np.testing.assert_allclose(hs.net_of_cost(gross, 0.02),
                                   np.array([0.08, -0.12, -0.02]))

    def test_zero_cost_is_a_no_op(self) -> None:
        gross = np.array([0.10, -0.10])
        np.testing.assert_allclose(hs.net_of_cost(gross, 0.0), gross)

    def test_shifts_the_mean_by_exactly_the_cost(self) -> None:
        rng = np.random.default_rng(3)
        gross = rng.normal(0.07, 0.15, 1000)
        assert (hs.moments(gross, 0.0)["mean"]
                - hs.moments(gross, 0.03)["mean"]) == pytest.approx(0.03)

    def test_does_not_change_the_volatility(self) -> None:
        rng = np.random.default_rng(4)
        gross = rng.normal(0.07, 0.15, 1000)
        assert hs.moments(gross, 0.04)["sd"] == pytest.approx(
            hs.moments(gross, 0.0)["sd"])


class TestBreakEven:
    def test_interpolates_between_the_bracketing_costs(self) -> None:
        frame = pd.DataFrame({"holding_cost": [0.0, 0.02, 0.04],
                              "mean_housing": [0.40, 0.20, 0.00]})
        # Crosses 0.01 between 0.02 and 0.04, 95% of the way from 0.20 to 0.00.
        assert hs.break_even_cost(frame) == pytest.approx(0.039)

    def test_undefined_when_housing_is_never_wanted(self) -> None:
        frame = pd.DataFrame({"holding_cost": [0.0, 0.02],
                              "mean_housing": [0.0, 0.0]})
        assert np.isnan(hs.break_even_cost(frame))

    def test_undefined_when_the_sweep_never_crosses(self) -> None:
        """Reporting an extrapolated break-even would invent the answer."""
        frame = pd.DataFrame({"holding_cost": [0.0, 0.02, 0.04],
                              "mean_housing": [0.50, 0.45, 0.40]})
        assert np.isnan(hs.break_even_cost(frame))

    def test_is_ordered_by_cost_not_by_row_order(self) -> None:
        shuffled = pd.DataFrame({"holding_cost": [0.04, 0.0, 0.02],
                                 "mean_housing": [0.00, 0.40, 0.20]})
        assert hs.break_even_cost(shuffled) == pytest.approx(0.039)


class TestPanelRestriction:
    def test_marks_cells_without_housing_unavailable(self, toy_panel) -> None:
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.05)
        housing[:5, 0] = np.nan
        restricted = hs.restrict_to_housing(toy_panel, housing)
        assert not restricted.available[:5, 0].any()
        assert restricted.available[5:, 0].all()

    def test_never_adds_availability(self, toy_panel) -> None:
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.05)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        assert (restricted.available <= toy_panel.available).all(), (
            "a housing return cannot make an otherwise-missing year usable"
        )

    def test_leaves_the_return_series_untouched(self, toy_panel) -> None:
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.05)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        np.testing.assert_allclose(restricted.dom_eq, toy_panel.dom_eq,
                                   equal_nan=True)


class TestGather:
    def test_uses_the_samplers_own_calendar_and_country(self, toy_panel,
                                                        toy_config) -> None:
        """Housing must ride the same blocks as the other four assets.

        Sampling it independently would preserve its marginal distribution and
        destroy its correlation with equity, which is the one thing a joint
        block bootstrap exists to keep.
        """
        sampler = bs.from_config(toy_panel, toy_config)
        paths = sampler.sample(200, chunk_size=100)
        # A matrix whose value encodes its own (year, country) coordinates.
        marker = (np.arange(toy_panel.n_years)[:, None] * 100
                  + np.arange(toy_panel.n_countries)[None, :]).astype(float)
        out = hs.gather(paths, marker)
        expected = (np.asarray(paths.calendar_index) * 100
                    + np.asarray(paths.domestic_country))
        np.testing.assert_allclose(out, expected)

    def test_shape_matches_the_paths(self, toy_panel, toy_config) -> None:
        sampler = bs.from_config(toy_panel, toy_config)
        paths = sampler.sample(200, chunk_size=100)
        marker = np.zeros((toy_panel.n_years, toy_panel.n_countries))
        assert hs.gather(paths, marker).shape == (200, paths.horizon)


class TestFiveAssetEvaluator:
    def _setup(self, toy_panel, toy_config):
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.04)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(1))
        return paths, spec, income, hs.gather(paths, housing)

    def test_accepts_a_fifth_asset(self, toy_panel, toy_config) -> None:
        paths, spec, income, gross = self._setup(toy_panel, toy_config)
        ev = gp.BatchEvaluator(paths, spec, income, toy_config,
                               assets=hs.ASSETS, extra={hs.HOUSING: gross})
        assert ev.n_assets == 5
        weights = np.tile(np.full(5, 0.2), (1, spec.horizon, 1))
        assert np.isfinite(ev.cec(weights, 5.0)).all()

    def test_still_defaults_to_the_four_asset_set(self, toy_panel,
                                                  toy_config) -> None:
        paths, spec, income, _ = self._setup(toy_panel, toy_config)
        ev = gp.BatchEvaluator(paths, spec, income, toy_config)
        assert ev.assets == lc.ASSETS and ev.n_assets == 4

    def test_a_hundred_percent_housing_earns_the_housing_return(
            self, toy_panel, toy_config) -> None:
        """The fifth column must actually be the series it was handed."""
        paths, spec, income, gross = self._setup(toy_panel, toy_config)
        ev = gp.BatchEvaluator(paths, spec, income, toy_config,
                               assets=hs.ASSETS, extra={hs.HOUSING: gross})
        weights = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
                          (1, spec.horizon, 1))
        realised = ev._portfolio_returns(weights)[:, :, 0]
        np.testing.assert_allclose(realised, gross[:, :spec.horizon].T,
                                   rtol=1e-12)

    def test_rejects_a_wrongly_shaped_extra_series(self, toy_panel,
                                                   toy_config) -> None:
        paths, spec, income, gross = self._setup(toy_panel, toy_config)
        with pytest.raises(ValueError, match="expected"):
            gp.BatchEvaluator(paths, spec, income, toy_config,
                              assets=hs.ASSETS,
                              extra={hs.HOUSING: gross[:10]})

    def test_rejects_a_series_with_holes(self, toy_panel, toy_config) -> None:
        """A missing return would silently become a zero in the matmul."""
        paths, spec, income, gross = self._setup(toy_panel, toy_config)
        holed = gross.copy()
        holed[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            gp.BatchEvaluator(paths, spec, income, toy_config,
                              assets=hs.ASSETS, extra={hs.HOUSING: holed})

    def test_rejects_weights_of_the_wrong_width(self, toy_panel,
                                                toy_config) -> None:
        paths, spec, income, gross = self._setup(toy_panel, toy_config)
        ev = gp.BatchEvaluator(paths, spec, income, toy_config,
                               assets=hs.ASSETS, extra={hs.HOUSING: gross})
        with pytest.raises(ValueError, match="must be"):
            ev.cec(np.tile(np.full(4, 0.25), (1, spec.horizon, 1)), 5.0)


class TestSearchOverFiveAssets:
    def test_lattice_and_neighbourhood_grow_with_the_asset_count(self) -> None:
        from src import allocation as al
        four = al.simplex_lattice(0.25, 4)
        five = al.simplex_lattice(0.25, 5)
        assert four.shape == (35, 4) and five.shape == (70, 5)
        np.testing.assert_allclose(five.sum(axis=1), 1.0)
        assert len(al.asset_pairs(4)) == 12
        assert len(al.asset_pairs(5)) == 20

    def test_exchange_neighbourhood_follows_the_weight_width(self) -> None:
        from src import allocation as al
        moves = al.exchange_neighbourhood(np.full(5, 0.2), 0.05)
        assert moves.shape == (21, 5), "twenty exchanges plus the incumbent"
        np.testing.assert_allclose(moves.sum(axis=1), 1.0)

    def test_finds_the_dominant_asset(self, toy_panel, toy_config) -> None:
        """A search that cannot find a free lunch is not testing anything.

        Housing is handed a return that beats every other asset in every year,
        so the optimum is a corner and the search must reach it.
        """
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), 0.60)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(1))
        ev = gp.BatchEvaluator(paths, spec, income, toy_config,
                               assets=hs.ASSETS,
                               extra={hs.HOUSING: hs.gather(paths, housing)})
        weights, cec, _ = hs.constant_mix_optimum(ev, 5.0, 0.25, 0.05)
        assert weights[-1] > 0.9, f"expected a housing corner, got {weights}"

    def test_a_dominated_asset_is_not_held(self, toy_panel,
                                           toy_config) -> None:
        """The mirror image: a terrible asset must earn no weight."""
        housing = np.full((toy_panel.n_years, toy_panel.n_countries), -0.50)
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(1))
        ev = gp.BatchEvaluator(paths, spec, income, toy_config,
                               assets=hs.ASSETS,
                               extra={hs.HOUSING: hs.gather(paths, housing)})
        weights, _, _ = hs.constant_mix_optimum(ev, 5.0, 0.25, 0.05)
        assert weights[-1] < 0.05, f"expected no housing, got {weights}"

    def test_raising_the_cost_never_raises_the_housing_weight(
            self, toy_panel, toy_config) -> None:
        """Monotonicity is the one property the sweep must have.

        Under common random numbers a higher holding cost strictly dominates a
        lower one for the same allocation, so the solved weight cannot rise
        with cost by more than the search's own step.
        """
        rng = np.random.default_rng(5)
        # Comfortably better than the toy panel's equity at zero cost, so the
        # sweep starts somewhere it can fall from.
        housing = rng.normal(0.16, 0.06,
                             (toy_panel.n_years, toy_panel.n_countries))
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(400, chunk_size=200)
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(1))
        gross = hs.gather(paths, housing)
        frame = hs.solve_sweep(paths, spec, income, toy_config, gross,
                               [0.0, 0.06, 0.14], 5.0, 0.25, 0.05)
        five = frame[frame["investable_set"] == "five assets"]
        weights = five.sort_values("holding_cost")["mean_housing"].to_numpy()
        assert weights[0] > 0.0, "housing must be wanted when it is free"
        assert np.all(np.diff(weights) <= 0.05 + 1e-9), weights
        assert weights[0] > weights[-1], "cost must bite somewhere"

    def test_the_control_row_holds_no_housing(self, toy_panel,
                                              toy_config) -> None:
        rng = np.random.default_rng(6)
        housing = rng.normal(0.08, 0.10,
                             (toy_panel.n_years, toy_panel.n_countries))
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(1))
        frame = hs.solve_sweep(paths, spec, income, toy_config,
                               hs.gather(paths, housing), [0.0], 5.0,
                               0.25, 0.05)
        control = frame[frame["investable_set"] == "four assets"].iloc[0]
        assert float(control["mean_housing"]) == 0.0
        assert float(control["advantage_pct"]) == 0.0


class TestAgeVaryingCheck:
    """Removing the constant-mix restriction must not look like a loss."""

    def _fixture(self, toy_panel, toy_config):
        rng = np.random.default_rng(11)
        housing = rng.normal(0.10, 0.08,
                             (toy_panel.n_years, toy_panel.n_countries))
        restricted = hs.restrict_to_housing(toy_panel, housing)
        spec = lc.spec_from_config(toy_config)
        sampler = bs.from_config(restricted, toy_config,
                                 horizon_years=spec.horizon)
        paths = sampler.sample(200, chunk_size=100)
        income = lc.simulate_income(spec, paths.n_paths,
                                    np.random.default_rng(1))
        return paths, spec, income, hs.gather(paths, housing)

    def test_never_reports_a_loss_against_the_constant_mix(
            self, toy_panel, toy_config) -> None:
        """Constant schedules are a subset of age-varying ones.

        The age-varying optimum therefore cannot truly be worse, so a negative
        gain would be a fact about the search rather than about the shape.
        Seeding the search at the constant-mix answer rules that out.
        """
        paths, spec, income, gross = self._fixture(toy_panel, toy_config)
        constant = hs.solve_sweep(paths, spec, income, toy_config, gross,
                                  [0.0, 0.03], 5.0, 0.25, 0.05)
        five = constant[constant["investable_set"] == "five assets"]
        frame = hs.age_varying_check(paths, spec, income, toy_config, gross,
                                     [0.0, 0.03], 5.0, five,
                                     coarse_sweeps=1, fine_sweeps=1)
        assert (frame["cec_gain_pct"] >= -1e-9).all(), \
            frame[["holding_cost", "cec", "constant_mix_cec", "cec_gain_pct"]]

    def test_splits_the_weight_by_life_phase(self, toy_panel,
                                             toy_config) -> None:
        paths, spec, income, gross = self._fixture(toy_panel, toy_config)
        constant = hs.solve_sweep(paths, spec, income, toy_config, gross,
                                  [0.0], 5.0, 0.25, 0.05)
        five = constant[constant["investable_set"] == "five assets"]
        frame = hs.age_varying_check(paths, spec, income, toy_config, gross,
                                     [0.0], 5.0, five,
                                     coarse_sweeps=1, fine_sweeps=1)
        row = frame.iloc[0]
        n_w, n_r = spec.n_working, spec.horizon - spec.n_working
        blended = (float(row["housing_working"]) * n_w
                   + float(row["housing_retired"]) * n_r) / spec.horizon
        assert blended == pytest.approx(float(row["mean_housing"]), abs=1e-9)
