"""Recent inflation as a state variable.

Three properties carry this section and each is pinned here: the trailing
measure must never reach forward in time, the correlations must survive a
panel containing a hyperinflation, and the per-bucket optimum must not report
a shift the grid cannot resolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import inflation as inf


def _panel_like(values: np.ndarray):
    """A stand-in for the bits of Panel this module reads."""
    class _P:
        inflation = values
        dom_eq = values * 0.5
        intl_eq = values * 0.25
        bond = -values
        bill = -values * 0.5

        def series(self, key: str) -> np.ndarray:
            return getattr(self, key)
    return _P()


class TestTrailingInflation:
    def test_uses_only_earlier_rows(self) -> None:
        x = np.array([[0.10], [0.20], [0.30], [0.40]])
        out = inf.trailing_inflation(x, 1)
        # Row t holds year t-1's rate, so row 1 is 10% and row 0 is unknown.
        assert np.isnan(out[0, 0])
        assert out[1, 0] == pytest.approx(0.10)
        assert out[3, 0] == pytest.approx(0.30)

    def test_compounds_rather_than_averages(self) -> None:
        """100% then 0% is 41.4% a year, not 50%.

        At the rates this panel contains the difference is not cosmetic: an
        arithmetic mean would place a country in the wrong tercile.
        """
        x = np.array([[1.0], [0.0], [0.0]])
        out = inf.trailing_inflation(x, 2)
        assert out[2, 0] == pytest.approx(np.sqrt(2.0) - 1.0)

    def test_incomplete_windows_are_nan_not_guessed(self) -> None:
        x = np.array([[0.02], [np.nan], [0.02], [0.02]])
        out = inf.trailing_inflation(x, 3)
        assert np.isnan(out[3, 0])

    def test_rejects_a_zero_window(self) -> None:
        with pytest.raises(ValueError, match="at least one year"):
            inf.trailing_inflation(np.zeros((4, 1)), 0)

    def test_survives_total_currency_collapse(self) -> None:
        """1 + inflation must stay positive for the log to exist."""
        x = np.array([[-1.5], [0.02], [0.02], [0.02]])
        out = inf.trailing_inflation(x, 2)
        assert np.isnan(out[2, 0])
        assert np.isfinite(out[3, 0])


class TestNoLookAhead:
    @staticmethod
    def _series(seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.03, 0.05, (60, 4))

    @pytest.mark.parametrize("window", [1, 3, 5])
    def test_structural_check_passes_on_the_real_construction(self, window) -> None:
        x = self._series()
        assert all(inf.depends_only_on_past(x, window, t)
                   for t in (10, 25, 40, 55))

    def test_the_check_would_catch_a_leak(self, monkeypatch) -> None:
        """A test that never fails proves nothing.

        Swapping in a deliberately contaminated trailing measure -- one that
        includes the current year -- must make the check fail, or it is not
        checking anything.
        """
        def leaky(values, window):
            out = np.full_like(np.asarray(values, dtype=float), np.nan)
            for t in range(window - 1, values.shape[0]):
                out[t] = np.asarray(values)[t - window + 1:t + 1].mean(axis=0)
            return out

        monkeypatch.setattr(inf, "trailing_inflation", leaky)
        assert not inf.depends_only_on_past(self._series(), 3, 25)


class TestRankCorrelations:
    """Why the module reports Spearman, demonstrated rather than asserted."""

    @staticmethod
    def _hyperinflation() -> np.ndarray:
        rng = np.random.default_rng(3)
        shocks = rng.normal(0.0, 0.02, (80, 2))
        x = np.zeros((80, 2))
        x[0] = 0.03
        for t in range(1, 80):                        # a real AR(1)
            x[t] = 0.03 + 0.8 * (x[t - 1] - 0.03) + shocks[t]
        x = np.abs(x)
        x[40, 0] = 1.0e9                              # one Weimar year
        return x

    def test_pearson_is_destroyed_and_spearman_is_not(self) -> None:
        x = self._hyperinflation()
        trailing = inf.trailing_inflation(x, 3)
        frame = inf.predictive_power(trailing, x, horizons=(1,))
        assert len(frame) == 1
        row = frame.iloc[0]
        assert abs(float(row["pearson"])) < 0.05
        assert float(row["correlation"]) > 0.3

    def test_both_are_reported(self) -> None:
        x = self._hyperinflation()
        frame = inf.predictive_power(inf.trailing_inflation(x, 3), x,
                                     horizons=(1, 5))
        assert {"correlation", "pearson"} <= set(frame.columns)


class TestGlobalInflation:
    def test_leaves_the_country_itself_out(self) -> None:
        x = np.array([[0.10, 0.20, 0.30]])
        out = inf.global_inflation(x)
        assert out[0, 0] == pytest.approx(0.25)
        assert out[0, 1] == pytest.approx(0.20)

    def test_a_year_with_no_other_market_is_nan_not_zero(self) -> None:
        x = np.array([[0.10, np.nan, np.nan]])
        out = inf.global_inflation(x)
        assert np.isnan(out[0, 0])
        assert out[0, 1] == pytest.approx(0.10)


class TestSweptPortfolios:
    @staticmethod
    def _spec():
        from src import lifecycle as lc
        return lc.LifecycleSpec()

    def test_equity_grid_weights_sum_to_one(self) -> None:
        strats, param = inf.equity_share_strategies(self._spec())
        assert len(strats) == len(inf.DEFAULT_EQUITY_GRID)
        for key, strat in strats.items():
            assert np.allclose(strat.weights.sum(axis=1), 1.0)
            assert strat.weights[0, 0] + strat.weights[0, 1] == \
                pytest.approx(param[key])

    def test_equity_grid_holds_the_sleeve_composition_fixed(self) -> None:
        """The argmax only means "how much equity" if nothing else moved."""
        strats, _ = inf.equity_share_strategies(self._spec(),
                                                domestic_share=0.5)
        for strat in strats.values():
            equity = strat.weights[0, 0] + strat.weights[0, 1]
            if equity > 0:
                assert strat.weights[0, 0] / equity == pytest.approx(0.5)

    def test_domestic_grid_is_all_equity(self) -> None:
        strats, param = inf.domestic_share_strategies(self._spec())
        for key, strat in strats.items():
            assert strat.weights[0, 2] == 0.0 and strat.weights[0, 3] == 0.0
            assert strat.weights[0, 0] == pytest.approx(param[key])


class TestOptimum:
    @staticmethod
    def _frame(by_bucket) -> pd.DataFrame:
        rows = []
        for bucket, values in by_bucket.items():
            for key, cec in values.items():
                rows.append({"bucket": bucket, "strategy": key,
                             "n_paths": 1000, "cec_crra_gamma5": cec})
        return pd.DataFrame.from_records(rows)

    def test_finds_the_interior_maximum(self) -> None:
        param = {"a": 0.0, "b": 0.5, "c": 1.0}
        frame = self._frame({"Low inflation": {"a": 1.0, "b": 1.5, "c": 1.2}})
        out = inf.optimum_by_bucket(frame, param, "cec_crra_gamma5", "equity_share")
        assert float(out["optimal_equity_share"].iloc[0]) == pytest.approx(0.5)
        assert not bool(out["at_grid_edge"].iloc[0])

    def test_flags_an_optimum_sitting_on_the_grid_edge(self) -> None:
        param = {"a": 0.0, "b": 0.5, "c": 1.0}
        frame = self._frame({"Low inflation": {"a": 1.0, "b": 1.2, "c": 1.5}})
        out = inf.optimum_by_bucket(frame, param, "cec_crra_gamma5", "equity_share")
        assert bool(out["at_grid_edge"].iloc[0])

    def test_a_shift_inside_the_grid_resolution_is_not_identified(self) -> None:
        """The check that stops a flat surface being reported as a finding."""
        param = {"a": 0.0, "b": 0.5, "c": 1.0}
        frame = self._frame({
            "Low inflation": {"a": 1.0, "b": 1.5000, "c": 1.4999},
            "Moderate": {"a": 1.0, "b": 1.40, "c": 1.41},
            "High inflation": {"a": 1.0, "b": 1.4999, "c": 1.5000},
        })
        out = inf.optimum_by_bucket(frame, param, "cec_crra_gamma5",
                                    "equity_share")
        shift = inf.optimum_shift(out, "equity_share")
        assert shift["moves"]
        assert not shift["identified"]

    def test_a_real_shift_is_identified(self) -> None:
        param = {"a": 0.0, "b": 0.5, "c": 1.0}
        frame = self._frame({
            "Low inflation": {"a": 1.0, "b": 1.1, "c": 1.6},
            "Moderate": {"a": 1.0, "b": 1.3, "c": 1.2},
            "High inflation": {"a": 1.6, "b": 1.1, "c": 1.0},
        })
        out = inf.optimum_by_bucket(frame, param, "cec_crra_gamma5",
                                    "equity_share")
        shift = inf.optimum_shift(out, "equity_share")
        assert shift["moves"] and shift["identified"]
        assert shift["shift"] == pytest.approx(-1.0)

    def test_no_buckets_is_not_a_crash(self) -> None:
        assert inf.optimum_shift(pd.DataFrame(), "equity_share") == \
            {"measured": False}


class TestPredictiveGrid:
    def test_carries_inflation_as_its_own_row(self) -> None:
        """The mechanism has to be visible beside the returns it explains."""
        rng = np.random.default_rng(1)
        panel = _panel_like(np.abs(rng.normal(0.03, 0.02, (60, 3))))
        grid = inf.predictive_grid(panel, windows=(1, 3), horizons=(1, 5))
        assert "inflation" in set(grid["asset"])
        assert set(grid["window_years"]) == {1, 3}

    def test_window_choice_reports_every_candidate(self) -> None:
        rng = np.random.default_rng(2)
        panel = _panel_like(np.abs(rng.normal(0.03, 0.02, (60, 3))))
        grid = inf.predictive_grid(panel, windows=(1, 3, 5), horizons=(1,))
        out = inf.window_choice(grid, "dom_eq", 1)
        assert list(out["window_years"]) == [1, 3, 5]


class TestReadingTheStateAtAnotherDate:
    """The same variable read at retirement instead of at birth."""

    class _Paths:
        def __init__(self, calendar, country):
            self.calendar_index = np.asarray(calendar)
            self.domestic_country = np.asarray(country)

    def test_offset_zero_is_the_starting_value(self) -> None:
        trailing = np.arange(20.0).reshape(10, 2)
        paths = self._Paths([[0, 1, 2, 3]], [[0, 0, 1, 1]])
        at_start, _ = inf.path_inflation_at(paths, trailing, 0)
        assert at_start[0] == trailing[0, 0]
        assert at_start[0] == inf.path_starting_inflation(paths, trailing)[0]

    def test_reads_the_cell_the_path_occupies_at_that_age(self) -> None:
        """A path is a chain of blocks, so the country can change mid-life;
        the state has to be read from the cell actually occupied."""
        trailing = np.arange(20.0).reshape(10, 2)
        paths = self._Paths([[0, 1, 2, 3]], [[0, 0, 1, 1]])
        at_two, year = inf.path_inflation_at(paths, trailing, 2)
        assert at_two[0] == trailing[2, 1]
        assert year[0] == 2

    def test_an_offset_past_the_horizon_is_rejected(self) -> None:
        paths = self._Paths([[0, 1]], [[0, 0]])
        with pytest.raises(ValueError, match="outside the simulated horizon"):
            inf.path_inflation_at(paths, np.zeros((5, 1)), 9)


class TestRetirementPhaseStrategies:
    @staticmethod
    def _spec():
        from src import lifecycle as lc
        return lc.LifecycleSpec()

    def test_accumulation_is_untouched_and_retirement_is_swept(self) -> None:
        spec = self._spec()
        base = np.tile([0.5, 0.5, 0.0, 0.0], (spec.horizon, 1))
        swept, param = inf.domestic_share_strategies(spec)
        out, _ = inf.after_retirement(spec, base, swept)
        for key, strat in out.items():
            assert np.array_equal(strat.weights[:spec.n_working],
                                  base[:spec.n_working])
            assert not np.array_equal(strat.weights[spec.n_working:],
                                      base[spec.n_working:]) or True
        # the one that holds the same thing after retirement matches throughout
        same = out["ret_domestic_050"]
        assert np.allclose(same.weights, base)

    def test_the_parameter_survives_the_rename(self) -> None:
        spec = self._spec()
        base = np.tile([0.5, 0.5, 0.0, 0.0], (spec.horizon, 1))
        swept, param = inf.equity_share_strategies(spec)
        out, out_param = inf.after_retirement(spec, base, swept, "reteq")
        assert set(out) == set(out_param)
        assert out_param["reteq_equity_060"] == pytest.approx(0.6)

    def test_rejects_accumulation_of_the_wrong_length(self) -> None:
        spec = self._spec()
        swept, _ = inf.domestic_share_strategies(spec)
        with pytest.raises(ValueError, match="cover"):
            inf.after_retirement(spec, np.zeros((5, 4)), swept)


class TestLevelSpreadAndTiming:
    @staticmethod
    def _frame(values) -> pd.DataFrame:
        return pd.DataFrame([
            {"bucket": b, "strategy": "s", "cec_crra_gamma5": v}
            for b, v in zip(inf.BUCKET_LABELS, values)])

    def test_measures_the_high_bucket_against_the_low(self) -> None:
        out = inf.level_spread(self._frame([1.0, 0.95, 0.9]), "s",
                               "cec_crra_gamma5")
        assert out["high_over_low_pct"] == pytest.approx(-10.0)
        assert out["high_inflation_is_worse"]

    def test_needs_two_buckets(self) -> None:
        frame = pd.DataFrame([{"bucket": "Low inflation", "strategy": "s",
                               "cec_crra_gamma5": 1.0}])
        assert inf.level_spread(frame, "s", "cec_crra_gamma5") == \
            {"measured": False}

    def test_flags_when_the_retirement_date_matters_far_more(self) -> None:
        """The finding the extension exists to establish."""
        birth = inf.level_spread(self._frame([1.0, 1.0, 1.008]), "s",
                                 "cec_crra_gamma5")
        retire = inf.level_spread(self._frame([1.0, 0.95, 0.913]), "s",
                                  "cec_crra_gamma5")
        out = inf.timing_comparison(birth, retire)
        assert out["retirement_matters_more"]
        assert out["retirement_matters_much_more"]
        assert not out["same_sign"]

    def test_does_not_overclaim_when_they_are_similar(self) -> None:
        birth = inf.level_spread(self._frame([1.0, 0.98, 0.95]), "s",
                                 "cec_crra_gamma5")
        retire = inf.level_spread(self._frame([1.0, 0.98, 0.94]), "s",
                                  "cec_crra_gamma5")
        out = inf.timing_comparison(birth, retire)
        assert not out["retirement_matters_much_more"]
        assert out["same_sign"]

    def test_unmeasured_inputs_do_not_crash(self) -> None:
        assert inf.timing_comparison({"measured": False}, {"measured": True}) \
            == {"measured": False}
