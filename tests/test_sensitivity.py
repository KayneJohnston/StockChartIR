"""Tests for the sensitivity sweep engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import lifecycle as lc
from src import sensitivity as sn


@pytest.fixture()
def ctx(toy_panel, toy_config) -> sn.SweepContext:
    return sn.SweepContext.build(toy_config, toy_panel, n_paths=400,
                                 max_horizon=12, max_working=10)


class TestStrategyBuilders:
    def test_constant_mix_weights_are_tiled(self) -> None:
        strat = sn.constant_mix("x", "X", {"dom_eq": 0.4, "bond": 0.6}, 5)
        assert strat.weights.shape == (5, 4)
        np.testing.assert_allclose(strat.weights[0], [0.4, 0.0, 0.6, 0.0])
        np.testing.assert_allclose(strat.weights.sum(axis=1), 1.0)

    def test_constant_mix_rejects_bad_weights(self) -> None:
        with pytest.raises(ValueError, match="sum to"):
            sn.constant_mix("x", "X", {"dom_eq": 0.4}, 5)

    @pytest.mark.parametrize("share", [0.0, 0.25, 0.5, 1.0])
    def test_all_equity_split_puts_everything_in_equities(self, share) -> None:
        strat = sn.all_equity_split(share, 6)
        assert strat.weights[0, 0] == pytest.approx(share)
        assert strat.weights[0, 1] == pytest.approx(1 - share)
        assert strat.equity_share()[0] == pytest.approx(1.0)

    def test_all_equity_split_clamps_out_of_range_input(self) -> None:
        assert sn.all_equity_split(1.8, 4).weights[0, 0] == pytest.approx(1.0)
        assert sn.all_equity_split(-0.3, 4).weights[0, 0] == pytest.approx(0.0)

    def test_equity_mix_splits_both_sleeves(self) -> None:
        strat = sn.equity_fixed_income_mix(0.6, 4, domestic_share=0.5,
                                           bond_share=0.7)
        np.testing.assert_allclose(strat.weights[0],
                                   [0.30, 0.30, 0.28, 0.12])


class TestSweepContext:
    def test_caches_paths_at_the_requested_horizon(self, ctx) -> None:
        assert ctx.paths.n_paths == 400
        assert ctx.paths.horizon == 12

    def test_spec_with_overrides_only_named_fields(self, ctx) -> None:
        spec = ctx.spec_with(age_death=40)
        assert spec.age_death == 40
        assert spec.age_retire == ctx.base_spec.age_retire
        assert spec.savings_rate == ctx.base_spec.savings_rate

    def test_income_uses_common_random_numbers(self, ctx) -> None:
        a = ctx.income_for(ctx.base_spec)
        b = ctx.income_for(ctx.base_spec)
        np.testing.assert_array_equal(a, b)

    def test_shorter_career_is_a_prefix_of_a_longer_one(self, ctx,
                                                        toy_config) -> None:
        cfg = dict(toy_config)
        cfg["lifecycle"] = dict(toy_config["lifecycle"])
        cfg["lifecycle"]["income"] = dict(toy_config["lifecycle"]["income"],
                                          shocks_enabled=True)
        context = sn.SweepContext.build(cfg, ctx.panel, n_paths=200,
                                        max_horizon=12, max_working=10)
        long_career = context.income_for(context.spec_with(age_retire=32))
        short_career = context.income_for(context.spec_with(age_retire=30))
        n = short_career.shape[1]
        np.testing.assert_allclose(long_career[:, :n], short_career[:, :n])

    def test_rejects_a_spec_longer_than_the_cached_draw(self, ctx) -> None:
        strategies = ctx.strategies_from_config()
        with pytest.raises(ValueError, match="exceeds the cached"):
            ctx.run(strategies, ctx.spec_with(age_death=60))

    def test_run_returns_one_outcome_per_strategy(self, ctx) -> None:
        results = ctx.run(ctx.strategies_from_config())
        assert set(results) == set(ctx.cfg["strategies"])
        for outcome in results.values():
            assert outcome.n_paths == 400


class TestSummarise:
    def test_emits_requested_preference_columns(self, ctx) -> None:
        results = ctx.run(ctx.strategies_from_config())
        frame = sn.summarise(results, ctx.cfg, ctx.base_spec,
                             gammas=[2.0, 7.0], ies=[1.0])
        assert "cec_crra_gamma2" in frame.columns
        assert "cec_crra_gamma7" in frame.columns
        assert "cec_ez_gamma7_psi1" in frame.columns
        assert frame["cec_crra_gamma2"].gt(0).all()

    def test_extra_columns_are_prepended(self, ctx) -> None:
        results = ctx.run(ctx.strategies_from_config())
        frame = sn.summarise(results, ctx.cfg, ctx.base_spec,
                             extra={"knob": 0.5})
        assert (frame["knob"] == 0.5).all()
        assert list(frame.columns)[0] == "knob"


class TestAdvantage:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "knob": [1, 1, 2, 2],
            "strategy": ["balanced_all_equity", "target_date_fund",
                         "balanced_all_equity", "target_date_fund"],
            "cec": [1.10, 1.00, 0.90, 1.00],
        })

    def test_computes_percentage_advantage_per_group(self) -> None:
        out = sn.advantage(self._frame(), "cec",
                           incumbents=["target_date_fund"], group=["knob"])
        assert out.loc[out.knob == 1, "advantage_vs_target_date_fund_pct"] \
            .iloc[0] == pytest.approx(10.0)
        assert out.loc[out.knob == 2, "advantage_vs_target_date_fund_pct"] \
            .iloc[0] == pytest.approx(-10.0)

    def test_flags_where_the_challenger_loses(self) -> None:
        out = sn.advantage(self._frame(), "cec",
                           incumbents=["target_date_fund"], group=["knob"])
        assert bool(out.loc[out.knob == 1, "challenger_wins_all"].iloc[0])
        assert not bool(out.loc[out.knob == 2, "challenger_wins_all"].iloc[0])


class TestOptimalAllocation:
    def test_picks_the_argmax_per_risk_aversion(self) -> None:
        frame = pd.DataFrame({
            "domestic_share": [0.0, 0.5, 1.0],
            "cec_crra_gamma5": [1.0, 1.2, 0.9],
            "prob_ruin": [0.2, 0.1, 0.3],
        })
        out = sn.optimal_allocation(frame, "domestic_share", [5.0])
        assert out["optimal_domestic_share"].iloc[0] == pytest.approx(0.5)
        assert out["cec_at_optimum"].iloc[0] == pytest.approx(1.2)

    def test_skips_risk_aversions_not_present(self) -> None:
        frame = pd.DataFrame({"domestic_share": [0.0],
                              "cec_crra_gamma5": [1.0], "prob_ruin": [0.1]})
        assert sn.optimal_allocation(frame, "domestic_share", [99.0]).empty


class TestCrossover:
    def _frame(self, challenger_cec, incumbent_cec) -> pd.DataFrame:
        gammas = [1.0, 2.0, 5.0, 10.0]
        return pd.DataFrame({
            "risk_aversion": gammas * 2,
            "strategy": ["balanced_all_equity"] * 4 + ["target_date_fund"] * 4,
            "cec": list(challenger_cec) + list(incumbent_cec),
        })

    def test_reports_infinity_when_the_challenger_always_leads(self) -> None:
        out = sn.crossover_risk_aversion(
            self._frame([1.4, 1.3, 1.1, 0.9], [1.2, 1.1, 0.95, 0.8]),
            incumbents=["target_date_fund"])
        assert np.isinf(out["crossover_risk_aversion"].iloc[0])
        assert bool(out["challenger_leads_at_max_gamma"].iloc[0])

    def test_interpolates_a_finite_crossover(self) -> None:
        # Gap goes +0.2, +0.1, -0.1, -0.3: it crosses between gamma 2 and 5.
        out = sn.crossover_risk_aversion(
            self._frame([1.4, 1.3, 1.0, 0.7], [1.2, 1.2, 1.1, 1.0]),
            incumbents=["target_date_fund"])
        crossover = float(out["crossover_risk_aversion"].iloc[0])
        assert 2.0 < crossover < 5.0
        assert not bool(out["challenger_leads_at_max_gamma"].iloc[0])


class TestSafeWithdrawalRate:
    def _curve(self, ruin) -> pd.DataFrame:
        rates = [0.01, 0.02, 0.03, 0.04, 0.05]
        return pd.DataFrame({
            "withdrawal_rate": rates,
            "strategy": ["s"] * len(rates),
            "label": ["S"] * len(rates),
            "prob_ruin": ruin,
        })

    def test_interpolates_the_target_ruin_crossing(self) -> None:
        out = sn.safe_withdrawal_rates(
            self._curve([0.01, 0.03, 0.07, 0.15, 0.30]), target_ruin=0.05)
        rate = float(out["safe_withdrawal_rate_at_5%_ruin"].iloc[0])
        assert 0.02 < rate < 0.03

    def test_returns_nan_when_never_safe(self) -> None:
        out = sn.safe_withdrawal_rates(
            self._curve([0.20, 0.30, 0.40, 0.50, 0.60]), target_ruin=0.05)
        assert np.isnan(out["safe_withdrawal_rate_at_5%_ruin"].iloc[0])

    def test_returns_the_grid_max_when_always_safe(self) -> None:
        out = sn.safe_withdrawal_rates(
            self._curve([0.001, 0.002, 0.003, 0.004, 0.005]), target_ruin=0.05)
        assert out["safe_withdrawal_rate_at_5%_ruin"].iloc[0] == pytest.approx(0.05)

    def test_also_reports_ruin_at_four_percent(self) -> None:
        out = sn.safe_withdrawal_rates(
            self._curve([0.01, 0.03, 0.07, 0.15, 0.30]))
        assert out["ruin_at_4pct"].iloc[0] == pytest.approx(0.15)


class TestTornado:
    def _sweep(self, advantages) -> pd.DataFrame:
        rows = []
        for knob, adv in enumerate(advantages):
            rows.append({"knob": knob, "strategy": "balanced_all_equity",
                         "label": "C", "cec": 1.0 + adv})
            rows.append({"knob": knob, "strategy": "target_date_fund",
                         "label": "T", "cec": 1.0})
        return pd.DataFrame.from_records(rows)

    def test_summarises_the_advantage_range(self) -> None:
        out = sn.tornado({"Knob": (self._sweep([0.05, 0.10, 0.20]), "knob")},
                         "cec", incumbents=["target_date_fund"])
        row = out.iloc[0]
        assert row["n_settings"] == 3
        assert row["min_advantage_pct"] == pytest.approx(5.0)
        assert row["max_advantage_pct"] == pytest.approx(20.0)
        assert bool(row["challenger_always_wins"])
        assert row["settings_lost"] == 0

    def test_counts_reversals(self) -> None:
        out = sn.tornado({"Knob": (self._sweep([0.05, -0.10]), "knob")},
                         "cec", incumbents=["target_date_fund"])
        row = out.iloc[0]
        assert not bool(row["challenger_always_wins"])
        assert row["settings_lost"] == 1

    def test_skips_frames_without_the_metric(self) -> None:
        frame = pd.DataFrame({"knob": [1], "strategy": ["x"], "label": ["X"]})
        assert sn.tornado({"K": (frame, "knob")}, "cec").empty

    def test_overall_verdict_aggregates(self) -> None:
        out = sn.tornado({"A": (self._sweep([0.05, 0.10]), "knob"),
                          "B": (self._sweep([0.02, -0.03]), "knob")},
                         "cec", incumbents=["target_date_fund"])
        verdict = sn.overall_verdict(out)
        assert verdict["n_settings"] == 4
        assert verdict["n_lost"] == 1
        assert not verdict["always_wins"]

    def test_overall_verdict_handles_an_empty_frame(self) -> None:
        verdict = sn.overall_verdict(pd.DataFrame())
        assert verdict["n_settings"] == 0 and not verdict["always_wins"]


class TestSweepValidation:
    def test_rejects_an_unknown_lifecycle_field(self, ctx) -> None:
        with pytest.raises(ValueError, match="unknown lifecycle field"):
            sn.sweep_lifecycle_field(ctx, ctx.strategies_from_config(),
                                     "risk_appetite", [1, 2])

    def test_lifecycle_sweep_returns_a_row_per_point_and_strategy(self, ctx
                                                                  ) -> None:
        strategies = ctx.strategies_from_config()
        out = sn.sweep_lifecycle_field(ctx, strategies, "savings_rate",
                                       [0.05, 0.10, 0.15])
        assert len(out) == 3 * len(strategies)
        assert set(out["savings_rate"]) == {0.05, 0.10, 0.15}

    def test_domestic_sweep_is_monotone_in_the_swept_column(self, ctx) -> None:
        out = sn.sweep_domestic_share(ctx, grid=[0.0, 0.5, 1.0])
        assert list(out["domestic_share"]) == [0.0, 0.5, 1.0]
