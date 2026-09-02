"""Tests for the realised-cohort backtest.

The arithmetic here is simple; what is easy to get wrong is the bookkeeping.
A cohort that steps over a market closure is not a lifetime anyone lived, an
interval built by resampling cohorts would be far too narrow, and a win rate
counted by cohort silently gives the United States sixty-four votes and
Germany four. Each of those has a test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import cohorts as ch
from src import lifecycle as lc


class TestEnumeration:
    def test_a_cohort_never_steps_over_a_gap(self, toy_panel) -> None:
        # `toy_panel` has country GAP missing years 6-8 of 20.
        found = ch.enumerate_cohorts(toy_panel, horizon=5)
        for _, row in found.iterrows():
            start = int(row["start_index"])
            usable = toy_panel.available[start:start + 5,
                                         int(row["country_index"])]
            assert usable.all()

    def test_the_gap_costs_more_cohorts_than_it_costs_years(self,
                                                            toy_panel) -> None:
        counts = ch.enumerate_cohorts(toy_panel, horizon=5).groupby("iso").size()
        # Three missing years remove far more than three lifetimes.
        assert counts["GAP"] <= counts["AAA"] - 3

    def test_a_horizon_longer_than_the_record_yields_nothing(self,
                                                             toy_panel) -> None:
        assert len(ch.enumerate_cohorts(toy_panel, horizon=100)) == 0

    def test_the_census_covers_every_country(self, toy_panel) -> None:
        census = ch.census(toy_panel, horizon=5)
        assert set(census["iso"]) == set(toy_panel.countries)

    def test_the_census_longest_run_respects_the_gap(self, toy_panel) -> None:
        census = ch.census(toy_panel, horizon=5).set_index("iso")
        assert census.loc["GAP", "longest_unbroken"] < census.loc["AAA",
                                                                  "longest_unbroken"]


class TestEffectiveSample:
    def test_overlapping_cohorts_are_not_independent_evidence(self,
                                                              toy_panel) -> None:
        found = ch.effective_sample(toy_panel, horizon=5)
        assert found["n_independent"] < found["n_cohorts"]
        assert found["overlap_ratio"] > 1.0

    def test_the_independent_count_is_the_non_overlapping_one(self) -> None:
        # 20 clean years and a 5-year lifetime is exactly four lifetimes.
        from src.data_loader import Panel
        years = np.arange(2000, 2020)
        n = years.size
        ones = np.ones((n, 1))
        panel = Panel(years=years, countries=("ONE",), tier=("A",),
                      dom_eq=ones, intl_eq=ones, bond=ones, bill=ones,
                      inflation=ones,
                      real_exchange_rate=ones,
                      available=np.ones((n, 1), dtype=bool), name="clean",
                      provenance=("test",))
        assert ch.effective_sample(panel, horizon=5)["n_independent"] == 4


class TestPaths:
    def test_a_cohort_path_is_a_slice_of_history(self, toy_panel) -> None:
        found = ch.enumerate_cohorts(toy_panel, horizon=5)
        paths = ch.cohort_paths(toy_panel, found, horizon=5)
        row = found.iloc[3]
        expected = toy_panel.dom_eq[int(row["start_index"]):
                                    int(row["start_index"]) + 5,
                                    int(row["country_index"])]
        assert np.allclose(paths.dom_eq[3], expected)

    def test_each_lifetime_is_one_unbroken_block(self, toy_panel) -> None:
        found = ch.enumerate_cohorts(toy_panel, horizon=5)
        paths = ch.cohort_paths(toy_panel, found, horizon=5)
        # A block id that never changes within a row is what "not resampled"
        # means, and it is the property that distinguishes these paths from
        # every other set in the project.
        assert (paths.block_id == paths.block_id[:, :1]).all()

    def test_the_domestic_country_never_changes_mid_lifetime(self, toy_panel
                                                             ) -> None:
        found = ch.enumerate_cohorts(toy_panel, horizon=5)
        paths = ch.cohort_paths(toy_panel, found, horizon=5)
        assert (paths.domestic_country == paths.domestic_country[:, :1]).all()

    def test_an_empty_cohort_set_is_an_error_not_an_empty_run(self, toy_panel
                                                              ) -> None:
        with pytest.raises(ValueError, match="too short"):
            ch.cohort_paths(toy_panel, ch.enumerate_cohorts(toy_panel, 100), 100)


class TestInference:
    @staticmethod
    def _detail(per_country):
        rows = []
        for iso, gaps in per_country.items():
            for i, g in enumerate(gaps):
                rows.append({"iso": iso, "start_year": 1900 + i,
                             "gap_pct": g, "first_wins": g > 0})
        return pd.DataFrame.from_records(rows)

    def test_the_interval_resamples_countries_not_cohorts(self) -> None:
        # One country dissenting loudly must widen the interval, which it
        # cannot do if the bootstrap resamples the pooled rows.
        agree = self._detail({f"C{i}": [5.0] * 20 for i in range(8)})
        dissent = self._detail({**{f"C{i}": [5.0] * 20 for i in range(7)},
                                "C7": [-40.0] * 20})
        wide = ch.cluster_bootstrap(dissent, n_boot=400)
        narrow = ch.cluster_bootstrap(agree, n_boot=400)
        assert wide["se"] > narrow["se"]

    def test_a_unanimous_panel_excludes_zero(self) -> None:
        detail = self._detail({f"C{i}": [3.0 + i] * 10 for i in range(6)})
        assert ch.cluster_bootstrap(detail, n_boot=400)["excludes_zero"]

    def test_the_country_win_rate_ignores_how_many_cohorts_each_has(self
                                                                    ) -> None:
        # One market with sixty losing cohorts and five with two winning ones.
        detail = self._detail({"BIG": [-1.0] * 60,
                               **{f"S{i}": [1.0] * 2 for i in range(5)}})
        signs = ch.sign_test(detail)
        assert signs["cohort_win_rate"] < 0.5 < signs["country_win_rate"]
        assert signs["countries_won"] == 5

    def test_the_bootstrap_is_reproducible(self) -> None:
        detail = self._detail({f"C{i}": [float(i)] * 5 for i in range(6)})
        first = ch.cluster_bootstrap(detail, n_boot=200, seed=7)
        second = ch.cluster_bootstrap(detail, n_boot=200, seed=7)
        assert first == second


class TestRealisedReturns:
    def test_the_annualised_return_compounds(self, toy_panel) -> None:
        found = ch.enumerate_cohorts(toy_panel, horizon=5)
        realised = ch.long_run_returns(toy_panel, found, horizon=5)
        row = found.iloc[0]
        window = toy_panel.dom_eq[int(row["start_index"]):
                                  int(row["start_index"]) + 5,
                                  int(row["country_index"])]
        expected = np.prod(1.0 + window) ** (1 / 5) - 1.0
        assert realised["domestic_annualised"].iloc[0] == pytest.approx(expected)

    def test_the_dispersion_summary_covers_both_legs(self, toy_panel) -> None:
        realised = ch.long_run_returns(
            toy_panel, ch.enumerate_cohorts(toy_panel, 5), 5)
        spread = ch.dispersion(realised)
        assert spread["worst_domestic_pp"] <= spread["best_domestic_pp"]
        assert 0.0 <= spread["share_sleeve_ahead"] <= 1.0


class TestAgainstTheRealPanel:
    def test_income_is_deterministic_so_the_spread_is_about_returns(
            self, real_config_or_skip) -> None:
        from src import data_loader as dl
        panel = dl.build_tier_a(real_config_or_skip)
        spec = lc.spec_from_config(real_config_or_skip)
        strategies = lc.build_strategies(real_config_or_skip, spec)
        found = ch.enumerate_cohorts(panel, spec.horizon)
        outcomes = ch.run(panel, found, spec, strategies)
        first = next(iter(outcomes.values()))
        working = first.consumption[:, :spec.n_working]
        # Every cohort earns the identical wage profile, so working-life
        # consumption is one row repeated.
        assert np.allclose(working, working[0], atol=1e-9)

    def test_the_war_torn_markets_contribute_the_fewest_lifetimes(
            self, real_config_or_skip) -> None:
        from src import data_loader as dl
        panel = dl.build_tier_a(real_config_or_skip)
        spec = lc.spec_from_config(real_config_or_skip)
        census = ch.census(panel, spec.horizon).set_index("iso")
        # The structural bias the section is built around: a lifetime cannot
        # step over a closure, so the markets that closed are the ones this
        # design can least see.
        assert census.loc["DEU", "cohorts"] < census.loc["USA", "cohorts"]
        assert census.loc["JPN", "cohorts"] < census.loc["GBR", "cohorts"]
