"""Tests for the cross-country joint block bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from src import bootstrap as bs
from src.data_loader import CORE_SERIES


def make_sampler(panel, **kwargs):
    defaults = dict(horizon=12, mean_block=4.0, min_block=1, max_block=10,
                    country_draw="per_lifetime", country_weighting="history",
                    seed=11)
    defaults.update(kwargs)
    return bs.MultiCountryBlockBootstrap(panel, **defaults)


class TestRunLengths:
    def test_counts_forward_runs(self) -> None:
        available = np.array([[True], [True], [False], [True], [True], [True]])
        np.testing.assert_array_equal(
            bs.run_lengths(available).ravel(), [2, 1, 0, 3, 2, 1])

    def test_all_missing_is_all_zero(self) -> None:
        assert bs.run_lengths(np.zeros((5, 2), dtype=bool)).sum() == 0

    def test_matches_panel_gap(self, toy_panel) -> None:
        runs = bs.run_lengths(toy_panel.available)
        # Country 1 is missing years 6-8, so year 5 can only start a 1-year
        # block and year 9 starts an 11-year run to the end.
        assert runs[5, 1] == 1
        assert runs[9, 1] == 11
        assert runs[0, 0] == 20


class TestBlockIndex:
    def test_every_enumerated_start_is_admissible(self, toy_panel) -> None:
        index = bs.build_block_index(toy_panel, max_length=10)
        for c in range(toy_panel.n_countries):
            for length in range(1, index.max_length + 1):
                lo = index.offset[c, length]
                starts = index.starts_flat[lo:lo + index.count[c, length]]
                for t in starts:
                    assert toy_panel.available[t:t + length, c].all()

    def test_counts_shrink_with_length(self, toy_panel) -> None:
        index = bs.build_block_index(toy_panel, max_length=10)
        counts = index.count[0, 1:]
        assert np.all(np.diff(counts) <= 0)

    def test_max_run_is_respected(self, toy_panel) -> None:
        index = bs.build_block_index(toy_panel, max_length=20)
        assert index.max_run[0] == 20
        assert index.max_run[1] == 11


class TestCountryProbabilities:
    def test_uniform(self, toy_panel) -> None:
        probs = bs.country_probabilities(toy_panel, "uniform")
        np.testing.assert_allclose(probs, np.full(3, 1 / 3))

    def test_history_weighted_matches_usable_years(self, toy_panel) -> None:
        probs = bs.country_probabilities(toy_panel, "history")
        expected = toy_panel.available.sum(axis=0) / toy_panel.available.sum()
        np.testing.assert_allclose(probs, expected)

    def test_unknown_weighting_raises(self, toy_panel) -> None:
        with pytest.raises(ValueError, match="country_weighting"):
            bs.country_probabilities(toy_panel, "cap_weighted")


class TestSampling:
    def test_shapes_and_finiteness(self, toy_panel) -> None:
        paths = make_sampler(toy_panel).sample(500, chunk_size=250)
        assert paths.n_paths == 500 and paths.horizon == 12
        for key in CORE_SERIES:
            assert paths.series(key).shape == (500, 12)
            assert np.isfinite(paths.series(key)).all()

    def test_every_drawn_value_exists_in_the_panel(self, toy_panel) -> None:
        paths = make_sampler(toy_panel).sample(300)
        for key in CORE_SERIES:
            source = toy_panel.series(key)
            expected = source[paths.calendar_index, paths.domestic_country]
            np.testing.assert_allclose(paths.series(key), expected)

    def test_blocks_never_span_a_gap(self, toy_panel) -> None:
        paths = make_sampler(toy_panel, mean_block=6.0).sample(2000)
        drawn = toy_panel.available[paths.calendar_index,
                                    paths.domestic_country]
        assert drawn.all()

    def test_calendar_index_is_contiguous_within_a_block(self,
                                                         toy_panel) -> None:
        paths = make_sampler(toy_panel).sample(400)
        same_block = np.diff(paths.block_id, axis=1) == 0
        steps = np.diff(paths.calendar_index.astype(int), axis=1)
        assert np.all(steps[same_block] == 1)

    def test_country_is_fixed_over_a_lifetime(self, toy_panel) -> None:
        paths = make_sampler(toy_panel, country_draw="per_lifetime").sample(300)
        assert (paths.domestic_country
                == paths.domestic_country[:, [0]]).all()

    def test_per_block_draw_can_switch_country(self, toy_panel) -> None:
        paths = make_sampler(toy_panel, country_draw="per_block").sample(500)
        assert not (paths.domestic_country
                    == paths.domestic_country[:, [0]]).all()

    def test_reproducible_given_the_seed(self, toy_panel) -> None:
        a = make_sampler(toy_panel, seed=99).sample(200)
        b = make_sampler(toy_panel, seed=99).sample(200)
        np.testing.assert_array_equal(a.dom_eq, b.dom_eq)

    def test_different_seeds_differ(self, toy_panel) -> None:
        a = make_sampler(toy_panel, seed=1).sample(200)
        b = make_sampler(toy_panel, seed=2).sample(200)
        assert not np.array_equal(a.dom_eq, b.dom_eq)

    def test_reproducible_for_a_fixed_chunk_size(self, toy_panel) -> None:
        # The documented contract is (seed, n_paths, chunk_size).
        a = make_sampler(toy_panel).sample(400, chunk_size=100)
        b = make_sampler(toy_panel).sample(400, chunk_size=100)
        np.testing.assert_array_equal(a.dom_eq, b.dom_eq)

    def test_chunking_does_not_change_the_sampling_distribution(
            self, toy_panel) -> None:
        # Re-cutting the stream gives a different sample from the same law,
        # so the moments must agree to within sampling error.
        one = make_sampler(toy_panel).sample(6000, chunk_size=6000)
        many = make_sampler(toy_panel).sample(6000, chunk_size=500)
        assert abs(one.dom_eq.mean() - many.dom_eq.mean()) < 0.01
        assert abs(one.dom_eq.std() - many.dom_eq.std()) < 0.01

    def test_fixed_block_length_is_honoured(self, toy_panel) -> None:
        paths = make_sampler(toy_panel, mean_block=3.0,
                             block_length_distribution="fixed").sample(200)
        # Country 0 has no gaps, so its blocks are exactly 3 long except
        # where the horizon truncates them.
        rows = np.flatnonzero(paths.domestic_country[:, 0] == 0)[:50]
        for row in rows:
            ids = paths.block_id[row]
            _, counts = np.unique(ids, return_counts=True)
            assert set(counts.tolist()) <= {3}

    def test_short_history_country_gets_truncated_blocks(self,
                                                         toy_panel) -> None:
        # Country 1's longest run is 11 years, so a 20-year request truncates.
        sampler = make_sampler(toy_panel, mean_block=20.0, max_block=20,
                               block_length_distribution="fixed")
        paths = sampler.sample(400)
        drawn = toy_panel.available[paths.calendar_index,
                                    paths.domestic_country]
        assert drawn.all()


class TestValidation:
    def test_rejects_unknown_country_draw(self, toy_panel) -> None:
        with pytest.raises(ValueError, match="country_draw"):
            make_sampler(toy_panel, country_draw="per_decade")

    def test_rejects_unknown_block_distribution(self, toy_panel) -> None:
        with pytest.raises(ValueError, match="block_length_distribution"):
            make_sampler(toy_panel, block_length_distribution="poisson")

    def test_rejects_non_positive_horizon(self, toy_panel) -> None:
        with pytest.raises(ValueError, match="horizon"):
            make_sampler(toy_panel, horizon=0)


class TestDiagnostics:
    def test_mean_preservation_is_within_sampling_error(self,
                                                        toy_panel) -> None:
        sampler = make_sampler(toy_panel, horizon=30, seed=5)
        diag = bs.diagnose(sampler, n_paths=6000, chunk_size=2000)
        for _, row in diag["moments"].iterrows():
            assert abs(row["bootstrap_mean"] - row["panel_pooled_mean"]) < 0.02
            assert 0.75 < row["std_ratio"] < 1.25

    def test_cross_asset_correlation_is_preserved(self, toy_panel) -> None:
        sampler = make_sampler(toy_panel, horizon=30, seed=5)
        diag = bs.diagnose(sampler, n_paths=6000, chunk_size=2000)
        assert np.abs(diag["correlation_gap"].to_numpy()).max() < 0.12

    def test_country_frequencies_track_the_target(self, toy_panel) -> None:
        sampler = make_sampler(toy_panel, seed=5)
        diag = bs.diagnose(sampler, n_paths=8000, chunk_size=4000)
        gap = (diag["countries"]["realised_frequency"]
               - diag["countries"]["target_probability"]).abs().max()
        assert gap < 0.02

    def test_realised_block_length_is_near_target(self, toy_panel) -> None:
        sampler = make_sampler(toy_panel, horizon=60, mean_block=5.0, seed=5)
        diag = bs.diagnose(sampler, n_paths=4000, chunk_size=2000)
        blocks = diag["blocks"].set_index("statistic")["value"]
        assert 2.0 < blocks["mean_length"] <= 5.0

    def test_shorter_blocks_kill_persistence(self, persistent_panel,
                                             toy_config) -> None:
        # The property only has content when the source data is persistent,
        # so this runs on a panel built from an AR(1) process.
        cfg = dict(toy_config)
        cfg["bootstrap"] = dict(toy_config["bootstrap"], horizon_years=40)
        table = bs.block_length_sensitivity(
            persistent_panel, cfg, grid=[1.0, 12.0], n_paths=4000,
            chunk_size=2000)
        short, long = table.iloc[0], table.iloc[1]
        assert abs(short["within_path_ar1"]) < 0.05
        assert long["within_path_ar1"] > 0.20
        assert short["dom_eq_sd_annualised"] < long["dom_eq_sd_annualised"]
