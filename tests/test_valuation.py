"""Tests for the starting-valuation study.

The load-bearing property of this module is that nothing conditions on
information the investor could not have had. Most of the tests below exist to
attack that claim rather than to confirm it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import bootstrap as bs
from src import lifecycle as lc
from src import valuation as vln


@pytest.fixture()
def toy_jst() -> pd.DataFrame:
    """Two countries with dividend and capital-gain series known by hand."""
    years = np.arange(2000, 2010)
    rows = []
    for k, iso in enumerate(("AAA", "BBB")):
        for i, year in enumerate(years):
            rows.append({
                "iso": iso,
                "year": int(year),
                # D_t / P_(t-1): rises through the sample so the lag is visible.
                "eq_dp": 0.02 + 0.01 * i + 0.005 * k,
                "eq_capgain": 0.10,
                "eq_tr": 0.12,
            })
    return pd.DataFrame(rows)


class TestTrailingYield:
    def test_is_last_years_dividend_over_last_years_price(self, toy_jst) -> None:
        years = np.arange(2000, 2010)
        out = vln.trailing_yield(toy_jst, ["AAA"], years)
        # Row t holds D_(t-1)/P_(t-1) = eq_dp_(t-1) / (1 + eq_capgain_(t-1)).
        for i in range(1, years.size):
            expected = (0.02 + 0.01 * (i - 1)) / 1.10
            assert out[i, 0] == pytest.approx(expected)

    def test_first_row_is_undefined(self, toy_jst) -> None:
        out = vln.trailing_yield(toy_jst, ["AAA"], np.arange(2000, 2010))
        assert np.isnan(out[0, 0]), "no year precedes the first, so no yield"

    def test_row_uses_the_previous_year_not_the_current_one(self, toy_jst) -> None:
        """The distinguishing test: change year t, row t must not move."""
        years = np.arange(2000, 2010)
        before = vln.trailing_yield(toy_jst, ["AAA"], years)
        bumped = toy_jst.copy()
        bumped.loc[bumped["year"] == 2005, "eq_dp"] = 5.0
        after = vln.trailing_yield(bumped, ["AAA"], years)
        row = int(np.flatnonzero(years == 2005)[0])
        assert after[row, 0] == pytest.approx(before[row, 0])
        # ... while the row that legitimately consumes it does move.
        assert after[row + 1, 0] != pytest.approx(before[row + 1, 0])

    def test_missing_country_yields_nan_column(self, toy_jst) -> None:
        out = vln.trailing_yield(toy_jst, ["ZZZ"], np.arange(2000, 2010))
        assert np.isnan(out).all()


class TestNoLookAhead:
    def test_accepts_the_lagged_construction(self, toy_jst) -> None:
        assert vln.depends_only_on_past(
            toy_jst, ["AAA", "BBB"], np.arange(2000, 2010), 2005)

    def test_rejects_a_leaking_construction(self, toy_jst) -> None:
        """The positive control.

        A check that has never been shown to fail is not evidence. Hand it the
        contemporaneous yield -- the mistake this module exists to avoid -- and
        it must reject it.
        """
        def leaking(jst, isos, years):
            out = np.full((len(years), len(isos)), np.nan)
            for j, iso in enumerate(isos):
                block = jst[jst["iso"] == iso].set_index("year")
                dp = block["eq_dp"].reindex(years).to_numpy(dtype=float)
                gain = block["eq_capgain"].reindex(years).to_numpy(dtype=float)
                out[:, j] = dp / (1.0 + gain)     # no lag: uses year t
            return out

        assert not vln.depends_only_on_past(
            toy_jst, ["AAA", "BBB"], np.arange(2000, 2010), 2005,
            builder=leaking)

    def test_holds_on_the_real_panel(self, real_panel_or_skip) -> None:
        panel, jst = real_panel_or_skip
        probes = [int(y) for y in np.linspace(int(panel.years[5]),
                                              int(panel.years[-2]), 6)]
        for year in probes:
            assert vln.depends_only_on_past(jst, panel.countries,
                                            panel.years, year), year


class TestInternationalSleeve:
    def test_is_the_leave_one_out_mean(self) -> None:
        domestic = np.array([[0.02, 0.04, 0.06]])
        out = vln.international_yield(domestic)
        np.testing.assert_allclose(
            out, np.array([[(0.04 + 0.06) / 2, (0.02 + 0.06) / 2,
                            (0.02 + 0.04) / 2]]))

    def test_matches_an_equal_money_portfolios_yield(self) -> None:
        """Why the mean and not the median.

        Hold one dollar in each of three markets. The portfolio's dividend
        yield is total dividends over total price, which for equal money is the
        plain mean of the constituent yields.
        """
        yields = np.array([0.01, 0.05, 0.09])
        dividends = yields.sum()            # one dollar of price in each
        assert dividends / 3.0 == pytest.approx(float(yields.mean()))
        # A fourth country's sleeve holds exactly those three markets, so its
        # sleeve yield is that same portfolio yield.
        leg = vln.international_yield(np.array([[*yields, 0.04]]))
        assert leg[0, 3] == pytest.approx(float(yields.mean()))

    def test_skips_countries_without_a_yield(self) -> None:
        domestic = np.array([[0.02, np.nan, 0.06]])
        out = vln.international_yield(domestic)
        assert out[0, 0] == pytest.approx(0.06)
        assert np.isnan(out[0, 1]), "a country with no yield has no own row"

    def test_a_lone_country_has_no_sleeve(self) -> None:
        assert np.isnan(vln.international_yield(np.array([[0.03]]))).all()

    def test_median_variant_resists_a_single_outlier(self) -> None:
        domestic = np.array([[0.03, 0.03, 0.03, 0.03, 9.0]])
        mean = vln.international_yield(domestic)
        median = vln.international_yield_median(domestic)
        assert mean[0, 0] > 1.0, "one distressed market drags the mean"
        assert median[0, 0] == pytest.approx(0.03)

    def test_blend_weights_the_two_legs(self) -> None:
        dom = np.array([[0.02]])
        intl = np.array([[0.06]])
        assert vln.blended_yield(dom, intl, 0.5)[0, 0] == pytest.approx(0.04)
        assert vln.blended_yield(dom, intl, 1.0)[0, 0] == pytest.approx(0.02)


class TestBuckets:
    def test_terciles_are_balanced(self) -> None:
        starting = np.linspace(0.01, 0.09, 900)
        index, meta = vln.bucket_paths(starting)
        assert meta["counts"] == [300, 300, 300]
        assert meta["cuts"] == sorted(meta["cuts"])

    def test_cheap_bucket_holds_the_high_yields(self) -> None:
        starting = np.linspace(0.01, 0.09, 900)
        index, meta = vln.bucket_paths(starting)
        assert starting[index == 2].min() > starting[index == 0].max(), (
            "a high dividend yield is a low price: 'Cheap' must be the top "
            "tercile of the yield"
        )

    def test_paths_without_a_yield_are_unassigned(self) -> None:
        starting = np.array([0.01, np.nan, 0.05, 0.09])
        index, meta = vln.bucket_paths(starting)
        assert index[1] == -1
        assert meta["unassigned"] == 1
        assert sum(meta["counts"]) == 3, "unassigned paths enter no bucket"


class TestLocate:
    def test_percentile_of_the_distribution(self) -> None:
        reference = np.arange(100, dtype=float)
        assert vln.locate(25.0, reference) == pytest.approx(25.0)

    def test_ignores_missing_values(self) -> None:
        reference = np.array([0.0, np.nan, 1.0, np.nan])
        assert vln.locate(0.5, reference) == pytest.approx(50.0)

    def test_undefined_against_an_empty_reference(self) -> None:
        assert np.isnan(vln.locate(1.0, np.array([np.nan])))


class TestPredictivePower:
    def test_recovers_a_planted_relationship(self) -> None:
        """High yields are followed by high returns, by construction."""
        rng = np.random.default_rng(0)
        yields = rng.uniform(0.01, 0.09, (200, 4))
        returns = (yields - 0.05) + rng.normal(0.0, 0.01, (200, 4))
        frame = vln.predictive_power(yields, returns, horizons=(1, 5))
        by_h = frame.set_index("horizon_years")
        # At one year the planted relationship is nearly the whole signal.
        assert float(by_h.loc[1, "correlation"]) > 0.8
        # Averaging five independent draws dilutes it but cannot reverse it.
        assert float(by_h.loc[5, "correlation"]) > 0.3
        assert (frame["gap"] > 0).all()

    def test_reports_no_gap_when_there_is_none(self) -> None:
        rng = np.random.default_rng(1)
        yields = rng.uniform(0.01, 0.09, (300, 4))
        returns = rng.normal(0.05, 0.15, (300, 4))
        frame = vln.predictive_power(yields, returns, horizons=(1,))
        assert abs(float(frame["correlation"].iloc[0])) < 0.15

    def test_skips_horizons_without_enough_data(self) -> None:
        yields = np.full((10, 1), 0.04)
        returns = np.full((10, 1), 0.05)
        assert vln.predictive_power(yields, returns, horizons=(9,)).empty


class TestPathAlignment:
    """The invariant the whole step rests on.

    Bucket ``i`` is computed from a re-drawn chunk stream; outcome row ``i``
    came from step 3's. If those two streams ever fell out of step the
    conditioning would silently compare unrelated lifetimes, and the symptom
    would be a null result rather than an error.
    """

    def test_redrawing_reproduces_the_same_paths(self, toy_panel,
                                                 toy_config) -> None:
        sampler = bs.from_config(toy_panel, toy_config)
        first = np.concatenate([c.calendar_index[:, 0]
                                for c in sampler.chunks(200, 100)])
        second = np.concatenate([c.calendar_index[:, 0]
                                 for c in sampler.chunks(200, 100)])
        np.testing.assert_array_equal(first, second)

    def test_starting_yield_uses_only_the_first_window(self, toy_panel,
                                                       toy_config) -> None:
        sampler = bs.from_config(toy_panel, toy_config)
        paths = sampler.sample(200, chunk_size=100)
        blended = np.arange(
            toy_panel.n_years * toy_panel.n_countries, dtype=float
        ).reshape(toy_panel.n_years, toy_panel.n_countries)
        starting = vln.path_starting_yield(paths, blended)
        expected = blended[np.asarray(paths.calendar_index)[:, 0],
                           np.asarray(paths.domestic_country)[:, 0]]
        np.testing.assert_allclose(starting, expected)

    def test_bucket_index_lines_up_with_outcome_rows(self, toy_panel,
                                                     toy_config) -> None:
        """End-to-end: the paths behind row ``i`` are the paths bucket ``i``
        was computed from.

        Checked through the simulator rather than around it -- the first-year
        portfolio return of a 50/50 outcome must equal the 50/50 blend of the
        first-year returns of the very chunk stream the buckets came from.
        """
        sampler = bs.from_config(toy_panel, toy_config)
        spec = lc.spec_from_config(toy_config)
        strategies = lc.build_strategies(toy_config, spec)
        results = lc.run_chunked(sampler, strategies, spec, 200, 100)

        redrawn_dom, redrawn_intl = [], []
        for chunk in sampler.chunks(200, 100):
            redrawn_dom.append(chunk.dom_eq[:, 0])
            redrawn_intl.append(chunk.intl_eq[:, 0])
        dom = np.concatenate(redrawn_dom)
        intl = np.concatenate(redrawn_intl)

        realised = results["all_equity"].portfolio_return[:, 0]
        np.testing.assert_allclose(realised, 0.5 * dom + 0.5 * intl,
                                   rtol=1e-10, atol=1e-12)

    def test_subsetting_an_outcome_keeps_rows_together(self, toy_panel,
                                                       toy_config) -> None:
        sampler = bs.from_config(toy_panel, toy_config)
        spec = lc.spec_from_config(toy_config)
        strategies = lc.build_strategies(toy_config, spec)
        results = lc.run_chunked(sampler, strategies, spec, 200, 100)
        outcome = results["all_equity"]
        mask = np.zeros(200, dtype=bool)
        mask[[3, 17, 99]] = True
        sub = vln.outcome_subset(outcome, mask)
        assert sub.consumption.shape[0] == 3
        np.testing.assert_allclose(sub.consumption, outcome.consumption[mask])
        np.testing.assert_allclose(sub.ruin, outcome.ruin[mask])
        np.testing.assert_allclose(sub.wealth_at_retirement,
                                   outcome.wealth_at_retirement[mask])
