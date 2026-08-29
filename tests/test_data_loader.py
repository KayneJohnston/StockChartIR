"""Tests for real-return construction and panel assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl


class TestDeflate:
    def test_matches_the_closed_form(self) -> None:
        nominal = np.array([0.10, 0.00, -0.20])
        inflation = np.array([0.05, 0.05, 0.05])
        expected = (1 + nominal) / (1 + inflation) - 1
        np.testing.assert_allclose(dl.deflate(nominal, inflation), expected)

    def test_zero_inflation_is_a_no_op(self) -> None:
        nominal = np.array([0.07, -0.31, 0.0])
        np.testing.assert_allclose(dl.deflate(nominal, np.zeros(3)), nominal)

    def test_hyperinflation_is_floored_not_infinite(self) -> None:
        # Germany 1923: nominal bonds returned 0 against 1e9 inflation.
        real = dl.deflate(np.array([0.0]), np.array([1.0e9]))
        assert np.isfinite(real).all()
        assert real[0] == pytest.approx(dl.GROSS_RETURN_FLOOR - 1.0, abs=1e-9)

    def test_nan_propagates(self) -> None:
        assert np.isnan(dl.deflate(np.array([np.nan]), np.array([0.02])))[0]


class TestInternationalLeg:
    def test_leave_one_out_average(self) -> None:
        # Three countries, no FX movement, no inflation: the international
        # leg for country i is the plain average of the other two.
        usd_gross = np.array([[1.10, 1.20, 1.30]])
        fx_gain = np.ones((1, 3))
        inflation = np.zeros((1, 3))
        out = dl.build_international_leg(usd_gross, fx_gain, inflation)
        np.testing.assert_allclose(
            out, np.array([[(1.20 + 1.30) / 2 - 1,
                            (1.10 + 1.30) / 2 - 1,
                            (1.10 + 1.20) / 2 - 1]]))

    def test_currency_translation(self) -> None:
        # Country 0's currency halves against the USD: fx_gain = 0.5, so a
        # foreign gross USD return of 1.0 becomes 2.0 in local terms.
        usd_gross = np.array([[np.nan, 1.0]])
        fx_gain = np.array([[0.5, 1.0]])
        inflation = np.zeros((1, 2))
        out = dl.build_international_leg(usd_gross, fx_gain, inflation)
        assert out[0, 0] == pytest.approx(1.0)

    def test_domestic_inflation_deflates_the_foreign_leg(self) -> None:
        usd_gross = np.array([[np.nan, 1.10]])
        fx_gain = np.ones((1, 2))
        inflation = np.array([[0.10, 0.0]])
        out = dl.build_international_leg(usd_gross, fx_gain, inflation)
        assert out[0, 0] == pytest.approx(1.10 / 1.10 - 1.0)

    def test_single_country_has_no_international_leg(self) -> None:
        out = dl.build_international_leg(
            np.array([[1.1]]), np.ones((1, 1)), np.zeros((1, 1)))
        assert np.isnan(out).all()

    def test_winsorisation_clips_only_the_tails(self) -> None:
        rng = np.random.default_rng(0)
        usd_gross = 1.0 + rng.normal(0.05, 0.1, (400, 6))
        usd_gross[10, 0] = 60.0            # an administered-rate artefact
        fx_gain = np.ones((400, 6))
        inflation = np.zeros((400, 6))
        raw = dl.build_international_leg(usd_gross, fx_gain, inflation)
        clipped = dl.build_international_leg(usd_gross, fx_gain, inflation,
                                             winsor_pct=0.5)
        assert clipped.max() < raw.max()
        # The bulk of the distribution is untouched.
        assert (np.isclose(raw, clipped)).mean() > 0.98


class TestRealExchangeRate:
    def test_common_inflation_leaves_the_index_flat(self) -> None:
        cpi = np.array([[100.0, 100.0], [110.0, 110.0], [121.0, 121.0]])
        xrusd = np.ones((3, 2))
        index = dl.build_real_exchange_rate(cpi, xrusd)
        np.testing.assert_allclose(index, np.ones((3, 2)), atol=1e-12)

    def test_relative_inflation_moves_the_index(self) -> None:
        cpi = np.array([[100.0, 100.0], [120.0, 100.0]])
        xrusd = np.ones((2, 2))
        index = dl.build_real_exchange_rate(cpi, xrusd)
        assert index[1, 0] > 1.0 > index[1, 1]


class TestPanelContainer:
    def test_round_trips_through_disk(self, toy_panel, tmp_path) -> None:
        path = toy_panel.save(tmp_path / "panel.npz")
        restored = dl.Panel.load(path)
        assert restored.countries == toy_panel.countries
        assert restored.provenance == toy_panel.provenance
        np.testing.assert_array_equal(restored.available, toy_panel.available)
        np.testing.assert_allclose(restored.dom_eq, toy_panel.dom_eq,
                                   equal_nan=True)

    def test_fingerprint_is_content_addressed(self, toy_panel) -> None:
        assert toy_panel.fingerprint() == toy_panel.fingerprint()
        mutated = dl.dataclasses.replace(
            toy_panel, dom_eq=toy_panel.dom_eq + 0.01)
        assert mutated.fingerprint() != toy_panel.fingerprint()

    def test_stacked_matches_series_order(self, toy_panel) -> None:
        stacked = toy_panel.stacked()
        assert stacked.shape == (toy_panel.n_years, toy_panel.n_countries,
                                 len(dl.CORE_SERIES))
        for i, key in enumerate(dl.CORE_SERIES):
            np.testing.assert_allclose(stacked[:, :, i], toy_panel.series(key),
                                       equal_nan=True)

    def test_unknown_series_raises(self, toy_panel) -> None:
        with pytest.raises(KeyError):
            toy_panel.series("gold")

    def test_subset_preserves_alignment(self, toy_panel) -> None:
        sub = toy_panel.subset(["CCC", "AAA"])
        assert sub.countries == ("CCC", "AAA")
        np.testing.assert_allclose(sub.dom_eq[:, 0],
                                   toy_panel.dom_eq[:, 2], equal_nan=True)


class TestDiagnostics:
    def test_summary_statistics_shape(self, toy_panel) -> None:
        summary = dl.summary_statistics(toy_panel)
        assert set(summary["series"]) == set(dl.CORE_SERIES)
        assert len(summary) == toy_panel.n_countries * len(dl.CORE_SERIES)
        gap_row = summary[(summary.iso == "GAP") & (summary.series == "dom_eq")]
        assert int(gap_row["n_years"].iloc[0]) == 17

    def test_coverage_matrix_counts_available_years(self, toy_panel) -> None:
        """Undecaded, the matrix is the availability mask transposed."""
        coverage = dl.coverage_matrix(toy_panel, decade=False)
        assert coverage.loc["AAA"].sum() == 20
        assert coverage.loc["GAP"].sum() == 17

    def test_decade_coverage_is_a_share_not_a_count(self, toy_panel) -> None:
        """A partial final decade must compare with the full ones.

        Counting raw years puts a one-year bucket at 1 against a scale of 10,
        which reads as missing data rather than as a short bucket.
        """
        coverage = dl.coverage_matrix(toy_panel, decade=True)
        assert coverage.to_numpy().max() <= 1.0
        assert (coverage.loc["AAA"] == 1.0).all(), (
            "a country with no gaps is fully covered in every decade it spans"
        )
        # Same information, different unit: share times bucket width is the
        # count the undecaded matrix reports.
        widths = np.array([int((toy_panel.years // 10 * 10 == d).sum())
                           for d in coverage.columns], dtype=float)
        assert float((coverage.loc["GAP"].to_numpy() * widths).sum()) == 17.0

    def test_monthly_disaggregation_preserves_annual_returns(self,
                                                             toy_panel) -> None:
        monthly = dl.to_monthly(toy_panel, seed=1)
        annual_log = np.log1p(toy_panel.dom_eq)
        rebuilt = np.log1p(monthly["dom_eq"]).reshape(
            toy_panel.n_years, 12, toy_panel.n_countries).sum(axis=1)
        mask = np.isfinite(annual_log)
        np.testing.assert_allclose(rebuilt[mask], annual_log[mask], atol=1e-9)
