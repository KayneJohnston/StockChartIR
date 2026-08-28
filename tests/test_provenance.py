"""Tests for the data provenance audit.

The audit exists to stop the project from overstating its evidence, so the
tests that matter are the ones that would fire if it started understating the
problem: a simulated country counted as observed, or an international leg
reported as cleaner than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import provenance as pv


def _retier(panel: dl.Panel, tier, provenance, observed=None,
            name=None) -> dl.Panel:
    return dl.Panel(
        years=panel.years, countries=panel.countries, tier=tier,
        dom_eq=panel.dom_eq, intl_eq=panel.intl_eq, bond=panel.bond,
        bill=panel.bill, inflation=panel.inflation,
        real_exchange_rate=panel.real_exchange_rate,
        available=panel.available, name=name or panel.name,
        provenance=provenance, observed=observed or {})


@pytest.fixture()
def observed_panel(toy_panel) -> dl.Panel:
    """The toy panel with every country marked observed."""
    note = "JST/JKKST empirical 2000-2019; inflation 100% empirical"
    return _retier(toy_panel, ("A",) * len(toy_panel.countries),
                   (note,) * len(toy_panel.countries))


EMPIRICAL_NOTE = "JST/JKKST empirical 2000-2019; inflation 100% empirical"
SIMULATED_NOTE = ("Tier-B calibrated: inflation Clio-Infra CPI (80% of 20 "
                  "active years empirical, remainder factor-model); "
                  "equity/bond/bill simulated from donor AAA; market "
                  "inception 2000")


@pytest.fixture()
def mixed_panel(toy_panel) -> dl.Panel:
    """The toy panel relabelled so one of its three countries is simulated.

    No explicit masks: this is the fixture that checks the tier fallback, so
    the audit has to reach the same answer from the labels alone.
    """
    return _retier(toy_panel, ("A", "A", "C"),
                   (EMPIRICAL_NOTE, EMPIRICAL_NOTE, SIMULATED_NOTE),
                   name="mixed")


@pytest.fixture()
def partial_panel(toy_panel) -> dl.Panel:
    """One country whose rates were recovered but whose equity is simulated.

    The masks are explicit here, because the whole point of Tier B is that a
    country-level label cannot express it.
    """
    shape = toy_panel.available.shape
    observed = {"dom_eq": np.ones(shape, dtype=bool),
                "bond": np.ones(shape, dtype=bool),
                "bill": np.ones(shape, dtype=bool),
                "inflation": np.ones(shape, dtype=bool)}
    observed["dom_eq"][:, 2] = False            # equity simulated throughout
    return _retier(toy_panel, ("A", "A", "B"),
                   (EMPIRICAL_NOTE, EMPIRICAL_NOTE, SIMULATED_NOTE),
                   observed=observed, name="partial")


class TestFingerprints:
    def test_a_missing_file_is_reported_not_raised(self, tmp_path) -> None:
        digest = pv.file_digest(tmp_path / "nope.xlsx")
        assert digest["exists"] is False and digest["sha256"] == ""

    def test_a_real_file_hashes_reproducibly(self, tmp_path) -> None:
        path = tmp_path / "x.bin"
        path.write_bytes(b"provenance")
        first, second = pv.file_digest(path), pv.file_digest(path)
        assert first == second and first["bytes"] == 10


class TestCountryProvenance:
    def test_simulated_countries_are_named_as_such(self, mixed_panel) -> None:
        frame = pv.country_provenance(mixed_panel)
        simulated = frame[frame["tier"] == "C"]
        assert len(simulated) == 1
        assert simulated["returns_source"].iloc[0] == "factor model"
        assert int(simulated["returns_empirical_years"].iloc[0]) == 0

    def test_a_partly_observed_country_is_counted_per_series(
            self, partial_panel) -> None:
        """The reason cells replaced countries: this row is both things."""
        frame = pv.country_provenance(partial_panel).set_index("tier")
        row = frame.loc["B"]
        assert int(row["dom_eq_observed_years"]) == 0
        assert int(row["bond_observed_years"]) == int(row["usable_years"])
        assert int(row["bill_observed_years"]) == int(row["usable_years"])
        assert float(row["share_returns_observed"]) == pytest.approx(2 / 3)
        assert "recovered from published rates" in row["returns_source"]

    def test_observed_countries_count_all_their_years_as_empirical(
            self, mixed_panel) -> None:
        frame = pv.country_provenance(mixed_panel)
        observed = frame[frame["tier"] == "A"]
        assert (observed["returns_source"] == "JST/JKKST").all()
        assert (observed["returns_simulated_years"] == 0).all()
        # One cell per series per year, so an observed country's empirical
        # count is its usable years times the number of return series.
        assert (observed["returns_empirical_years"]
                == observed["usable_years"] * len(pv.RETURN_SERIES)).all()
        assert (observed["share_returns_observed"] == 1.0).all()

    def test_the_donor_is_recovered_from_the_note(self, mixed_panel) -> None:
        frame = pv.country_provenance(mixed_panel)
        assert frame[frame["tier"] == "C"]["donor"].iloc[0] == "AAA"

    def test_the_inflation_share_is_recovered_from_the_note(self, mixed_panel
                                                            ) -> None:
        frame = pv.country_provenance(mixed_panel)
        assert float(frame[frame["tier"] == "C"]
                     ["inflation_empirical_share"].iloc[0]) == pytest.approx(0.8)
        assert float(frame[frame["tier"] == "A"]
                     ["inflation_empirical_share"].iloc[0]) == pytest.approx(1.0)


class TestRecoveredSeries:
    def test_it_reports_only_the_partly_observed_countries(self, partial_panel
                                                           ) -> None:
        frame = pv.recovered_series(partial_panel)
        assert set(frame["iso"]) == {partial_panel.countries[2]}
        assert set(frame["series"]) == {"bond", "bill"}, (
            "equity is simulated for this country and must not be listed"
        )

    def test_it_counts_the_years_that_stopped_being_simulated(
            self, partial_panel) -> None:
        frame = pv.recovered_series(partial_panel)
        usable = int(partial_panel.available[:, 2].sum())
        assert int(frame["observed_years"].sum()) == 2 * usable

    def test_a_panel_with_nothing_recovered_gives_an_empty_frame(
            self, mixed_panel) -> None:
        frame = pv.recovered_series(mixed_panel)
        assert frame.empty
        assert "observed_years" in frame.columns, (
            "callers sum this column, so it must exist even when empty"
        )

    def test_the_span_comes_from_the_mask_not_the_calendar(self, partial_panel
                                                           ) -> None:
        frame = pv.recovered_series(partial_panel).set_index("series")
        years = np.asarray(partial_panel.years)
        column = partial_panel.available[:, 2]
        assert int(frame.loc["bond", "first_year"]) == int(years[column].min())
        assert int(frame.loc["bond", "last_year"]) == int(years[column].max())


class TestPanelShares:
    def test_shares_are_consistent_with_the_availability_mask(self, mixed_panel
                                                              ) -> None:
        summary = pv.panel_summary(mixed_panel)
        available = mixed_panel.available
        simulated = np.array(mixed_panel.tier) != "A"
        assert summary["country_years"] == int(available.sum())
        assert summary["country_years_simulated"] \
            == int(available[:, simulated].sum())
        assert summary["country_years_empirical"] \
            + summary["country_years_simulated"] == summary["country_years"]

    def test_an_all_observed_panel_reports_nothing_simulated(
            self, observed_panel) -> None:
        summary = pv.panel_summary(observed_panel)
        assert summary["share_country_years_simulated"] == 0.0
        assert summary["share_draws_simulated"] == 0.0

    def test_era_shares_sum_back_to_the_panel(self, mixed_panel) -> None:
        era = pv.simulated_share_by_era(
            mixed_panel, edges=(2000, 2010, 2020))
        summary = pv.panel_summary(mixed_panel)
        assert int(era["country_years"].sum()) == summary["country_years"]
        assert int(era["return_cells"].sum()) == summary["return_cells"]
        assert int(era["simulated"].sum()) == summary["return_cells_simulated"]


class TestInternationalLeg:
    def test_contamination_is_zero_when_nothing_is_simulated(
            self, observed_panel) -> None:
        frame = pv.international_leg_contamination(
            observed_panel, edges=(2000, 2010, 2020))
        assert (frame["mean_synthetic_share_of_intl_leg"] == 0.0).all()

    def test_contamination_is_positive_when_something_is(self, mixed_panel
                                                         ) -> None:
        frame = pv.international_leg_contamination(
            mixed_panel, edges=(2000, 2010, 2020))
        whole = frame[frame["era"] == "whole panel"]
        assert float(whole["mean_synthetic_share_of_intl_leg"].iloc[0]) > 0.0

    def test_it_measures_the_leg_of_observed_countries_only(self, mixed_panel
                                                            ) -> None:
        """With two observed and one simulated country, each observed
        investor's leg is one other observed country and the simulated one."""
        frame = pv.international_leg_contamination(
            mixed_panel, edges=(2000, 2020))
        whole = frame[frame["era"] == "whole panel"]
        assert float(whole["mean_synthetic_share_of_intl_leg"].iloc[0]) \
            == pytest.approx(0.5, abs=0.2)


class TestSourceAudit:
    def test_the_identity_holds_for_a_constructed_frame(self) -> None:
        cg = np.array([0.10, -0.20, 0.05])
        dp = np.array([0.03, 0.04, 0.02])
        frame = pd.DataFrame({"iso": ["X"] * 3, "year": [1, 2, 3],
                              "eq_capgain": cg, "eq_dp": dp,
                              "eq_tr": (1 + cg) * (1 + dp) - 1})
        out = pv.identity_check(frame).iloc[0]
        assert int(out["violations_above_tolerance"]) == 0
        assert float(out["max_error"]) < 1e-12

    def test_a_broken_component_is_detected(self) -> None:
        frame = pd.DataFrame({"iso": ["X"] * 2, "year": [1, 2],
                              "eq_capgain": [0.10, 0.10],
                              "eq_dp": [0.03, 0.03],
                              "eq_tr": [0.1330, 0.5000]})
        out = pv.identity_check(frame).iloc[0]
        assert int(out["violations_above_tolerance"]) == 1

    def test_anchor_check_flags_a_value_that_is_wrong(self) -> None:
        frame = pd.DataFrame({"iso": ["USA"], "year": [2008],
                              "eq_tr": [0.35]})
        out = pv.anchor_check(frame, anchors=[
            {"iso": "USA", "year": 2008, "series": "eq_tr",
             "expected": -0.37, "tolerance": 0.05, "what": "crisis"}])
        assert not bool(out["within_tolerance"].iloc[0])

    def test_anchor_check_passes_a_value_that_is_right(self) -> None:
        frame = pd.DataFrame({"iso": ["USA"], "year": [2008],
                              "eq_tr": [-0.3875]})
        out = pv.anchor_check(frame, anchors=[
            {"iso": "USA", "year": 2008, "series": "eq_tr",
             "expected": -0.37, "tolerance": 0.05, "what": "crisis"}])
        assert bool(out["within_tolerance"].iloc[0])


class TestTailTest:
    def frame(self, tail_scale: float) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for iso in ("AAA", "BBB", "CCC"):
            for year in range(1950, 2021):
                scale = 0.2 * (tail_scale if year >= 2016 else 1.0)
                rows.append({"iso": iso, "year": year,
                             "eq_tr": float(rng.normal(0.05, scale))})
        return pd.DataFrame.from_records(rows)

    def test_a_smoothed_tail_is_detected_in_every_country(self) -> None:
        out = pv.tail_variance_test(self.frame(0.05))
        assert len(out) == 3
        assert bool(out["tail_smoother"].all())
        verdict = pv.tail_verdict(out)
        assert verdict["smoother"] == 3
        assert verdict["p_value"] == pytest.approx(0.25)

    def test_an_unsmoothed_tail_does_not_look_significant(self) -> None:
        """The claim the test makes is about the sign test, not about any one
        country: with a homoscedastic tail the evidence must be weak."""
        out = pv.tail_variance_test(self.frame(1.0))
        assert pv.tail_verdict(out)["p_value"] >= 0.25

    def test_a_short_series_is_skipped_rather_than_guessed(self) -> None:
        frame = pd.DataFrame({"iso": ["AAA"] * 5,
                              "year": list(range(2016, 2021)),
                              "eq_tr": [0.1] * 5})
        assert pv.tail_variance_test(frame).empty

    def test_the_sign_test_matches_the_binomial(self) -> None:
        assert pv.sign_test_p_value(16, 16) == pytest.approx(2 / 2 ** 16)
        assert pv.sign_test_p_value(0, 16) == pytest.approx(2 / 2 ** 16)
        assert pv.sign_test_p_value(8, 16) == pytest.approx(1.0)

    def test_the_sign_test_reports_nan_for_no_trials(self) -> None:
        assert np.isnan(pv.sign_test_p_value(0, 0))
