"""Tests for the observed-series recovery (`src.observed`).

The point of this module is that a country-year stops being simulated, so the
tests are mostly about arithmetic being right and provenance being honest: a
recovered value has to be the value the source implies, and a cell may only be
called observed when *everything* that went into it was observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import observed as obs


# ---------------------------------------------------------------------------
# bond_return_from_yield
# ---------------------------------------------------------------------------
def test_flat_yield_curve_returns_the_carry():
    """With no yield change the return is last year's yield, whatever D."""
    yields = np.full(5, 0.04)
    for duration in (1.0, 7.0, 20.0):
        got = obs.bond_return_from_yield(yields, duration)
        assert np.isnan(got[0])
        assert got[1:] == pytest.approx(0.04)


def test_first_year_is_nan_not_filled():
    got = obs.bond_return_from_yield(np.array([0.05, 0.05]), 7.0)
    assert np.isnan(got[0]), "no previous yield exists to earn carry at"


def test_falling_yield_makes_a_capital_gain():
    """A 100bp fall on a seven-year bond is roughly seven points of gain."""
    got = obs.bond_return_from_yield(np.array([0.05, 0.04]), 7.0)
    assert got[1] == pytest.approx(0.05 + 7.0 * 0.01)


def test_rising_yield_can_make_the_return_negative():
    got = obs.bond_return_from_yield(np.array([0.02, 0.05]), 10.0)
    assert got[1] < 0.0


def test_duration_scales_the_capital_leg_only():
    short = obs.bond_return_from_yield(np.array([0.05, 0.04]), 2.0)[1]
    long = obs.bond_return_from_yield(np.array([0.05, 0.04]), 8.0)[1]
    assert (long - short) == pytest.approx((8.0 - 2.0) * 0.01)


def test_nan_yield_propagates_rather_than_being_filled():
    got = obs.bond_return_from_yield(np.array([0.05, np.nan, 0.04]), 7.0)
    assert np.isnan(got[1]) and np.isnan(got[2])


# ---------------------------------------------------------------------------
# deflation
# ---------------------------------------------------------------------------
def test_real_return_is_the_gross_ratio():
    got = obs._real(np.array([0.10]), np.array([0.05]))
    assert got[0] == pytest.approx(1.10 / 1.05 - 1.0)


def test_missing_inflation_gives_nan_not_the_nominal_return():
    got = obs._real(np.array([0.10]), np.array([np.nan]))
    assert np.isnan(got[0]), "an undeflated nominal return must not survive"


# ---------------------------------------------------------------------------
# bond_from_clio -- the units contract
# ---------------------------------------------------------------------------
@pytest.fixture()
def clio_yields():
    """A year-indexed frame in the shape `load_clio_wide` returns.

    That loader has already rescaled percent to decimals, so 4% arrives as
    0.04 and must not be divided again.
    """
    return pd.DataFrame({"Someland": [0.04, 0.04, 0.04]},
                        index=pd.Index([1990, 1991, 1992], name="year"))


def test_clio_yields_are_read_as_decimals(clio_yields):
    years = np.array([1990, 1991, 1992])
    got = obs.bond_from_clio(clio_yields, "Someland", years,
                             np.zeros(3), duration=7.0)
    assert got[1] == pytest.approx(0.04), (
        "a 4% yield must give a 4% return, not 0.04%; `load_clio_wide` "
        "already applied the percent-to-decimal scaling"
    )


def test_clio_recovery_is_not_dominated_by_the_deflator(clio_yields):
    """The regression this guards: double-scaling left return ~= -inflation."""
    years = np.array([1990, 1991, 1992])
    got = obs.bond_from_clio(clio_yields, "Someland", years,
                             np.full(3, 0.03), duration=7.0)
    assert got[1] > 0.0
    assert got[1] == pytest.approx(1.04 / 1.03 - 1.0)


def test_absent_column_gives_all_nan(clio_yields):
    got = obs.bond_from_clio(clio_yields, "Nowhere", np.array([1990, 1991]),
                             np.zeros(2), duration=7.0)
    assert np.isnan(got).all()


def test_years_outside_the_source_are_nan(clio_yields):
    years = np.array([1988, 1990, 1991, 2015])
    got = obs.bond_from_clio(clio_yields, "Someland", years,
                             np.zeros(4), duration=7.0)
    assert np.isnan(got[0]) and np.isnan(got[3])


# ---------------------------------------------------------------------------
# rates_from_jst
# ---------------------------------------------------------------------------
@pytest.fixture()
def jst_rates():
    """Three years of a country with rates and prices but no return series."""
    return pd.DataFrame({
        "iso": ["XXX"] * 3,
        "year": [1990, 1991, 1992],
        "cpi": [100.0, 103.0, 106.09],   # exactly 3% a year
        "ltrate": [5.0, 5.0, 5.0],       # percent, as the workbook stores it
        "stir": [4.0, 4.0, 4.0],
    })


def test_jst_rates_are_read_as_percent(jst_rates):
    years = np.array([1990, 1991, 1992])
    got = obs.rates_from_jst(jst_rates, "XXX", years, duration=7.0)
    assert got["bond"][1] == pytest.approx(1.05 / 1.03 - 1.0)
    assert got["bill"][1] == pytest.approx(1.04 / 1.03 - 1.0)


def test_bill_return_is_the_lagged_short_rate(jst_rates):
    """A bill bought at t-1 earns the rate that was quoted then."""
    frame = jst_rates.copy()
    frame["stir"] = [4.0, 9.0, 9.0]
    got = obs.rates_from_jst(frame, "XXX", np.array([1990, 1991, 1992]),
                             duration=7.0)
    assert got["bill"][1] == pytest.approx(1.04 / 1.03 - 1.0)


def test_unknown_country_gives_nan_arrays(jst_rates):
    got = obs.rates_from_jst(jst_rates, "ZZZ", np.array([1990, 1991]), 7.0)
    assert np.isnan(got["bond"]).all() and np.isnan(got["bill"]).all()


def test_jst_recovery_uses_the_countrys_own_price_index(jst_rates):
    """Nothing outside the country's own block may enter its deflator."""
    other = jst_rates.copy()
    other["iso"] = "YYY"
    other["cpi"] = [100.0, 200.0, 400.0]
    combined = pd.concat([jst_rates, other], ignore_index=True)
    alone = obs.rates_from_jst(jst_rates, "XXX", np.array([1990, 1991]), 7.0)
    with_noise = obs.rates_from_jst(combined, "XXX", np.array([1990, 1991]), 7.0)
    assert with_noise["bond"][1] == pytest.approx(alone["bond"][1])


# ---------------------------------------------------------------------------
# housing
# ---------------------------------------------------------------------------
def test_housing_returns_are_deflated_and_shaped(jst_rates):
    frame = jst_rates.copy()
    frame["housing_tr"] = [0.08, 0.08, 0.08]
    got = obs.housing_returns(frame, ["XXX", "ZZZ"], np.array([1990, 1991]))
    assert got.shape == (2, 2)
    assert got[1, 0] == pytest.approx(1.08 / 1.03 - 1.0)
    assert np.isnan(got[:, 1]).all(), "a country not in the file stays NaN"


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------
def test_summarise_names_the_source_per_country():
    frame = obs.summarise({
        "CAN": {"bond": np.array([0.01, np.nan]), "bill": np.array([0.01, 0.01])},
        "NZL": {"bond": np.array([0.02, 0.02])},
    })
    assert set(frame["iso"]) == {"CAN", "NZL"}
    can = frame.set_index("iso").loc["CAN"]
    assert can["bond_years"] == 1 and can["bill_years"] == 2
    assert "Jordà" in can["source"]
    assert "Clio" in frame.set_index("iso").loc["NZL"]["source"]


def test_summarise_of_nothing_is_empty():
    assert obs.summarise({}).empty


# ---------------------------------------------------------------------------
# derive_tiers -- the label may not drift from the masks
# ---------------------------------------------------------------------------
def _masks(dom_eq, bond, bill):
    return {"dom_eq": np.array(dom_eq, dtype=bool),
            "bond": np.array(bond, dtype=bool),
            "bill": np.array(bill, dtype=bool)}


def test_fully_observed_country_is_tier_a():
    available = np.ones((3, 1), dtype=bool)
    observed = _masks([[1]] * 3, [[1]] * 3, [[1]] * 3)
    assert dl.derive_tiers(observed, available, ["C"]) == ["A"]


def test_wholly_generated_country_is_tier_c():
    available = np.ones((3, 1), dtype=bool)
    observed = _masks([[0]] * 3, [[0]] * 3, [[0]] * 3)
    assert dl.derive_tiers(observed, available, ["A"]) == ["C"]


def test_rates_observed_equity_simulated_is_tier_b():
    available = np.ones((3, 1), dtype=bool)
    observed = _masks([[0]] * 3, [[1]] * 3, [[1]] * 3)
    assert dl.derive_tiers(observed, available, ["C"]) == ["B"]


def test_one_observed_cell_is_enough_to_leave_tier_c():
    available = np.ones((3, 1), dtype=bool)
    observed = _masks([[0]] * 3, [[1], [0], [0]], [[0]] * 3)
    assert dl.derive_tiers(observed, available, ["C"]) == ["B"]


def test_unavailable_cells_do_not_count_against_a_country():
    """Tier A means every *available* cell is observed, not every cell."""
    available = np.array([[True], [False], [True]])
    observed = _masks([[1], [0], [1]], [[1], [0], [1]], [[1], [0], [1]])
    assert dl.derive_tiers(observed, available, ["C"]) == ["A"]


def test_country_with_no_data_keeps_its_fallback_label():
    available = np.zeros((3, 1), dtype=bool)
    observed = _masks([[0]] * 3, [[0]] * 3, [[0]] * 3)
    assert dl.derive_tiers(observed, available, ["B"]) == ["B"]


def test_empty_masks_leave_every_label_alone():
    available = np.ones((3, 2), dtype=bool)
    assert dl.derive_tiers({}, available, ["A", "B"]) == ["A", "B"]
