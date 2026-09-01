"""Shared fixtures.  Also puts the repository root on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import Panel  # noqa: E402


@pytest.fixture()
def toy_panel() -> Panel:
    """A small, fully controlled panel with a deliberate war-time gap.

    Three countries, 20 years.  Country ``GAP`` is missing years 6-8, which
    is what the block-admissibility tests exercise.
    """
    years = np.arange(2000, 2020)
    n_t, n_c = years.size, 3
    rng = np.random.default_rng(0)
    dom_eq = rng.normal(0.07, 0.18, (n_t, n_c))
    intl_eq = rng.normal(0.06, 0.17, (n_t, n_c))
    bond = rng.normal(0.02, 0.08, (n_t, n_c))
    bill = rng.normal(0.01, 0.04, (n_t, n_c))
    inflation = rng.normal(0.03, 0.04, (n_t, n_c))
    available = np.ones((n_t, n_c), dtype=bool)
    available[6:9, 1] = False           # the gap in country index 1
    for arr in (dom_eq, intl_eq, bond, bill, inflation):
        arr[~available] = np.nan
    return Panel(
        years=years,
        countries=("AAA", "GAP", "CCC"),
        tier=("A", "A", "B"),
        dom_eq=dom_eq, intl_eq=intl_eq, bond=bond, bill=bill,
        inflation=inflation,
        real_exchange_rate=np.ones((n_t, n_c)),
        available=available,
        name="toy",
        provenance=("test", "test", "test"),
    )


@pytest.fixture()
def toy_config() -> dict:
    """Minimal configuration for the lifecycle and utility layers."""
    return {
        "run": {"seed": 1},
        "bootstrap": {
            "panel": "toy", "n_paths": 200, "chunk_size": 100,
            "horizon_years": 12, "mean_block_years": 4.0,
            "block_length_distribution": "geometric",
            "min_block_years": 1, "max_block_years": 10,
            "country_draw": "per_lifetime", "country_weighting": "history",
            "seed": 3,
        },
        "lifecycle": {
            "age_start": 25, "age_retire": 31, "age_death": 37,
            "savings_rate": 0.10, "rebalancing": "annual",
            "income": {"initial_real_income": 1.0, "b1": 0.045, "b2": -0.0009,
                       "permanent_shock_sd": 0.10, "transitory_shock_sd": 0.25,
                       "shocks_enabled": False},
            "social_security": {"enabled": True, "formula": "progressive",
                                "replacement_rate": 0.45, "pia_bend1": 0.21,
                                "pia_bend2": 1.28, "pia_rate1": 0.90,
                                "pia_rate2": 0.32, "pia_rate3": 0.15},
            "retirement": {"rule": "fixed_real_rule", "rule_rate": 0.04,
                           "allow_ruin": True},
        },
        "strategies": {
            "all_equity": {"label": "All equity", "type": "constant",
                           "weights": {"dom_eq": 0.5, "intl_eq": 0.5}},
            "glide": {"label": "Glide", "type": "glide",
                      "glide_ages": [25, 31, 37],
                      "glide_equity": [0.9, 0.5, 0.3],
                      "equity_split": {"dom_eq": 0.6, "intl_eq": 0.4},
                      "fixed_income_split": {"bond": 0.7, "bill": 0.3}},
        },
        "utility": {
            "discount_factor": 0.96, "risk_aversions": [2.0, 5.0],
            "baseline_risk_aversion": 5.0, "epstein_zin_ies": [0.5, 1.5],
            "baseline_ies": 1.5, "bequest_weight": 2.0,
            "bequest_enabled": True, "bequest_shift": 1.0,
            "consumption_floor": 1e-4, "consumption_window": "retirement",
        },
        "report": {"percentiles": [5, 50, 95],
                   "consumption_target_replacement": 0.70},
    }


@pytest.fixture()
def persistent_panel() -> Panel:
    """A single-country panel whose returns follow a strongly persistent AR(1).

    Block-length tests need a source process with real persistence -- against
    i.i.d. data every block length looks alike and the test asserts nothing.
    """
    n_t, n_c, rho = 400, 2, 0.7
    rng = np.random.default_rng(7)
    shocks = rng.normal(0.0, 0.10, (n_t, n_c))
    series = np.zeros((n_t, n_c))
    for t in range(1, n_t):
        series[t] = rho * series[t - 1] + shocks[t]
    series = series + 0.06
    return Panel(
        years=np.arange(1600, 1600 + n_t),
        countries=("PER", "SIS"),
        tier=("A", "A"),
        dom_eq=series,
        intl_eq=series[:, ::-1].copy(),
        bond=series * 0.3,
        bill=series * 0.1,
        inflation=np.full((n_t, n_c), 0.02),
        real_exchange_rate=np.ones((n_t, n_c)),
        available=np.ones((n_t, n_c), dtype=bool),
        name="persistent",
        provenance=("test", "test"),
    )


@pytest.fixture(scope="session")
def real_panel_or_skip():
    """The real 16-country panel and its source workbook, or a skip.

    A handful of properties -- the no-look-ahead guarantee above all -- are
    claims about the actual data, not about the code, and a toy fixture cannot
    stand in for them. Skips rather than fails when the raw files are absent,
    so the suite still runs on a fresh clone.
    """
    from src import data_loader as dl

    try:
        cfg = dl.load_config("config.yaml")
        panel = dl.build_panel(cfg)
        jst = dl.load_jst(cfg)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        pytest.skip(f"raw data unavailable: {exc}")
    return panel, jst


@pytest.fixture(scope="session")
def real_config_or_skip():
    """The project's own configuration, or a skip.

    Some properties -- that the GDP weights are genuinely lagged, that two
    panels are paired on identical history -- are claims about the real data
    and the real config, and the toy fixtures cannot stand in for them.
    """
    from src import data_loader as dl

    try:
        cfg = dl.load_config("config.yaml")
        dl.load_jst(cfg)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        pytest.skip(f"raw data unavailable: {exc}")
    return cfg
