"""Tests for the international-sleeve weighting comparison.

The load-bearing tests are the paired ones. This section's whole claim is
that the equal- and GDP-weighted panels differ in the sleeve and in nothing
else, so a test that lets availability, the country list or a domestic-only
strategy drift is a test that would let the comparison lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import sleeve as sv


# ---------------------------------------------------------------------------
# The weighted leave-one-out average itself
# ---------------------------------------------------------------------------
class TestWeightedInternationalLeg:
    def test_equal_weights_reproduce_the_equal_weighting_exactly(self) -> None:
        # The headline panel is built by the "equal" branch, so a GDP weighting
        # with flat weights must land on it bit for bit or every existing
        # result silently depends on which branch ran.
        rng = np.random.default_rng(0)
        gross = 1.0 + rng.normal(0.07, 0.2, size=(40, 5))
        fx = np.full((40, 5), 1.0)
        infl = np.zeros((40, 5))
        equal = dl.build_international_leg(gross, fx, infl, weighting="equal")
        flat = dl.build_international_leg(
            gross, fx, infl, weighting="gdp",
            wide_weight=np.full((40, 5), 3.7))
        assert np.allclose(equal, flat, equal_nan=True)

    def test_a_dominant_market_pulls_the_sleeve_towards_itself(self) -> None:
        gross = np.array([[1.0, 2.0, 1.0]])          # market 1 doubles
        fx = np.ones((1, 3))
        infl = np.zeros((1, 3))
        weight = np.array([[1.0, 99.0, 1.0]])
        out = dl.build_international_leg(gross, fx, infl, weighting="gdp",
                                         wide_weight=weight)
        # For market 0 the sleeve is markets 1 and 2, and market 1 carries
        # 99/100 of it, so the sleeve return sits just under +100%.
        assert out[0, 0] == pytest.approx(0.99, abs=1e-9)
        # For market 1 the sleeve is markets 0 and 2, both flat.
        assert out[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_the_home_market_is_excluded_from_its_own_sleeve(self) -> None:
        gross = np.array([[3.0, 1.0, 1.0]])
        weight = np.array([[1000.0, 1.0, 1.0]])
        out = dl.build_international_leg(gross, np.ones((1, 3)),
                                         np.zeros((1, 3)), weighting="gdp",
                                         wide_weight=weight)
        # If the home column leaked in, market 0's own +200% would dominate.
        assert out[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_a_market_without_a_weight_is_dropped_not_zero_weighted(self
                                                                   ) -> None:
        gross = np.array([[1.0, 2.0, 4.0]])
        weight = np.array([[1.0, np.nan, 1.0]])
        out = dl.build_international_leg(gross, np.ones((1, 3)),
                                         np.zeros((1, 3)), weighting="gdp",
                                         wide_weight=weight)
        # Market 0's sleeve is markets 1 and 2; 1 has no weight so the sleeve
        # is market 2 alone, at +300%. A zero weight would give the same, but
        # market 1's own sleeve must then be 0 and 2 equally weighted.
        assert out[0, 0] == pytest.approx(3.0, abs=1e-9)
        assert out[0, 1] == pytest.approx(1.5, abs=1e-9)

    def test_gdp_weighting_without_weights_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="wide_weight"):
            dl.build_international_leg(np.ones((2, 2)), np.ones((2, 2)),
                                       np.zeros((2, 2)), weighting="gdp")

    def test_a_mis_shaped_weight_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            dl.build_international_leg(np.ones((2, 3)), np.ones((2, 3)),
                                       np.zeros((2, 3)), weighting="gdp",
                                       wide_weight=np.ones((2, 2)))

    def test_a_negative_weight_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            dl.build_international_leg(np.ones((1, 2)), np.ones((1, 2)),
                                       np.zeros((1, 2)), weighting="gdp",
                                       wide_weight=np.array([[1.0, -1.0]]))

    def test_an_unknown_weighting_is_still_rejected(self) -> None:
        with pytest.raises(NotImplementedError):
            dl.build_international_leg(np.ones((2, 2)), np.ones((2, 2)),
                                       np.zeros((2, 2)), weighting="cap")


# ---------------------------------------------------------------------------
# Concentration and moments
# ---------------------------------------------------------------------------
class TestConcentration:
    def test_equal_sizes_give_the_market_count_back(self) -> None:
        frame = pd.DataFrame({
            "hhi": [0.25, 0.5], "eff": [4.0, 2.0]})   # documentation only
        share = np.full(4, 0.25)
        assert 1.0 / float((share ** 2).sum()) == pytest.approx(4.0)
        assert len(frame) == 2

    def test_effective_markets_never_exceeds_the_market_count(self,
                                                              real_config_or_skip
                                                              ) -> None:
        panel = dl.build_tier_a(real_config_or_skip, weighting="gdp")
        frame = sv.concentration(real_config_or_skip, panel.countries, panel.years)
        assert len(frame)
        assert (frame["effective_markets"] <= frame["markets"] + 1e-9).all()
        assert (frame["largest_share"] > 0.0).all()
        assert (frame["largest_share"] <= 1.0).all()
        assert (frame["top3_share"] >= frame["largest_share"] - 1e-9).all()

    def test_the_largest_market_matches_the_largest_share(self, real_config_or_skip
                                                          ) -> None:
        panel = dl.build_tier_a(real_config_or_skip, weighting="gdp")
        frame = sv.concentration(real_config_or_skip, panel.countries,
                                 panel.years, ["gdp"])
        jst = dl.add_real_returns(dl.load_jst(real_config_or_skip))
        window = jst[jst["year"].between(int(panel.years[0]) - 1,
                                         int(panel.years[-1]))]
        size = dl.economy_size(window, panel.years, list(panel.countries))
        for _, row in frame.head(20).iterrows():
            i = int(np.flatnonzero(panel.years == row["year"])[0])
            expected = panel.countries[int(np.nanargmax(size[i]))]
            assert row["largest_market"] == expected


# ---------------------------------------------------------------------------
# The paired comparison
# ---------------------------------------------------------------------------
class TestPanels:
    def test_the_two_panels_are_paired_on_identical_history(self, real_config_or_skip
                                                            ) -> None:
        panels = sv.build_panels(real_config_or_skip)
        equal, gdp = panels["equal"], panels["gdp"]
        assert equal.countries == gdp.countries
        assert np.array_equal(equal.years, gdp.years)
        assert np.array_equal(equal.available, gdp.available)
        # Only the sleeve moves.
        for series in ("dom_eq", "bond", "bill", "inflation"):
            assert np.array_equal(getattr(equal, series),
                                  getattr(gdp, series), equal_nan=True)
        assert not np.array_equal(equal.intl_eq, gdp.intl_eq, equal_nan=True)

    def test_the_panels_are_named_apart(self, real_config_or_skip) -> None:
        # Two panels with the same name would collide in every cache and in
        # every table that keys on it.
        panels = sv.build_panels(real_config_or_skip)
        assert panels["equal"].name != panels["gdp"].name

    def test_the_nan_pattern_of_the_sleeve_is_unchanged(self, real_config_or_skip
                                                        ) -> None:
        panels = sv.build_panels(real_config_or_skip)
        assert np.array_equal(np.isnan(panels["equal"].intl_eq),
                              np.isnan(panels["gdp"].intl_eq))


class TestVerdict:
    @staticmethod
    def _frame(equal_gap: float, gdp_gap: float) -> pd.DataFrame:
        rows = []
        for scheme, gap in (("equal", equal_gap), ("gdp", gdp_gap)):
            rows += [
                {"weighting": scheme, "strategy": "international_equity",
                 "label": "100% International", "cec_crra_gamma5": 1.0 + gap,
                 "prob_ruin": 0.1},
                {"weighting": scheme, "strategy": "balanced_all_equity",
                 "label": "50/50", "cec_crra_gamma5": 1.0, "prob_ruin": 0.12},
            ]
        return pd.DataFrame.from_records(rows)

    def test_a_surviving_ordering_is_reported_as_surviving(self) -> None:
        out = sv.verdict(self._frame(0.06, 0.04))
        assert out["survives"] is True
        assert out["ordering_changes"] is False
        assert out["equal"]["gap_pct"] == pytest.approx(6.0)
        assert out["gdp"]["gap_pct"] == pytest.approx(4.0)

    def test_a_flipped_ordering_is_reported_as_flipped(self) -> None:
        out = sv.verdict(self._frame(0.06, -0.02))
        assert out["survives"] is False
        assert out["ordering_changes"] is True
        assert out["winner_changes"] is True

    def test_ranking_shift_reports_the_change_against_the_reference(self
                                                                     ) -> None:
        shift = sv.ranking_shift(self._frame(0.06, 0.04))
        assert set(shift["strategy"]) == {"international_equity",
                                          "balanced_all_equity"}
        row = shift[shift["strategy"] == "international_equity"].iloc[0]
        assert float(row["gdp_change_pct"]) == pytest.approx(
            (1.04 / 1.06 - 1.0) * 100.0)
        # The reference scheme gets no change column of its own.
        assert "equal_change_pct" not in shift.columns

    def test_a_frame_without_a_cec_column_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            sv.verdict(pd.DataFrame({"weighting": ["equal"],
                                     "strategy": ["x"], "label": ["x"]}))


class TestEconomySize:
    def test_the_weights_are_lagged_by_exactly_one_year(self, real_config_or_skip
                                                        ) -> None:
        jst = dl.add_real_returns(dl.load_jst(real_config_or_skip))
        isos = list(dl.TIER_A_ISO)
        years = np.arange(1950, 1960)
        lagged = dl.economy_size(jst, years, isos)
        direct = dl._pivot(jst, "rgdpmad", years - 1, isos) \
            * dl._pivot(jst, "pop", years - 1, isos)
        assert np.allclose(lagged, direct, equal_nan=True)
        # And it must genuinely differ from the contemporaneous version, or
        # the "no look-ahead" claim in docs/18 is vacuous.
        same_year = dl._pivot(jst, "rgdpmad", years, isos) \
            * dl._pivot(jst, "pop", years, isos)
        assert not np.allclose(lagged, same_year, equal_nan=True)

    def test_non_positive_sizes_become_missing(self, real_config_or_skip) -> None:
        frame = pd.DataFrame({
            "year": [1949, 1950], "iso": ["USA", "USA"],
            "rgdpmad": [0.0, 100.0], "pop": [10.0, 10.0]})
        out = dl.economy_size(frame, np.array([1950, 1951]), ["USA"])
        assert np.isnan(out[0, 0])
        assert out[1, 0] == pytest.approx(1000.0)


class TestTheOtherSchemes:
    """Population, GDP per capita and inverse volatility.

    The set is chosen so that two schemes concentrate the sleeve heavily and
    two barely concentrate it while tilting it elsewhere. If that separation
    collapses, ``docs/18`` section 5 is answering a question the data cannot
    distinguish, so it is pinned here.
    """

    def test_every_registered_scheme_builds_a_paired_panel(self,
                                                           real_config_or_skip
                                                           ) -> None:
        panels = sv.build_panels(real_config_or_skip, dl.SLEEVE_SCHEMES)
        assert set(panels) == set(dl.SLEEVE_SCHEMES)
        ref = panels["equal"]
        for name, panel in panels.items():
            assert panel.countries == ref.countries
            assert np.array_equal(panel.available, ref.available)
            assert np.array_equal(np.isnan(panel.intl_eq),
                                  np.isnan(ref.intl_eq))
            if name != "equal":
                assert not np.array_equal(panel.intl_eq, ref.intl_eq,
                                          equal_nan=True)

    def test_the_schemes_span_both_concentrated_and_near_equal(
            self, real_config_or_skip) -> None:
        panel = dl.build_tier_a(real_config_or_skip, weighting="equal")
        frame = sv.concentration(real_config_or_skip, panel.countries,
                                 panel.years, dl.SLEEVE_SCHEMES)
        mean = frame.groupby("weighting")["effective_markets"].mean()
        n = len(panel.countries)
        assert mean["gdp"] < 0.5 * n and mean["pop"] < 0.6 * n
        assert mean["gdp_pc"] > 0.8 * n and mean["inverse_vol"] > 0.8 * n
        assert mean["equal"] == pytest.approx(mean.max())

    def test_equal_weighting_is_the_concentration_ceiling(self,
                                                          real_config_or_skip
                                                          ) -> None:
        panel = dl.build_tier_a(real_config_or_skip, weighting="equal")
        frame = sv.concentration(real_config_or_skip, panel.countries,
                                 panel.years, dl.SLEEVE_SCHEMES)
        equal = frame[frame["weighting"] == "equal"].set_index("year")
        for scheme in dl.SLEEVE_SCHEMES:
            block = frame[frame["weighting"] == scheme].set_index("year")
            ceiling = equal.loc[block.index, "effective_markets"]
            assert (block["effective_markets"] <= ceiling + 1e-9).all(), scheme

    def test_an_unknown_scheme_is_rejected_by_the_dispatcher(self) -> None:
        with pytest.raises(NotImplementedError):
            dl.sleeve_weights("market_cap", pd.DataFrame(),
                              np.array([1990]), ["USA"])

    def test_inverse_vol_needs_the_returns_it_is_estimated_from(self) -> None:
        with pytest.raises(ValueError, match="gross return"):
            dl.sleeve_weights("inverse_vol", pd.DataFrame(),
                              np.array([1990]), ["USA"])


class TestTrailingInverseVolatility:
    def test_a_steadier_market_gets_the_larger_weight(self) -> None:
        rng = np.random.default_rng(4)
        calm = 1.0 + rng.normal(0.05, 0.05, size=40)
        wild = 1.0 + rng.normal(0.05, 0.30, size=40)
        weights = dl.trailing_inverse_volatility(np.column_stack([calm, wild]))
        assert (weights[-1, 0] > weights[-1, 1])

    def test_the_window_is_strictly_prior(self) -> None:
        # A single enormous return must not affect its own year's weight, only
        # later ones, or the scheme is not implementable.
        base = np.full((30, 2), 1.05)
        base[:, 1] += np.linspace(0, 0.01, 30)      # break the zero-variance tie
        spiked = base.copy()
        spiked[20, 0] = 5.0
        before = dl.trailing_inverse_volatility(base)
        after = dl.trailing_inverse_volatility(spiked)
        assert np.allclose(before[:21], after[:21])
        assert not np.allclose(before[21:], after[21:])

    def test_the_burn_in_falls_back_to_neutral_rather_than_dropping(self
                                                                    ) -> None:
        gross = 1.0 + np.random.default_rng(1).normal(0.05, 0.2, size=(20, 4))
        weights = dl.trailing_inverse_volatility(gross, window=10, min_obs=5)
        # Every cell is finite, so no market is silently dropped from the
        # sleeve during the burn-in.
        assert np.isfinite(weights).all()
        # And the first year, with no prior data at all, is flat.
        assert len(set(np.round(weights[0], 12))) == 1

    def test_weights_are_strictly_positive(self) -> None:
        gross = 1.0 + np.random.default_rng(2).normal(0.05, 0.2, size=(50, 6))
        weights = dl.trailing_inverse_volatility(gross)
        assert (weights > 0.0).all()
