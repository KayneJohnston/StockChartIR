"""Tests for the delete-one-country and sub-period robustness studies.

The load-bearing test is the first one. A leave-one-out study that slices an
already-built panel would leave the dropped market inside every other
country's international sleeve, and would then report -- reassuringly and
wrongly -- that no country matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import panel_robustness as pr


def _summary(intl: float, fifty: float, **extra) -> pd.DataFrame:
    """A minimal stand-in for the headline comparison table."""
    rows = [
        {"strategy": "international_equity", "label": "100% International",
         "cec_crra_gamma5": intl, "prob_ruin": 0.09, **extra},
        {"strategy": "balanced_all_equity", "label": "50/50",
         "cec_crra_gamma5": fifty, "prob_ruin": 0.12, **extra},
    ]
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# The deletion itself
# ---------------------------------------------------------------------------
class TestDroppingACountry:
    def test_a_dropped_market_leaves_everyone_elses_sleeve(
            self, real_config_or_skip) -> None:
        # If this fails, the whole study is vacuous: it would be measuring
        # only the loss of one domestic column, not one market's history.
        full = dl.build_tier_a(real_config_or_skip)
        kept = [c for c in full.countries if c != "USA"]
        without = dl.build_tier_a(real_config_or_skip, countries=kept)
        i, j = full.country_index("GBR"), without.country_index("GBR")
        assert np.array_equal(full.dom_eq[:, i], without.dom_eq[:, j],
                              equal_nan=True)
        assert not np.array_equal(full.intl_eq[:, i], without.intl_eq[:, j],
                                  equal_nan=True)

    def test_the_dropped_country_is_gone_from_the_panel(self,
                                                        real_config_or_skip
                                                        ) -> None:
        full = dl.build_tier_a(real_config_or_skip)
        kept = [c for c in full.countries if c != "JPN"]
        without = dl.build_tier_a(real_config_or_skip, countries=kept)
        assert "JPN" not in without.countries
        assert len(without.countries) == len(full.countries) - 1

    def test_the_other_assets_are_untouched(self, real_config_or_skip) -> None:
        full = dl.build_tier_a(real_config_or_skip)
        kept = [c for c in full.countries if c != "ITA"]
        without = dl.build_tier_a(real_config_or_skip, countries=kept)
        for series in ("bond", "bill", "inflation"):
            for iso in kept:
                a = getattr(full, series)[:, full.country_index(iso)]
                b = getattr(without, series)[:, without.country_index(iso)]
                assert np.array_equal(a, b, equal_nan=True), (series, iso)

    def test_an_unknown_country_is_rejected(self, real_config_or_skip) -> None:
        with pytest.raises(ValueError, match="not Tier-A"):
            dl.build_tier_a(real_config_or_skip, countries=["USA", "ATLANTIS"])

    def test_one_country_cannot_have_an_international_leg(self,
                                                          real_config_or_skip
                                                          ) -> None:
        with pytest.raises(ValueError, match="at least"):
            dl.build_tier_a(real_config_or_skip, countries=["USA"])


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------
class TestGapAndInfluence:
    def test_the_gap_is_the_percentage_lead(self) -> None:
        assert pr.gap(_summary(1.10, 1.00)) == pytest.approx(10.0)
        assert pr.gap(_summary(0.90, 1.00)) == pytest.approx(-10.0)

    def test_a_missing_strategy_gives_nan_not_a_wrong_number(self) -> None:
        frame = _summary(1.1, 1.0)
        assert np.isnan(pr.gap(frame[frame["strategy"] != "50_50"].iloc[:1]))

    def test_shift_is_measured_against_the_full_panel(self) -> None:
        full = _summary(1.10, 1.00)                 # lead of 10%
        loco = pd.concat([
            pr._tag(_summary(1.05, 1.00), dropped="AAA"),   # lead of 5%
            pr._tag(_summary(1.20, 1.00), dropped="BBB"),   # lead of 20%
        ], ignore_index=True)
        out = pr.influence(loco, full)
        by = {r["dropped"]: r for _, r in out.iterrows()}
        assert float(by["AAA"]["shift_pct"]) == pytest.approx(-5.0)
        assert float(by["BBB"]["shift_pct"]) == pytest.approx(10.0)
        # Most negative first, so the load-bearing country heads the table.
        assert list(out["dropped"]) == ["AAA", "BBB"]

    def test_a_deletion_that_flips_the_ordering_is_flagged(self) -> None:
        full = _summary(1.10, 1.00)
        loco = pr._tag(_summary(0.98, 1.00), dropped="AAA")
        out = pr.influence(loco, full)
        assert not bool(out["ordering_holds"].iloc[0])

    def test_the_baseline_is_recoverable_from_any_row(self) -> None:
        # The paper reconstructs it this way, because attrs do not survive
        # the CSV round trip.
        full = _summary(1.10, 1.00)
        loco = pd.concat([pr._tag(_summary(1.05, 1.00), dropped="AAA"),
                          pr._tag(_summary(1.20, 1.00), dropped="BBB")],
                         ignore_index=True)
        out = pr.influence(loco, full)
        for _, row in out.iterrows():
            assert float(row["gap_pct"] - row["shift_pct"]) \
                == pytest.approx(pr.gap(full))


class TestJackknife:
    def test_the_standard_error_matches_the_textbook_formula(self) -> None:
        values = np.array([4.0, 5.0, 6.0, 9.0])
        frame = pd.DataFrame({"dropped": list("ABCD"), "gap_pct": values})
        out = pr.jackknife(frame, baseline_gap=6.0)
        n = values.size
        expected = np.sqrt((n - 1) / n * ((values - values.mean()) ** 2).sum())
        assert out["standard_error"] == pytest.approx(expected)
        assert out["n"] == n

    def test_identical_deletions_give_no_uncertainty(self) -> None:
        frame = pd.DataFrame({"dropped": list("ABC"), "gap_pct": [5.0] * 3})
        out = pr.jackknife(frame, baseline_gap=5.0)
        assert out["standard_error"] == pytest.approx(0.0)
        assert np.isinf(out["t_stat"])

    def test_a_marginal_interval_is_reported_as_marginal(self) -> None:
        # t just over the threshold: the binary "excludes zero" would hide it.
        frame = pd.DataFrame({"dropped": list("ABCD"),
                              "gap_pct": [3.0, 5.0, 7.0, 9.0]})
        out = pr.jackknife(frame, baseline_gap=6.0)
        assert out["t_stat"] == pytest.approx(6.0 / out["standard_error"])
        if out["excludes_zero"]:
            assert out["marginal"] == (1.96 <= out["t_stat"] < 2.5)

    def test_a_comfortable_interval_is_not_marginal(self) -> None:
        frame = pd.DataFrame({"dropped": list("ABCD"),
                              "gap_pct": [5.9, 6.0, 6.1, 6.0]})
        out = pr.jackknife(frame, baseline_gap=6.0)
        assert out["excludes_zero"]
        assert not out["marginal"]

    def test_too_few_deletions_report_nothing_rather_than_a_number(self
                                                                   ) -> None:
        frame = pd.DataFrame({"dropped": ["A"], "gap_pct": [5.0]})
        assert np.isnan(pr.jackknife(frame)["standard_error"])


# ---------------------------------------------------------------------------
# Sub-periods
# ---------------------------------------------------------------------------
class TestRestrictYears:
    def test_only_availability_changes(self, real_config_or_skip) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        cut = pr.restrict_years(panel, 1950, 1990)
        for series in ("dom_eq", "intl_eq", "bond", "bill", "inflation"):
            assert np.array_equal(getattr(panel, series),
                                  getattr(cut, series), equal_nan=True)
        assert cut.available.sum() < panel.available.sum()

    def test_nothing_outside_the_window_is_available(self, real_config_or_skip
                                                     ) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        cut = pr.restrict_years(panel, 1950, 1990)
        outside = (panel.years < 1950) | (panel.years > 1990)
        assert not cut.available[outside].any()
        assert cut.available[~outside].any()

    def test_the_window_never_adds_availability(self, real_config_or_skip
                                                ) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        cut = pr.restrict_years(panel, 1890, 2100)
        assert np.array_equal(cut.available, panel.available)

    def test_period_summary_reports_one_row_per_window(self) -> None:
        periods = pd.concat([
            pr._tag(_summary(1.10, 1.00), window="A", start_year=1890,
                    end_year=1950, years=61, country_years=891),
            pr._tag(_summary(1.02, 1.00), window="B", start_year=1890,
                    end_year=2020, years=131, country_years=2010),
        ], ignore_index=True)
        out = pr.period_summary(periods)
        assert list(out["window"]) == ["A", "B"]
        assert float(out.loc[0, "gap_pct"]) == pytest.approx(10.0)
        assert bool(out["ordering_holds"].all())


class TestNoiseFloor:
    def test_the_spread_across_seeds_is_reported(self) -> None:
        floor = pd.concat([
            pr._tag(_summary(1.10, 1.00), seed=1),      # 10%
            pr._tag(_summary(1.12, 1.00), seed=2),      # 12%
            pr._tag(_summary(1.11, 1.00), seed=3),      # 11%
        ], ignore_index=True)
        out = pr.floor_summary(floor)
        assert out["seeds"] == 3
        assert out["range_pct"] == pytest.approx(2.0)
        assert out["min_gap_pct"] == pytest.approx(10.0)
        assert out["max_gap_pct"] == pytest.approx(12.0)

    def test_each_seed_is_actually_passed_through(self) -> None:
        seen = []

        def summarise(panel, paths, override=None):
            seen.append(int(override["bootstrap"]["seed"]))
            return _summary(1.1, 1.0)

        cfg = {"bootstrap": {"seed": 1, "n_paths": 10},
               "data": {}, "run": {}}

        class FakePanel:
            countries = ("AAA", "BBB")

        import unittest.mock as mock
        with mock.patch.object(dl, "build_tier_a",
                               return_value=FakePanel()):
            pr.noise_floor(cfg, summarise, 10, seeds=[7, 8, 9])
        assert seen == [7, 8, 9]

    def test_the_callers_own_config_is_not_mutated(self) -> None:
        cfg = {"bootstrap": {"seed": 1}, "data": {}, "run": {}}

        import unittest.mock as mock
        with mock.patch.object(dl, "build_tier_a", return_value=object()):
            pr.noise_floor(cfg, lambda p, n, override=None: _summary(1.1, 1.0),
                           10, seeds=[7])
        assert cfg["bootstrap"]["seed"] == 1


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
class TestVerdict:
    @staticmethod
    def _influence(gaps):
        base = 6.0
        return pd.DataFrame({
            "dropped": [f"C{i}" for i in range(len(gaps))],
            "gap_pct": gaps,
            "shift_pct": [g - base for g in gaps],
            "ordering_holds": [g > 0.0 for g in gaps],
        }).sort_values("shift_pct").reset_index(drop=True)

    def test_a_surviving_panel_is_reported_as_surviving(self) -> None:
        infl = self._influence([4.0, 5.0, 6.0, 7.0])
        out = pr.verdict(infl, pr.jackknife(infl, 6.0), {"range_pct": 0.2})
        assert out["survives_every_deletion"]
        assert out["n_deletions_that_break_it"] == 0
        assert out["worst_country"] == "C0"

    def test_a_breaking_deletion_is_counted(self) -> None:
        infl = self._influence([-1.0, 5.0, 6.0, 7.0])
        out = pr.verdict(infl, pr.jackknife(infl, 6.0), {"range_pct": 0.2})
        assert not out["survives_every_deletion"]
        assert out["n_deletions_that_break_it"] == 1

    def test_shifts_inside_the_noise_floor_are_not_called_material(self
                                                                   ) -> None:
        infl = self._influence([5.95, 6.0, 6.05, 6.02])
        out = pr.verdict(infl, pr.jackknife(infl, 6.0), {"range_pct": 1.0})
        assert out["n_material_countries"] == 0
        assert out["material_countries"] == []

    def test_a_large_shift_clears_the_noise_floor(self) -> None:
        infl = self._influence([2.0, 6.0, 6.05, 6.02])
        out = pr.verdict(infl, pr.jackknife(infl, 6.0), {"range_pct": 0.5})
        assert "C0" in out["material_countries"]

    def test_window_stability_is_summarised(self) -> None:
        infl = self._influence([4.0, 5.0, 6.0, 7.0])
        period = pd.DataFrame({"window": ["A", "B"], "gap_pct": [3.0, 6.0],
                               "ordering_holds": [True, True]})
        out = pr.verdict(infl, pr.jackknife(infl, 6.0), {"range_pct": 0.2},
                         period)
        assert out["all_windows_hold"]
        assert out["weakest_window"] == "A"
        assert out["weakest_window_gap_pct"] == pytest.approx(3.0)
