"""Tests for the out-of-sample transfer check.

The experiment's whole value rests on two things being true: the schedule is
solved on one window and scored on a *different* one, and the three
references it is scored against are computed on the same paths. Both are easy
to get wrong silently, so both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import oos


class TestSplit:
    def test_the_halves_do_not_share_a_year(self, toy_panel) -> None:
        halves = oos.split(toy_panel, 2009)
        first, second = (h.available for h in halves.values())
        assert not (first & second).any()

    def test_together_they_are_the_whole_record(self, toy_panel) -> None:
        halves = oos.split(toy_panel, 2009)
        first, second = (h.available for h in halves.values())
        assert np.array_equal(first | second, toy_panel.available)

    def test_only_availability_changes(self, toy_panel) -> None:
        # Masking rather than rebuilding: the sleeve in any year has to stay
        # the sleeve that year actually had.
        half = next(iter(oos.split(toy_panel, 2009).values()))
        assert np.array_equal(half.intl_eq, toy_panel.intl_eq, equal_nan=True)
        assert half.countries == toy_panel.countries

    def test_a_cut_outside_the_record_is_an_error(self, toy_panel) -> None:
        with pytest.raises(ValueError, match="outside"):
            oos.split(toy_panel, 1800)

    def test_the_windows_are_named_by_their_years(self, toy_panel) -> None:
        assert set(oos.split(toy_panel, 2009)) == {"2000-2009", "2010-2019"}


class TestSolve:
    def test_an_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown schedule family"):
            oos.solve("astrology", None, 5.0, {})

    def test_every_advertised_family_is_reachable(self) -> None:
        for family in oos.FAMILIES:
            assert family in oos.LABELS


class TestArithmetic:
    @staticmethod
    def _frame(in_sample, transferred, ceiling):
        return pd.DataFrame({
            "family": ["glide", "simplex"],
            "label": ["Glide", "Simplex"],
            "train_window": ["A", "B"],
            "test_window": ["B", "A"],
            "benchmark": ["international_equity"] * 2,
            "in_sample_gain_pct": in_sample,
            "transfer_gain_pct": transferred,
            "ceiling_gain_pct": ceiling,
            "retained_share": [t / s if s else np.nan
                               for t, s in zip(transferred, in_sample)],
        })

    def test_a_total_failure_is_reported_as_one(self) -> None:
        frame = self._frame([2.0, 3.0], [-1.0, -2.0], [2.5, 3.5])
        found = oos.verdict(frame, {"stable": True, "same_winner": True})
        assert found["no_run_transfers"]
        assert not found["every_run_transfers"]
        assert found["runs_that_beat_the_benchmark"] == 0

    def test_a_clean_transfer_is_reported_as_one(self) -> None:
        frame = self._frame([2.0, 3.0], [1.0, 2.0], [2.5, 3.5])
        found = oos.verdict(frame, {"stable": True, "same_winner": True})
        assert found["every_run_transfers"]
        assert found["median_retained_share"] == pytest.approx(0.5833, abs=1e-3)

    def test_the_retained_share_is_transfer_over_in_sample(self) -> None:
        frame = self._frame([4.0, 4.0], [1.0, 3.0], [4.0, 4.0])
        assert frame["retained_share"].tolist() == [0.25, 0.75]

    def test_a_mixed_result_names_which_family_failed(self) -> None:
        frame = self._frame([2.0, 3.0], [1.0, -2.0], [2.5, 3.5])
        found = oos.verdict(frame, {"stable": False, "same_winner": True})
        assert found["families_that_transfer"] == ["glide"]
        assert found["families_that_do_not"] == ["simplex"]
        assert found["worst_family"] == "simplex"

    def test_an_empty_experiment_does_not_crash(self) -> None:
        assert oos.verdict(pd.DataFrame(), {})["families"] == 0

    def test_a_direction_that_transfers_only_one_way_is_flagged(self) -> None:
        frame = self._frame([2.0, 2.0], [1.0, -1.0], [2.0, 2.0])
        frame["train_window"] = ["1890-1955", "1956-2020"]
        frame["test_window"] = ["1956-2020", "1890-1955"]
        found = oos.verdict(frame, {"stable": True, "same_winner": True})
        assert found["asymmetric"]
        assert found["transfers_forward"] and not found["transfers_backward"]
        assert found["earlier_window"] == "1890-1955"

    def test_a_symmetric_result_is_not_flagged_asymmetric(self) -> None:
        frame = self._frame([2.0, 2.0], [1.0, 0.5], [2.0, 2.0])
        frame["train_window"] = ["1890-1955", "1956-2020"]
        found = oos.verdict(frame, {"stable": True, "same_winner": True})
        assert not found["asymmetric"]


class TestStability:
    @staticmethod
    def _benchmarks(first, second):
        rows = []
        for window, order in (("A", first), ("B", second)):
            for rank, key in enumerate(order, start=1):
                rows.append({"window": window, "strategy": key, "label": key,
                             "cec": 2.0 - 0.1 * rank, "rank": rank})
        return pd.DataFrame.from_records(rows)

    def test_an_identical_order_is_stable(self) -> None:
        found = oos.ranking_is_stable(self._benchmarks("abc", "abc"))
        assert found["stable"] and found["n_positions_moved"] == 0

    def test_a_reordering_is_reported_with_its_size(self) -> None:
        found = oos.ranking_is_stable(self._benchmarks("abc", "acb"))
        assert not found["stable"]
        assert found["same_winner"]
        assert found["n_positions_moved"] == 2

    def test_a_changed_winner_is_flagged(self) -> None:
        found = oos.ranking_is_stable(self._benchmarks("abc", "bac"))
        assert not found["same_winner"]
