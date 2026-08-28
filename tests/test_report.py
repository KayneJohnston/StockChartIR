"""Tests for Markdown rendering and the dominance check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import report as rp


class TestMarkdownTable:
    def test_renders_a_header_rule_and_rows(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.25]})
        lines = rp.md_table(frame).splitlines()
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | 0.5000 |"

    def test_formats_floats_and_booleans(self) -> None:
        frame = pd.DataFrame({"x": [1.23456], "flag": [True]})
        rendered = rp.md_table(frame, floatfmt="{:.2f}")
        assert "| 1.23 | yes |" in rendered

    def test_nan_renders_as_a_dash(self) -> None:
        rendered = rp.md_table(pd.DataFrame({"x": [np.nan]}))
        assert "| -- |" in rendered

    def test_positive_infinity_renders_as_never(self) -> None:
        # An infinite crossover means the rival never overtakes, which is a
        # different statement from "not available".
        rendered = rp.md_table(pd.DataFrame({"x": [np.inf, -np.inf]}))
        assert "| never |" in rendered
        assert "| -inf |" in rendered

    def test_truncation_is_announced(self) -> None:
        frame = pd.DataFrame({"x": range(10)})
        rendered = rp.md_table(frame, max_rows=3)
        assert "7 further rows omitted" in rendered


class TestDominanceCheck:
    def _table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "strategy": ["challenger", "weaker", "tied", "stronger"],
            "label": ["C", "W", "T", "S"],
            "cec_crra_gamma5": [1.0, 0.8, 1.0, 1.2],
            "prob_ruin": [0.10, 0.20, 0.10, 0.05],
            "median_bequest": [10.0, 5.0, 10.0, 20.0],
        })

    def test_detects_strict_dominance(self) -> None:
        result = rp.dominance_check(self._table(), "challenger", ["weaker"])
        row = result.iloc[0]
        assert bool(row["strict_dominance"])
        assert row["criteria_lost"] == "-"
        assert row["criteria_won"] == 3

    def test_ties_are_not_losses(self) -> None:
        result = rp.dominance_check(self._table(), "challenger", ["tied"])
        row = result.iloc[0]
        assert bool(row["strict_dominance"])
        assert row["criteria_tied"] == 3
        assert row["criteria_won"] == 0

    def test_records_the_criteria_lost(self) -> None:
        result = rp.dominance_check(self._table(), "challenger", ["stronger"])
        row = result.iloc[0]
        assert not bool(row["strict_dominance"])
        assert "cec_crra_gamma5" in row["criteria_lost"]
        assert "prob_ruin" in row["criteria_lost"]

    def test_lower_is_better_for_ruin(self) -> None:
        table = pd.DataFrame({
            "strategy": ["a", "b"], "label": ["A", "B"],
            "prob_ruin": [0.01, 0.50],
        })
        assert bool(rp.dominance_check(table, "a", ["b"]).iloc[0][
            "strict_dominance"])
        assert not bool(rp.dominance_check(table, "b", ["a"]).iloc[0][
            "strict_dominance"])

    def test_missing_strategies_are_skipped(self) -> None:
        result = rp.dominance_check(self._table(), "challenger", ["absent"])
        assert result.empty
