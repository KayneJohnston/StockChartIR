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


class TestCostOfWorkingDocRenders:
    """`write_doc_32` is a long f-string over a dozen frames, and nothing but
    a four-hundred-second pipeline run used to exercise it. A shadowed
    variable name reached production once; this renders the whole document
    from synthetic frames so the next one costs a second instead."""

    @staticmethod
    def _frames() -> dict:
        from src import leisure as le

        ages = [55, 60, 63, 67]
        leisure = [1.0, 1.25, 1.50]
        swept = pd.DataFrame([
            {"leisure": g, "retire_age": a, "cec": 0.7 - 0.01 * abs(a - 60)
             - 0.05 * (g - 1.0), "prob_ruin": 0.12,
             "mean_retirement_consumption": 1.5, "n_paths": 100,
             "cec_survival_weighted": 0.7}
            for g in leisure for a in ages])
        optimal = le.optimal_age(swept)
        crossings = le.break_even(swept)
        anchors = pd.DataFrame({
            "label": ["no value on leisure", "a 20% consumption drop"],
            "consumption_drop": [0.0, 0.20],
            "leisure": [1.0, 1.25], "leisure_pct": [0.0, 25.0]})
        claim = pd.DataFrame({
            "retire_age": ages, "claim_factor": [0.6, 0.8, 1.0, 1.3],
            "adjustment_pct": [-40.0, -20.0, 0.0, 30.0],
            "per_year_pct": [-5.0, -6.0, np.nan, -7.0]})
        systems, sys_break, sys_opt = [], [], {}
        for name in le.SYSTEMS:
            block = swept.assign(system=name)
            systems.append(block)
            sys_opt[name] = le.optimal_age(block).assign(system=name)
            sys_break.append(le.break_even(block).assign(system=name))
        comparison = le.system_comparison(
            sys_opt, {n: b for n, b in zip(le.SYSTEMS, sys_break)})
        features = pd.DataFrame({
            "system": list(le.FEATURE_ARMS),
            "age_at_zero_leisure": [60, 58, 67, 67],
            "age_at_top": [52, 50, 60, 60],
            "cec_at_zero_leisure": [0.76, 0.77, 0.62, 0.62],
            "cost_per_year_pct": [4.9, 1.2, 19.2, 19.7]})
        decomposition = le.feature_decomposition(features)
        rules = pd.DataFrame([
            {"system": s, "rule": r, "cec": 0.6, "prob_ruin": 0.12,
             "mean_consumption": 1.2, "p5_consumption": 0.4,
             "median_wealth": 20.0}
            for s in le.SYSTEMS
            for r in ("fixed_real_rule", "replace_50", "replace_100")])
        return {
            "swept": swept, "claim": claim, "anchors": anchors,
            "optimal": optimal, "break_even": crossings,
            "unadjusted_optimal": optimal,
            "systems": pd.concat(systems, ignore_index=True),
            "system_comparison": comparison,
            "system_break_even": pd.concat(sys_break, ignore_index=True),
            "features": features, "decomposition": decomposition,
            "rules": rules,
        }

    @staticmethod
    def _notes(frames: dict) -> dict:
        from src import leisure as le

        return {
            "elapsed_seconds": 1.0, "gamma": 5.0, "n_paths": 100,
            "strategy": "balanced_all_equity", "reference_age": 63,
            "arms": ["unadjusted", "actuarial"], "headline_arm": "actuarial",
            "verdict": le.verdict(frames["swept"], frames["optimal"],
                                  frames["break_even"], frames["anchors"]),
            "systems": list(le.SYSTEMS), "age_pension_age": 67,
            "safety_net": 0.60,
            "system_verdict": le.system_verdict(frames["system_comparison"]),
            "feature_verdict": le.feature_verdict(frames["decomposition"]),
            "means_test_bite": {
                "economy_average_income": 1.5, "full_rate": 0.45,
                "full_rate_replacement": 0.29, "free_area_multiple": 3.0,
                "cutoff_multiple": 6.8, "median_wealth": 21.0,
                "median_wealth_multiple": 13.8, "median_over_cutoff": 2.0,
                "share_above_cutoff": 0.83, "mean_benefit": 0.06,
                "benefit_replacement": 0.04,
                "share_receiving_nothing": 0.59,
                "us_benefit_replacement": 0.43, "reference_age": 63.0},
            "rule_verdict": le.rule_verdict(
                frames["rules"], portfolio_rule="fixed_real_rule"),
            "replacement_targets": [0.50, 1.00],
        }

    def test_the_whole_document_renders(self, tmp_path) -> None:
        frames = self._frames()
        out = rp.write_doc_32(tmp_path / "32.md", _cfg(), frames,
                              ["results/figures/fig59.png"],
                              self._notes(frames))
        assert out.exists()
        assert len(out.read_text()) > 2_000

    def test_every_section_is_numbered_once_and_in_order(self, tmp_path
                                                         ) -> None:
        """A section inserted mid-document has to renumber the ones below it,
        and forgetting to is invisible until a reader trips on two sevens."""
        frames = self._frames()
        rendered = rp.write_doc_32(
            tmp_path / "32.md", _cfg(), frames, [],
            self._notes(frames)).read_text()
        numbers = [int(line.split(".")[0][3:])
                   for line in rendered.splitlines()
                   if line.startswith("## ") and line[3].isdigit()]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_the_new_sections_are_present(self, tmp_path) -> None:
        frames = self._frames()
        rendered = rp.write_doc_32(
            tmp_path / "32.md", _cfg(), frames, [],
            self._notes(frames)).read_text()
        assert "Which of the two differences does the work" in rendered
        assert "The withdrawal rule is half the comparison" in rendered

    def test_it_survives_the_optional_frames_being_absent(self, tmp_path
                                                          ) -> None:
        """The decomposition and rule comparison are additions; a config that
        switches them off must still produce a document."""
        frames = self._frames()
        notes = self._notes(frames)
        for key in ("features", "decomposition", "rules"):
            frames.pop(key)
        notes = {**notes, "feature_verdict": {"measured": False},
                 "rule_verdict": {"measured": False}}
        out = rp.write_doc_32(tmp_path / "32.md", _cfg(), frames, [], notes)
        assert out.exists()


def _cfg() -> dict:
    from src import data_loader as dl

    return dl.load_config("config.yaml")
