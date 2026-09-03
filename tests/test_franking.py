"""Dividend imputation, and the wedge it closes with withholding.

Four properties carry this section. The credit formula must be right at every
corner and must be free to come back negative, because a fund taxed on an
unfranked dividend really is worse off than the untaxed baseline and a formula
that could only produce a credit would be a thumb on the scale. The credit must
land on the home leg and nowhere else, since a resident-only credit that
touched the foreign leg would be modelling a different tax. The wedge must be
both taxes at once rather than either alone. And the sample the bootstrap can
draw must not move when the credit does, or the comparison stops being paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import franking as fk
from src import withholding as wh


class TestCreditRate:
    def test_the_classical_corner_is_zero(self) -> None:
        """No franking and no fund tax is this paper's own baseline."""
        assert fk.credit_rate(0.30, 0.0, 0.0) == pytest.approx(0.0)

    def test_a_pension_phase_fund_collects_the_whole_corporate_tax(self) -> None:
        # 30/70: the company paid 30 on a pre-tax 100 and the fund, taxed at
        # nothing, is handed all of it back.
        assert fk.credit_rate(0.30, 0.0, 1.0) == pytest.approx(0.30 / 0.70)

    def test_a_taxed_fund_keeps_less_of_it(self) -> None:
        assert fk.credit_rate(0.30, 0.15, 1.0) == pytest.approx(0.85 / 0.70 - 1)

    def test_an_unfranked_dividend_in_a_taxed_fund_is_a_drag(self) -> None:
        """The sign the formula is allowed to produce, and must.

        A credit-only formula would report zero here and quietly flatter the
        home leg at every partial franking level below the break-even.
        """
        assert fk.credit_rate(0.30, 0.15, 0.0) == pytest.approx(-0.15)

    def test_it_is_linear_in_the_franked_share(self) -> None:
        lo = fk.credit_rate(0.30, 0.15, 0.0)
        hi = fk.credit_rate(0.30, 0.15, 1.0)
        assert fk.credit_rate(0.30, 0.15, 0.5) == pytest.approx((lo + hi) / 2)

    def test_a_higher_corporate_rate_imputes_more(self) -> None:
        assert (fk.credit_rate(0.35, 0.0, 1.0)
                > fk.credit_rate(0.25, 0.0, 1.0))

    def test_impossible_rates_are_refused(self) -> None:
        for args in ((1.0, 0.0, 1.0), (0.3, 1.5, 1.0), (0.3, 0.0, 1.5),
                     (-0.1, 0.0, 1.0)):
            with pytest.raises(ValueError):
                fk.credit_rate(*args)

    def test_the_anchors_are_derived_not_written_down(self) -> None:
        frame = fk.anchor_credits()
        for _, row in frame.iterrows():
            assert float(row["credit"]) == pytest.approx(fk.credit_rate(
                float(row["company_tax"]), float(row["fund_tax"]),
                float(row["franked_share"])))

    def test_the_franking_grid_crosses_zero(self) -> None:
        """A taxed fund needs some franking before the credit is worth anything."""
        grid = fk.franking_grid(0.30, 0.15)
        assert float(grid["credit"].iloc[0]) < 0
        assert float(grid["credit"].iloc[-1]) > 0


def _panel(n_t: int = 12, n_c: int = 3, seed: int = 0):
    from src.data_loader import Panel

    rng = np.random.default_rng(seed)
    shape = (n_t, n_c)
    return Panel(
        years=np.arange(2000, 2000 + n_t),
        countries=tuple(f"C{i}" for i in range(n_c)),
        tier=("A",) * n_c,
        dom_eq=rng.normal(0.07, 0.18, shape),
        intl_eq=rng.normal(0.06, 0.17, shape),
        bond=rng.normal(0.02, 0.08, shape),
        bill=rng.normal(0.01, 0.04, shape),
        inflation=rng.normal(0.03, 0.04, shape),
        real_exchange_rate=np.ones(shape),
        available=np.ones(shape, dtype=bool),
        name="toy", provenance=("test",) * n_c)


class TestApplyFranking:
    def test_it_multiplies_the_home_leg_by_one_plus_c_q(self) -> None:
        panel = _panel()
        q = np.full(panel.dom_eq.shape, 0.04)
        out = fk.apply_franking(panel, 0.4286, q)
        assert np.allclose(1.0 + out.dom_eq,
                           (1.0 + panel.dom_eq) * (1.0 + 0.4286 * 0.04))

    def test_it_leaves_every_other_series_alone(self) -> None:
        """A resident's credit that touched the foreign leg would be a
        different tax, and one nobody pays."""
        panel = _panel()
        out = fk.apply_franking(panel, 0.4286,
                                np.full(panel.dom_eq.shape, 0.04))
        for key in ("intl_eq", "bond", "bill", "inflation"):
            assert np.array_equal(getattr(out, key), getattr(panel, key))

    def test_a_zero_credit_returns_the_panel_untouched(self) -> None:
        panel = _panel()
        assert fk.apply_franking(panel, 0.0,
                                 np.full(panel.dom_eq.shape, 0.04)) is panel

    def test_a_negative_credit_lowers_the_home_leg(self) -> None:
        panel = _panel()
        out = fk.apply_franking(panel, -0.15,
                                np.full(panel.dom_eq.shape, 0.04))
        assert (out.dom_eq < panel.dom_eq).all()

    def test_the_sample_the_bootstrap_can_draw_does_not_move(self) -> None:
        """Every credit must admit the same blocks, or the comparison stops
        being paired and starts confounding the tax with a change of sample."""
        panel = _panel()
        q = np.full(panel.dom_eq.shape, 0.04)
        for credit in (-0.15, 0.0, 0.2143, 0.4286):
            out = fk.apply_franking(panel, credit, q)
            assert np.array_equal(out.available, panel.available)
            assert out.countries == panel.countries
            assert np.array_equal(out.years, panel.years)

    def test_a_missing_dividend_credits_nothing_rather_than_dropping(self
                                                                     ) -> None:
        panel = _panel()
        q = np.full(panel.dom_eq.shape, 0.04)
        q[3, 1] = np.nan
        out = fk.apply_franking(panel, 0.4286, q)
        assert out.dom_eq[3, 1] == pytest.approx(panel.dom_eq[3, 1])
        assert np.isfinite(out.dom_eq).all()

    def test_the_name_records_the_credit(self) -> None:
        panel = _panel()
        out = fk.apply_franking(panel, 0.4286,
                                np.full(panel.dom_eq.shape, 0.04))
        assert "frank" in out.name and panel.name in out.name


class TestTheWedge:
    def test_it_is_both_taxes_and_neither_alone(self) -> None:
        panel = _panel()
        q = np.full(panel.dom_eq.shape, 0.04)
        both = fk.apply_wedge(panel, 0.4286, q, 0.15, q)
        home = fk.apply_franking(panel, 0.4286, q)
        away = wh.apply_withholding(panel, 0.15, q)
        assert np.allclose(both.dom_eq, home.dom_eq)
        assert np.allclose(both.intl_eq, away.intl_eq)
        # ...and each blade on its own leaves the other leg untouched.
        assert np.allclose(home.intl_eq, panel.intl_eq)
        assert np.allclose(away.dom_eq, panel.dom_eq)

    def test_the_two_blades_push_the_same_way(self) -> None:
        panel = _panel()
        q = np.full(panel.dom_eq.shape, 0.04)
        both = fk.apply_wedge(panel, 0.4286, q, 0.15, q)
        assert (both.dom_eq > panel.dom_eq).all()
        assert (both.intl_eq < panel.intl_eq).all()

    def test_the_named_positions_start_from_the_papers_baseline(self) -> None:
        positions = fk.wedge_positions(rate=0.15)
        assert positions[0][1] == 0.0 and positions[0][2] == 0.0
        # The second isolates withholding, so the table reads as a
        # decomposition rather than as a list of unrelated scenarios.
        assert positions[1][1] == 0.0 and positions[1][2] == pytest.approx(0.15)

    def test_every_named_position_carries_a_derived_credit(self) -> None:
        by_label = {row["label"]: float(row["credit"])
                    for _, row in fk.anchor_credits().iterrows()}
        for label, credit, rate in fk.wedge_positions(rate=0.15)[2:]:
            assert credit == pytest.approx(by_label[label])
            assert rate == pytest.approx(0.15)

    def test_a_zero_credit_anchor_is_not_repeated_as_a_position(self) -> None:
        """``no imputation`` is already the second row; a third would be it."""
        labels = [p[0] for p in fk.wedge_positions()]
        assert "no imputation, no fund tax" not in labels


class TestEffectiveCredit:
    def test_it_is_the_credit_times_the_dividend_share(self) -> None:
        q = np.array([[0.02, 0.05]])
        assert np.allclose(fk.effective_credit(0.4286, q), 0.4286 * q)

    def test_the_era_table_falls_with_the_dividend_yield(self) -> None:
        """The same law delivers less as payout ratios fall, which is the
        whole reason the credit is not quoted as a flat number."""
        years = np.arange(1890, 2021)
        q = np.linspace(0.06, 0.02, years.size)[:, None]
        out = fk.realised_credit(0.4286, q, years)
        eras = out[out["era"] != "whole panel"]
        assert list(eras["credit_bp"]) == sorted(eras["credit_bp"],
                                                 reverse=True)

    def test_a_negative_credit_shows_as_a_negative_era_row(self) -> None:
        years = np.arange(2000, 2020)
        q = np.full((years.size, 1), 0.04)
        out = fk.realised_credit(-0.15, q, years)
        assert (out["credit_bp"] < 0).all()


def _curve(leads: dict[str, list[float]],
           credits: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"credit": credits,
                          "credit_pct": [c * 100.0 for c in credits]})
    for rival, values in leads.items():
        frame[f"lead_over_{rival}_pct"] = values
    frame["winner"] = ["challenger" if v > 0 else rival
                       for rival, values in leads.items() for v in values][
        :len(credits)]
    return frame


class TestCrossing:
    def test_it_interpolates_between_the_straddling_points(self) -> None:
        curve = _curve({"rival": [2.0, -2.0]}, [0.0, 0.20])
        assert fk.crossing(curve, "rival") == pytest.approx(0.10)

    def test_a_lead_that_survives_the_grid_is_infinite_not_missing(self
                                                                  ) -> None:
        curve = _curve({"rival": [5.0, 3.0]}, [0.0, 0.60])
        assert fk.crossing(curve, "rival") == float("inf")

    def test_a_lead_that_was_never_there_is_zero(self) -> None:
        curve = _curve({"rival": [-1.0, -4.0]}, [0.0, 0.60])
        assert fk.crossing(curve, "rival") == 0.0


class TestVerdict:
    @staticmethod
    def _inputs(winners: list[str], crossing_pct: float = 19.8):
        credits = fk.anchor_credits()
        curve = pd.DataFrame({
            "credit": [-0.15, 0.0, 0.4286],
            "credit_pct": [-15.0, 0.0, 42.86],
            "winner": ["a", "a", "b"]})
        crossed = pd.DataFrame({
            "rival": ["balanced_all_equity"],
            "crossing_credit": [crossing_pct / 100.0],
            "crossing_pct": [crossing_pct],
            "equivalent_credit_bp": [75.0],
            "reached_on_grid": [True],
            "lead_at_zero_pct": [5.8]})
        optima = pd.DataFrame({
            "credit": [-0.15, 0.0, 0.4286],
            "optimal_domestic_share": [0.0, 0.25, 0.50],
            "cec_at_optimum": [1.0, 1.06, 1.10],
            "margin_over_runner_up_pct": [0.35, 1.02, 0.19]})
        comparison = pd.DataFrame({
            "position": ["neither tax", "withholding only", "pension"],
            "credit": [0.0, 0.0, 0.4286], "rate": [0.0, 0.15, 0.15],
            "winner": winners})
        return curve, crossed, optima, credits, comparison

    def test_it_reports_the_wedge_overturning_the_headline(self) -> None:
        found = fk.verdict(*self._inputs(
            ["international_equity", "international_equity",
             "balanced_all_equity"]), "international_equity")
        assert found["wedge_overturns_the_headline"]
        assert found["wedge_winner_at_baseline"] == "international_equity"
        assert found["wedge_winner_at_the_end"] == "balanced_all_equity"

    def test_it_reports_a_headline_that_survives(self) -> None:
        found = fk.verdict(*self._inputs(["international_equity"] * 3),
                           "international_equity")
        assert not found["wedge_overturns_the_headline"]
        assert not found["wedge_winner_changes"]

    def test_the_crossing_is_placed_against_the_real_anchors(self) -> None:
        """Whether a real fund clears it is the question; both anchors are
        recomputed rather than compared against a written-down number."""
        inside = fk.verdict(*self._inputs(["a", "a", "b"], crossing_pct=19.8),
                            "a")
        assert inside["crossing_within_accumulation"]
        assert inside["crossing_within_pension_phase"]
        between = fk.verdict(*self._inputs(["a", "a", "b"], crossing_pct=30.0),
                             "a")
        assert not between["crossing_within_accumulation"]
        assert between["crossing_within_pension_phase"]
        beyond = fk.verdict(*self._inputs(["a", "a", "b"], crossing_pct=90.0),
                            "a")
        assert not beyond["crossing_within_accumulation"]
        assert not beyond["crossing_within_pension_phase"]

    def test_the_neutral_row_is_carried_separately_from_the_grid_floor(self
                                                                      ) -> None:
        """The grid starts below zero, where a taxed fund is losing. A reader
        comparing against 'no credit' needs that row, not the floor."""
        found = fk.verdict(*self._inputs(["a", "a", "b"]), "a")
        assert found["optimal_domestic_at_bottom"] == pytest.approx(0.0)
        assert found["optimal_domestic_at_zero"] == pytest.approx(0.25)
        assert found["optimal_domestic_at_top"] == pytest.approx(0.50)
        assert found["optimum_moves_home"]

    def test_an_empty_curve_reports_nothing(self) -> None:
        assert fk.verdict(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                          pd.DataFrame(), pd.DataFrame(), "a") == {"levels": 0}


class TestWedgeComparison:
    def test_it_names_the_best_of_the_strategies_given(self) -> None:
        frame = pd.DataFrame({
            "position": ["p"] * 3, "credit": [0.4] * 3, "rate": [0.15] * 3,
            "strategy": ["a", "b", "c"], "cec": [1.0, 1.2, 0.9]})
        out = fk.wedge_comparison(frame, "cec", ["a", "b", "c"])
        assert out["winner"].iloc[0] == "b"
        assert out["best_cec"].iloc[0] == pytest.approx(1.2)

    def test_a_strategy_absent_from_the_frame_is_skipped(self) -> None:
        frame = pd.DataFrame({
            "position": ["p"] * 2, "credit": [0.4] * 2, "rate": [0.15] * 2,
            "strategy": ["a", "b"], "cec": [1.0, 1.2]})
        out = fk.wedge_comparison(frame, "cec", ["a", "b", "missing"])
        assert "cec_missing" not in out.columns
        assert out["winner"].iloc[0] == "b"

    def test_positions_keep_the_order_they_were_scored_in(self) -> None:
        frame = pd.DataFrame({
            "position": ["second", "second", "first", "first"],
            "credit": [0.4, 0.4, 0.0, 0.0], "rate": [0.15] * 4,
            "strategy": ["a", "b"] * 2, "cec": [1.0, 1.2, 1.1, 1.0]})
        out = fk.wedge_comparison(frame, "cec", ["a", "b"])
        assert list(out["position"]) == ["second", "first"]


class TestBreakEvenFrankedShare:
    """How much franking a taxed fund needs before the credit pays for itself."""

    def test_it_zeroes_the_credit(self) -> None:
        for company, fund in ((0.30, 0.15), (0.25, 0.15), (0.30, 0.30)):
            phi = fk.break_even_franked_share(company, fund)
            assert fk.credit_rate(company, fund, phi) == pytest.approx(0.0)

    def test_an_untaxed_fund_needs_none(self) -> None:
        """It is already level with the baseline, so any franking is a gain."""
        assert fk.break_even_franked_share(0.30, 0.0) == 0.0

    def test_a_taxed_fund_needs_a_real_share_of_it(self) -> None:
        phi = fk.break_even_franked_share(0.30, 0.15)
        assert 0.0 < phi < 1.0
        assert fk.credit_rate(0.30, 0.15, phi * 0.5) < 0
        assert fk.credit_rate(0.30, 0.15, min(phi * 1.5, 1.0)) > 0

    def test_no_corporate_tax_means_no_amount_of_franking_helps(self) -> None:
        assert fk.break_even_franked_share(0.0, 0.15) == float("inf")

    def test_a_lower_corporate_rate_demands_more_franking(self) -> None:
        assert (fk.break_even_franked_share(0.25, 0.15)
                > fk.break_even_franked_share(0.30, 0.15))


class TestAnchorParameters:
    """The pipeline, the document and the paper must read one number, once."""

    def test_the_accumulating_anchor_exists(self) -> None:
        company, fund, franked = fk.anchor_parameters(fk.ACCUMULATING)
        assert fk.ACCUMULATING in fk.ANCHORS
        assert franked == 1.0
        assert fund > 0.0
        assert company > 0.0

    def test_it_agrees_with_the_derived_credit_table(self) -> None:
        company, fund, franked = fk.anchor_parameters(fk.ACCUMULATING)
        row = fk.anchor_credits().set_index("label").loc[fk.ACCUMULATING]
        assert float(row["company_tax"]) == pytest.approx(company)
        assert float(row["fund_tax"]) == pytest.approx(fund)
        assert float(row["credit"]) == pytest.approx(
            fk.credit_rate(company, fund, franked))

    def test_an_unknown_anchor_names_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown anchor"):
            fk.anchor_parameters("a fund that does not exist")
