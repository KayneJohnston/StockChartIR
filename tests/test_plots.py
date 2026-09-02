"""Tests for the figure layer.

Two failures matter here and neither shows up in a numeric assertion. A tick
label can run off the edge of the canvas, in which case the reader sees
"...ternational Equity" and has to guess; or a label can be a raw config key,
in which case they see ``intl_eq``. Both were present before these tests.

The crop test measures ink at the image border rather than eyeballing the
result, so it fails if anyone reintroduces the original bug -- calling
``_save`` outside the ``rc_context`` that set ``savefig.bbox``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import data_loader as dl
from src import plots


LONG = "100% International Equity"


def _ink_on_the_border(path: Path, depth: int = 2) -> int:
    """Dark pixels within ``depth`` rows/columns of each edge.

    Anything here is text or a line that the crop cut through.
    """
    image = plt.imread(path)
    grey = image[..., :3].mean(axis=-1) if image.ndim == 3 else image
    dark = grey < 0.6
    return int(dark[:depth].sum() + dark[-depth:].sum()
               + dark[:, :depth].sum() + dark[:, -depth:].sum())


class TestSaveCrops:
    def test_a_long_tick_label_is_not_cut_off(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar([0, 1], [1.0, 2.0])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LONG, LONG], rotation=30, ha="right")
        assert _ink_on_the_border(plots._save(fig, tmp_path, "long")) == 0

    def test_a_long_y_label_is_not_cut_off(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh([0, 1], [1.0, 2.0])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([LONG, "60/40 Domestic Equity/Domestic Bonds"])
        assert _ink_on_the_border(plots._save(fig, tmp_path, "wide")) == 0

    def test_the_saved_resolution_is_the_configured_one(self,
                                                        tmp_path: Path) -> None:
        # The bug this guards: `savefig.dpi` in STYLE never reached savefig,
        # so every figure came out at `figure.dpi` instead.
        fig, ax = plt.subplots(figsize=(4.0, 2.0))
        ax.plot([0, 1], [0, 1])
        image = plt.imread(plots._save(fig, tmp_path, "dpi"))
        assert image.shape[0] >= 2.0 * plots.STYLE["savefig.dpi"] * 0.9

    def test_the_directory_is_created(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        out = plots._save(fig, tmp_path / "nested" / "deeper", "f")
        assert out.exists()

    def test_the_figure_is_closed(self, tmp_path: Path) -> None:
        fig, _ = plt.subplots()
        plots._save(fig, tmp_path, "closed")
        assert not plt.fignum_exists(fig.number)


class TestLabels:
    def test_every_configured_strategy_has_a_short_form(
            self, real_config_or_skip) -> None:
        for key, spec in real_config_or_skip["strategies"].items():
            assert key in plots.STRATEGY_LABEL, key
            assert spec["label"] in plots.STRATEGY_LABEL, spec["label"]

    def test_the_key_and_the_label_give_the_same_short_form(self) -> None:
        for key, label in plots.STRATEGY_KEYS.items():
            assert plots._flat(key) == plots._flat(label)

    def test_a_short_form_is_shorter_than_what_it_replaces(self) -> None:
        for long, short in plots.STRATEGY_LABEL.items():
            if " " not in long:            # a bare key, not a display label
                continue
            assert len(short) <= len(long)

    def test_every_panel_series_has_a_display_name(self) -> None:
        for series in dl.CORE_SERIES:
            for table in (plots.SERIES_LABEL, plots.SERIES_ABBR):
                assert series in table
                assert "_" not in table[series]

    def test_an_unknown_label_still_gets_a_readable_form(self) -> None:
        assert plots._flat("some_new_strategy", 40) == "some new strategy"

    def test_nothing_ever_renders_as_an_empty_label(self) -> None:
        for value in ("", "   ", "x"):
            assert plots._flat(value) is not None
            assert plots._abbr(value) is not None

    def test_wrapping_keeps_the_breaks_a_short_form_chose(self) -> None:
        assert plots._wrap("Dom.\nequity", 40) == "Dom.\nequity"

    def test_wrapping_still_breaks_a_long_run_of_words(self) -> None:
        assert "\n" in plots._wrap("one two three four five six", 10)

    def test_a_strategy_label_never_stacks_three_lines(self) -> None:
        # A y-axis tick has one row of height; three lines collide with the
        # bars either side of it.
        for key in plots.STRATEGY_LABEL:
            assert plots._flat(key).count("\n") <= 1, key

    def test_the_short_form_keeps_the_words(self) -> None:
        assert plots._flat("international_equity").replace("\n", " ") \
            == "100% international equity"

    def test_no_short_form_leaves_a_gap_where_a_break_was(self) -> None:
        for value in plots.STRATEGY_LABEL.values():
            assert "/ " not in value
            assert "  " not in value

    def test_a_variant_name_is_compressed_but_still_identifies_itself(self
                                                                      ) -> None:
        assert plots._variant("Wealth trigger 20x income") == "Trigger 20x"
        assert plots._variant("Fixed age 63 (baseline)") == "Age 63 (base)"
        assert plots._variant("Flexible +/-3 years, 25x income") \
            == "Flex \u00b13y, 25x"


class TestGrid:
    def test_a_row_of_four_wraps_into_a_grid(self) -> None:
        fig, axes = plots._grid(4, 2.5)
        assert len(axes) == 4
        # Two columns, two rows: as wide as the page and twice the panel high.
        assert fig.get_figwidth() == pytest.approx(plots.PAGE_WIDTH_IN)
        assert fig.get_figheight() == pytest.approx(5.0)
        plt.close(fig)

    def test_an_odd_panel_spans_the_hole(self) -> None:
        fig, axes = plots._grid(3, 2.5)
        widths = [ax.get_position().width for ax in axes]
        assert widths[2] > widths[0] * 1.5
        plt.close(fig)

    def test_the_hole_can_be_left_for_a_legend(self) -> None:
        fig, axes, holes = plots._grid(5, 2.5, span_last=False, spare=True)
        assert len(axes) == 5 and len(holes) == 1
        plt.close(fig)

    def test_no_figure_is_authored_wider_than_the_text_column(self) -> None:
        for n in (1, 2, 3, 4, 5, 6):
            fig, _ = plots._grid(n, 2.5)
            assert fig.get_figwidth() <= plots.PAGE_WIDTH_IN + 1e-9
            plt.close(fig)


class TestEveryFigureIsAuthoredForThePage:
    """A source check, because the failure it guards against is invisible.

    A figure drawn twenty inches wide still looks fine on its own; it only
    falls apart once the paper scales it into a 16.2 cm column, which no
    other test in this suite exercises.
    """

    @staticmethod
    def _sources() -> str:
        return (Path(__file__).resolve().parents[1]
                / "src" / "plots.py").read_text()

    def test_no_call_sets_its_own_figure_width(self) -> None:
        import re
        bad = [m.group(0) for m
               in re.finditer(r"plt\.subplots\(figsize=\([^)]*\)", self._sources())
               if "PAGE_WIDTH_IN" not in m.group(0)]
        assert bad == []

    def test_the_page_width_matches_the_paper(self) -> None:
        # paper/style.py: A4 less 2.4 cm of margin either side.
        column_cm = 21.0 - 2 * 2.4
        assert plots.PAGE_WIDTH_IN == pytest.approx(column_cm / 2.54, abs=0.06)

    def test_the_style_saves_at_print_resolution(self) -> None:
        assert plots.STYLE["savefig.dpi"] >= 300
        assert plots.STYLE["savefig.bbox"] == "tight"
