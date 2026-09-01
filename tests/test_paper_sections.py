"""Tests for the paper's section numbering.

Numbering used to be written by hand in ninety-odd places, and twice in this
project's history a cross-reference survived a renumber while quietly pointing
at the wrong heading -- a failure the PDF-level reference check cannot see,
because the reference still resolves to *a* section. The `#key` tokens remove
the failure mode by construction; these tests keep it removed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
sys.path.insert(0, str(PAPER))

content = pytest.importorskip("content")

_RAW = (PAPER / "content.py").read_text()

#: The registry block documents the token syntax with literal ``#key``
#: examples, so the scans below start after it.
SOURCE = _RAW[_RAW.index("return SECTION_TOKEN.sub(swap, text)"):]

#: Adjacent string literals are concatenated by the parser, so a reference
#: split over two source lines is invisible until they are joined.
FLAT = re.sub(r'"\s*\n\s*(f?)"', "", SOURCE)


class TestTheRegistry:
    def test_every_key_is_unique(self) -> None:
        assert len(set(content.SECTION_ORDER)) == len(content.SECTION_ORDER)

    def test_numbers_run_from_one_without_gaps(self) -> None:
        numbers = [content.section_number(k) for k in content.SECTION_ORDER]
        assert numbers == list(range(1, len(content.SECTION_ORDER) + 1))

    def test_an_unknown_key_names_the_valid_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown section"):
            content.section_number("nope")


class TestTokenResolution:
    def test_a_bare_token_becomes_the_section_number(self) -> None:
        n = content.section_number("housing")
        assert content.resolve_sections("Section #housing") == f"Section {n}"

    def test_a_subsection_token_keeps_its_suffix(self) -> None:
        n = content.section_number("leverage")
        assert content.resolve_sections("#leverage.4.1") == f"{n}.4.1"

    def test_a_heading_dot_is_not_swallowed(self) -> None:
        # "#glide. Solving for..." must not read the ". S" as a subsection.
        n = content.section_number("glide")
        assert content.resolve_sections("#glide. Solving") == f"{n}. Solving"

    def test_resolution_is_idempotent_on_resolved_text(self) -> None:
        once = content.resolve_sections("Section #data.6.2 and Section #methods")
        assert content.resolve_sections(once) == once


class TestTheSourceUsesTokensThroughout:
    def test_no_heading_carries_a_hand_written_number(self) -> None:
        literal = re.findall(r'ctx\.h[123]\(\s*f?"(\d)', SOURCE)
        assert not literal, (
            f"{len(literal)} heading(s) still numbered by hand; use a "
            "#key token so the number follows SECTION_ORDER")

    def test_no_cross_reference_carries_a_hand_written_number(self) -> None:
        literal = sorted(set(re.findall(r"Sections? \d[\d.]*", FLAT)))
        assert not literal, (
            f"hand-written cross-references remain: {literal}")

    def test_every_token_names_a_real_section(self) -> None:
        used = set(re.findall(r"#([a-z_]+)", FLAT))
        unknown = used - set(content.SECTION_ORDER)
        assert not unknown, f"tokens with no section: {sorted(unknown)}"

    def test_every_section_is_referred_to_or_at_least_titled(self) -> None:
        used = set(re.findall(r"#([a-z_]+)", FLAT))
        missing = set(content.SECTION_ORDER) - used
        assert not missing, (
            f"sections with no heading token, so they cannot be numbered: "
            f"{sorted(missing)}")


class TestTheOrderGuard:
    @staticmethod
    def _heading(text: str):
        style = type("S", (), {"name": "h1"})()
        return type("P", (), {"text": text, "style": style})()

    def test_a_correct_sequence_passes(self) -> None:
        parts = [self._heading(f"{i}. Title")
                 for i in range(1, len(content.SECTION_ORDER) + 1)]
        content._check_section_order(parts)          # must not raise

    def test_an_out_of_order_sequence_fails(self) -> None:
        parts = [self._heading("1. A"), self._heading("3. B"),
                 self._heading("2. C")]
        with pytest.raises(AssertionError, match="does not match"):
            content._check_section_order(parts)

    def test_a_missing_section_fails(self) -> None:
        parts = [self._heading(f"{i}. Title")
                 for i in range(1, len(content.SECTION_ORDER))]
        with pytest.raises(AssertionError, match="does not match"):
            content._check_section_order(parts)

    def test_appendix_headings_are_ignored(self) -> None:
        parts = [self._heading(f"{i}. Title")
                 for i in range(1, len(content.SECTION_ORDER) + 1)]
        parts.append(self._heading("Appendix A. Model Parameters"))
        content._check_section_order(parts)          # must not raise


class TestReadingOrder:
    """The groupings the roadmap promises the reader."""

    def test_the_headline_is_followed_by_its_robustness_checks(self) -> None:
        for key in ("sensitivity", "sleeve", "hedging"):
            assert content.section_number(key) \
                > content.section_number("baseline")
        # ...and they come before the searches for a better portfolio.
        assert content.section_number("hedging") \
            < content.section_number("glide")

    def test_the_portfolio_searches_relax_constraints_in_order(self) -> None:
        assert (content.section_number("glide")
                < content.section_number("allocation")
                < content.section_number("leverage"))

    def test_the_asset_is_priced_before_it_is_mortgaged(self) -> None:
        assert content.section_number("housing") \
            < content.section_number("mortgage")

    def test_the_non_portfolio_levers_run_in_lifecycle_order(self) -> None:
        # Where you start, what you save, what that responds to, when you
        # stop, how you draw down.
        order = ["valuation", "saving", "accumulation", "retirement",
                 "spending"]
        numbers = [content.section_number(k) for k in order]
        assert numbers == sorted(numbers)

    def test_the_closing_sections_come_last(self) -> None:
        assert list(content.SECTION_ORDER[-3:]) == [
            "discussion", "limitations", "conclusion"]
