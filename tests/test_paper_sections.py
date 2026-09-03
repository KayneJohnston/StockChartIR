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


class TestExtensionGroups:
    """The abstract announces a count and then describes that many studies.

    It has gone stale before. Deriving the counts from the reading order only
    helps if the groups and the order cannot drift apart, which is what these
    check.
    """

    def test_the_groups_partition_the_extensions(self) -> None:
        flat = [key for _, members in content.EXTENSION_GROUPS
                for key in members]
        assert sorted(flat) == sorted(content.EXTENSION_SECTIONS)

    def test_no_study_is_in_two_groups(self) -> None:
        flat = [key for _, members in content.EXTENSION_GROUPS
                for key in members]
        assert len(flat) == len(set(flat))

    def test_each_group_is_contiguous_in_the_reading_order(self) -> None:
        # A group the abstract introduces as "four ask whether..." has to be
        # four consecutive sections, or the prose walks the paper out of
        # order.
        for name, members in content.EXTENSION_GROUPS:
            numbers = [content.section_number(k) for k in members]
            assert numbers == list(range(min(numbers), max(numbers) + 1)), name

    def test_the_groups_run_in_reading_order(self) -> None:
        firsts = [content.section_number(members[0])
                  for _, members in content.EXTENSION_GROUPS]
        assert firsts == sorted(firsts)

    def test_the_count_word_matches_the_membership(self) -> None:
        for name, members in content.EXTENSION_GROUPS:
            word = content.group_count_word(name)
            assert content.NUMBER_WORDS.get(len(members), str(len(members))) \
                == word

    def test_an_unknown_group_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            content.group_count_word("nonexistent")

    def test_every_new_study_has_a_section(self) -> None:
        for key in ("cohorts", "out_of_sample", "human_capital", "mortality"):
            assert key in content.SECTION_ORDER
            assert hasattr(content, f"section_{key}")


class TestNewSectionsAreWiredIn:
    """The two sections added to answer reviewer objections."""

    def test_both_appear_in_the_reading_order(self) -> None:
        assert "pension" in content.SECTION_ORDER
        assert "turnover" in content.SECTION_ORDER

    def test_pension_sits_with_the_robustness_studies(self) -> None:
        """It relaxes a modelling assumption, so it belongs before the
        portfolio movement rather than among the searches."""
        order = list(content.SECTION_ORDER)
        assert order.index("mortality") < order.index("pension")
        assert order.index("pension") < order.index("glide")

    def test_turnover_sits_with_the_other_audit(self) -> None:
        """Costs and unseen data are the same question asked twice."""
        order = list(content.SECTION_ORDER)
        assert order.index("leverage") < order.index("turnover")
        assert order.index("turnover") < order.index("out_of_sample")

    def test_the_groups_still_partition_the_extensions(self) -> None:
        covered = [k for _, members in content.EXTENSION_GROUPS
                   for k in members]
        assert sorted(covered) == sorted(content.EXTENSION_SECTIONS)
        assert len(covered) == len(set(covered))

    def test_the_counts_the_abstract_quotes_are_derived(self) -> None:
        for name, members in content.EXTENSION_GROUPS:
            assert content.group_count_word(name) == \
                content.NUMBER_WORDS[len(members)]
        assert content.extension_count_word() == \
            content.NUMBER_WORDS[len(content.EXTENSION_SECTIONS)]


class TestLimitationsIsNotStale:
    """A Limitations section that contradicts the paper is worse than none.

    Three bullets survived past the sections that answered them, which is the
    single most damaging thing a careful reader can find. These pin the
    wording so it cannot drift back.
    """

    @staticmethod
    def _source() -> str:
        import inspect
        return inspect.getsource(content.section_limitations)

    def test_does_not_claim_the_paper_omits_fees(self) -> None:
        assert "<b>No fees.</b>" not in self._source()

    def test_does_not_claim_mortality_is_deterministic(self) -> None:
        assert "<b>Deterministic mortality.</b>" not in self._source()

    def test_does_not_call_housing_the_single_largest_omission(self) -> None:
        assert "single largest omission" not in self._source()

    def test_names_the_sections_that_answer_each_omission(self) -> None:
        source = self._source()
        for key in ("#fees", "#mortality", "#housing", "#pension",
                    "#cohorts", "#out_of_sample", "#turnover",
                    "#withholding"):
            assert key in source, f"limitations never mentions {key}"


class TestPensionSectionIsDerived:
    """No number in the pension section may be typed rather than computed.

    The section quotes two replacement rates against each other, and the
    comparison only means anything if both come out of the specs the sweep
    actually ran. An earlier draft hardcoded one of them.
    """

    @staticmethod
    def _source() -> str:
        import inspect
        return inspect.getsource(content.section_pension)

    def test_no_hardcoded_replacement_rate(self) -> None:
        source = self._source()
        for literal in ("0.442", "44.2%", "0.293", "29.3%"):
            assert literal not in source, \
                f"{literal!r} is typed into the pension section"

    def test_the_us_rate_is_evaluated_from_the_spec(self) -> None:
        assert "social_security_benefit" in self._source()

    def test_the_australian_rate_comes_from_the_config(self) -> None:
        assert "pension_full_rate" in self._source()


class TestInflationSectionIsWiredIn:
    """The companion to the valuation study, and placed as one."""

    def test_appears_in_the_reading_order(self) -> None:
        assert "inflation" in content.SECTION_ORDER

    def test_sits_beside_the_study_it_mirrors(self) -> None:
        """Both condition a lifetime on a state variable observable at its
        start, so they belong next to each other rather than pages apart."""
        order = list(content.SECTION_ORDER)
        assert order.index("inflation") == order.index("valuation") + 1

    def test_is_counted_among_the_robustness_studies(self) -> None:
        groups = dict(content.EXTENSION_GROUPS)
        assert "inflation" in groups["robustness"]

    def test_the_groups_still_partition_the_extensions(self) -> None:
        covered = [k for _, members in content.EXTENSION_GROUPS
                   for k in members]
        assert sorted(covered) == sorted(content.EXTENSION_SECTIONS)
        assert len(covered) == len(set(covered))

    def test_no_hardcoded_correlation_or_gap(self) -> None:
        """Every number in the section comes from the pipeline's own tables."""
        import inspect
        source = inspect.getsource(content.section_inflation)
        for literal in ("0.58", "-5.19", "-2.32", "+3.54", "10%", "100%"):
            assert f'"{literal}"' not in source, \
                f"{literal!r} is typed into the inflation section"


class TestWithholdingSectionIsWiredIn:
    """The concrete instance of the fee experiment, placed beside it."""

    def test_appears_in_the_reading_order(self) -> None:
        assert "withholding" in content.SECTION_ORDER

    def test_follows_the_fee_study_it_sharpens(self) -> None:
        order = list(content.SECTION_ORDER)
        assert order.index("withholding") == order.index("fees") + 1

    def test_is_counted_among_the_robustness_studies(self) -> None:
        assert "withholding" in dict(content.EXTENSION_GROUPS)["robustness"]

    def test_the_groups_still_partition_the_extensions(self) -> None:
        covered = [k for _, members in content.EXTENSION_GROUPS
                   for k in members]
        assert sorted(covered) == sorted(content.EXTENSION_SECTIONS)
        assert len(covered) == len(set(covered))

    def test_the_fee_break_even_is_read_not_typed(self) -> None:
        """The section's whole point is a comparison with Section #fees'
        break-even, so that number has to come from the fee module."""
        import inspect
        source = inspect.getsource(content.section_withholding)
        assert "break_even_differential_bp" in source
        for literal in ("114", "115", "29.2", "112"):
            assert f'"{literal}"' not in source
