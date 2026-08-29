"""Render the working paper to PDF.

    python paper/build_paper.py                 # -> paper/lifecycle_asset_allocation.pdf
    python paper/build_paper.py --out other.pdf

Two passes are used so that the table of contents can carry real page numbers.
Every quoted number is resolved from ``results/tables`` at build time (see
``paper/facts.py``), so the paper cannot drift away from the pipeline that
produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors                                   # noqa: E402
from reportlab.lib.units import cm                                 # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame,  # noqa: E402
                                KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph,
                                Spacer, Table)
from reportlab.platypus.tableofcontents import TableOfContents     # noqa: E402

import style as st                                                 # noqa: E402
from facts import Facts                                            # noqa: E402

SHORT_TITLE = "Beyond the Status Quo: A Computational Re-Examination"


class PaperDoc(BaseDocTemplate):
    """A document with a plain title page and a running-head body page."""

    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(filename, pagesize=st.PAGE_SIZE,
                         leftMargin=st.MARGIN_LEFT, rightMargin=st.MARGIN_RIGHT,
                         topMargin=st.MARGIN_TOP, bottomMargin=st.MARGIN_BOTTOM,
                         title=kwargs.pop("doc_title", SHORT_TITLE),
                         author=kwargs.pop("doc_author", ""),
                         subject=kwargs.pop("doc_subject", ""), **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width,
                      self.height, id="text", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="title", frames=[frame], onPage=self._title_page),
            PageTemplate(id="body", frames=[frame], onPage=self._body_page),
        ])
        self.styles: Dict[str, Any] = {}

    # -- page furniture ---------------------------------------------------
    def _title_page(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(st.SOFT_RULE)
        canvas.setLineWidth(0.5)
        y = doc.bottomMargin - 0.7 * cm
        canvas.line(doc.leftMargin, y, doc.leftMargin + doc.width, y)
        canvas.setFont(self.styles["running"].fontName, 7.6)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(doc.leftMargin + doc.width / 2.0,
                                 y - 0.42 * cm,
                                 "Every figure in this document is computed "
                                 "from the sources described in Section 3.")
        canvas.restoreState()

    def _body_page(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(self.styles["running"].fontName, 7.6)
        canvas.setFillColor(colors.HexColor("#666666"))
        top = doc.bottomMargin + doc.height + 0.55 * cm
        canvas.drawString(doc.leftMargin, top, SHORT_TITLE)
        canvas.drawRightString(doc.leftMargin + doc.width, top,
                               f"{canvas.getPageNumber()}")
        canvas.setStrokeColor(st.SOFT_RULE)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, top - 0.18 * cm,
                    doc.leftMargin + doc.width, top - 0.18 * cm)
        canvas.restoreState()

    # -- table of contents ------------------------------------------------
    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        name = getattr(flowable, "style", None)
        if name is None:
            return
        level = {"h1": 0, "h2": 1}.get(name.name)
        if level is None:
            return
        text = flowable.getPlainText()
        self.notify("TOCEntry", (level, text, self.page))


class Context:
    """Everything the content module needs: styles, numbers, counters."""

    def __init__(self, facts: Facts, styles: Dict[str, Any]) -> None:
        self.f = facts
        self.s = styles
        self.width = st.FRAME_WIDTH
        self._figure_no = 0
        self._table_no = 0
        self._equation_no = 0
        self.figure_index: List[str] = []
        self.table_index: List[str] = []

    # -- text -------------------------------------------------------------
    def h1(self, text: str) -> Paragraph:
        return Paragraph(text, self.s["h1"])

    def h2(self, text: str) -> Paragraph:
        return Paragraph(text, self.s["h2"])

    def h3(self, text: str) -> Paragraph:
        return Paragraph(text, self.s["h3"])

    def p(self, text: str) -> Paragraph:
        return Paragraph(text, self.s["body"])

    def note(self, text: str) -> Paragraph:
        return Paragraph(text, self.s["note"])

    def quote(self, text: str) -> Paragraph:
        return Paragraph(text, self.s["quote"])

    def code(self, text: str) -> Paragraph:
        body = text.strip("\n").replace("&", "&amp;").replace("<", "&lt;") \
            .replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;")
        return Paragraph(body, self.s["code"])

    def bullets(self, items: Sequence[str]) -> List[Paragraph]:
        return [Paragraph(f"•&nbsp;&nbsp;{item}", self.s["bullet"])
                for item in items]

    def equation(self, text: str, tag: bool = True) -> Paragraph:
        if tag:
            self._equation_no += 1
            text = f"{text}&nbsp;&nbsp;&nbsp;&nbsp;({self._equation_no})"
        return Paragraph(text, self.s["equation"])

    def gap(self, height: float = 6.0) -> Spacer:
        return Spacer(1, height)

    # -- floats -----------------------------------------------------------
    def figure(self, name: str, caption: str, max_height: float = 9.6 * cm,
               width_scale: float = 1.0) -> List[Flowable]:
        self._figure_no += 1
        self.figure_index.append(f"Figure {self._figure_no}. {caption}")
        image = st.figure_flowable(self.f.figure(name),
                                   self.width * width_scale, max_height)
        head = Paragraph(f"Figure {self._figure_no}.", self.s["caption_head"])
        body = Paragraph(caption, self.s["caption"])
        return [KeepTogether([image, head, body])]

    def table(self, rows: Sequence[Sequence[str]], caption: str,
              note: str | None = None, **kwargs: Any) -> List[Flowable]:
        self._table_no += 1
        self.table_index.append(f"Table {self._table_no}. {caption}")
        head = Paragraph(f"Table {self._table_no}. {caption}",
                         self.s["caption_head"])
        table = st.make_table(rows, self.s, total_width=self.width, **kwargs)
        parts: List[Flowable] = [head, self.gap(2), table]
        if note:
            parts.append(Paragraph(f"<i>Notes.</i> {note}", self.s["note"]))
        else:
            parts.append(self.gap(10))
        return [KeepTogether(parts)] if len(rows) <= 14 else parts

    def rule(self) -> Flowable:
        return st.HorizontalRule(self.width)


def _walk(flowables: Sequence[Any]) -> List[Any]:
    """Flatten nested KeepTogether/Table containers into their leaf flowables."""
    out: List[Any] = []
    for item in flowables:
        out.append(item)
        for attr in ("_content", "_cellvalues"):
            nested = getattr(item, attr, None)
            if isinstance(nested, (list, tuple)):
                flat = [x for row in nested
                        for x in (row if isinstance(row, (list, tuple)) else [row])]
                out.extend(_walk(flat))
    return out


def build(out_path: str, config_path: str = "config.yaml") -> Path:
    fonts = st.register_fonts()
    styles = st.build_styles(fonts)
    facts = Facts(config_path=config_path)
    ctx = Context(facts, styles)

    import content
    story = content.story(ctx)

    prose = "".join(flowable.getPlainText() for flowable in _walk(story)
                    if isinstance(flowable, Paragraph))
    missing = st.missing_glyphs(prose, fonts["Serif"])
    if missing:
        print(f"warning: the text face cannot render {missing!r}; these would "
              "be drawn as black boxes", file=sys.stderr)

    doc = PaperDoc(out_path,
                   doc_title="Beyond the Status Quo: A Computational "
                             "Re-Examination of Lifecycle Asset Allocation",
                   doc_author="StockChartIR replication project",
                   doc_subject="Lifecycle asset allocation, block bootstrap, "
                               "certainty equivalent consumption")
    doc.styles = styles
    doc.multiBuild(story)
    return Path(out_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(Path(__file__).resolve().parent
                                    / "lifecycle_asset_allocation.pdf"))
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    path = build(args.out, args.config)
    size = path.stat().st_size / 1024.0
    print(f"wrote {path} ({size:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
