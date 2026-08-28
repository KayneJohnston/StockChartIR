"""Typography, page furniture and flowable helpers for the working paper.

The paper is rendered with ReportLab rather than LaTeX because the container
has no TeX distribution. The layout below is a single-column working-paper
style -- generous margins, a serif text face, small-caps-ish section heads --
which reads closer to an NBER or SSRN working paper than to a two-column
journal article, and which survives the large tables this project produces.

Fonts are registered from the system Liberation family (Times-metric) so that
Greek and mathematical characters render as glyphs rather than as the black
boxes ReportLab's built-in Type 1 faces produce for anything outside Latin-1.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Flowable, Image, KeepTogether, PageBreak,
                                Paragraph, Spacer, Table, TableStyle)

PAGE_SIZE = A4
MARGIN_LEFT = 2.4 * cm
MARGIN_RIGHT = 2.4 * cm
MARGIN_TOP = 2.5 * cm
MARGIN_BOTTOM = 2.4 * cm
FRAME_WIDTH = PAGE_SIZE[0] - MARGIN_LEFT - MARGIN_RIGHT

#: Matches ``src/plots.py`` so figures and text agree on their accents.
ACCENT = colors.HexColor("#0072B2")
RULE = colors.HexColor("#2b2b2b")
SOFT_RULE = colors.HexColor("#9a9a9a")
BAND = colors.HexColor("#f2f4f7")

_LIBERATION = "/usr/share/fonts/truetype/liberation"
_DEJAVU = "/usr/share/fonts/truetype/dejavu"


def register_fonts() -> Dict[str, str]:
    """Register the Times-metric text face and a monospace face.

    Returns the family names to use, falling back to ReportLab's built-ins if
    the system fonts are missing so that a build never fails outright over
    typography.
    """
    faces = {
        "Serif": ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
                  "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"),
        "Sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
                 "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"),
    }
    registered: Dict[str, str] = {}
    for family, files in faces.items():
        paths = [os.path.join(_LIBERATION, f) for f in files]
        if not all(os.path.exists(p) for p in paths):
            registered[family] = "Times-Roman" if family == "Serif" else "Helvetica"
            continue
        for suffix, path in zip(("", "-Bold", "-Italic", "-BoldItalic"), paths):
            pdfmetrics.registerFont(TTFont(f"Paper{family}{suffix}", path))
        pdfmetrics.registerFontFamily(
            f"Paper{family}", normal=f"Paper{family}",
            bold=f"Paper{family}-Bold", italic=f"Paper{family}-Italic",
            boldItalic=f"Paper{family}-BoldItalic")
        registered[family] = f"Paper{family}"

    mono_path = os.path.join(_DEJAVU, "DejaVuSansMono.ttf")
    mono_bold = os.path.join(_DEJAVU, "DejaVuSansMono-Bold.ttf")
    if os.path.exists(mono_path):
        pdfmetrics.registerFont(TTFont("PaperMono", mono_path))
        if os.path.exists(mono_bold):
            pdfmetrics.registerFont(TTFont("PaperMono-Bold", mono_bold))
            pdfmetrics.registerFontFamily("PaperMono", normal="PaperMono",
                                          bold="PaperMono-Bold",
                                          italic="PaperMono",
                                          boldItalic="PaperMono-Bold")
        registered["Mono"] = "PaperMono"
    else:
        registered["Mono"] = "Courier"
    return registered


def build_styles(fonts: Dict[str, str]) -> Dict[str, ParagraphStyle]:
    """The paragraph styles the paper uses, keyed by role."""
    serif, sans, mono = fonts["Serif"], fonts["Sans"], fonts["Mono"]
    base = getSampleStyleSheet()["Normal"]
    S: Dict[str, ParagraphStyle] = {}

    S["title"] = ParagraphStyle(
        "title", parent=base, fontName=f"{serif}-Bold", fontSize=20,
        leading=25, alignment=TA_CENTER, spaceAfter=6)
    S["subtitle"] = ParagraphStyle(
        "subtitle", parent=base, fontName=f"{serif}-Italic", fontSize=13,
        leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
        spaceAfter=18)
    S["author"] = ParagraphStyle(
        "author", parent=base, fontName=serif, fontSize=11.5, leading=15,
        alignment=TA_CENTER, spaceAfter=3)
    S["date"] = ParagraphStyle(
        "date", parent=base, fontName=serif, fontSize=10.5, leading=14,
        alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
    S["abstract_head"] = ParagraphStyle(
        "abstract_head", parent=base, fontName=f"{serif}-Bold", fontSize=11,
        leading=14, alignment=TA_CENTER, spaceBefore=6, spaceAfter=8)
    S["abstract"] = ParagraphStyle(
        "abstract", parent=base, fontName=serif, fontSize=9.8, leading=13.6,
        alignment=TA_JUSTIFY, leftIndent=14, rightIndent=14, spaceAfter=6)
    S["keywords"] = ParagraphStyle(
        "keywords", parent=base, fontName=serif, fontSize=9.2, leading=12.6,
        alignment=TA_JUSTIFY, leftIndent=14, rightIndent=14, spaceBefore=6)

    S["h1"] = ParagraphStyle(
        "h1", parent=base, fontName=f"{serif}-Bold", fontSize=14, leading=18,
        spaceBefore=20, spaceAfter=8, keepWithNext=1)
    S["h2"] = ParagraphStyle(
        "h2", parent=base, fontName=f"{serif}-Bold", fontSize=11.6,
        leading=15, spaceBefore=14, spaceAfter=5, keepWithNext=1)
    S["h3"] = ParagraphStyle(
        "h3", parent=base, fontName=f"{serif}-BoldItalic", fontSize=10.6,
        leading=14, spaceBefore=11, spaceAfter=4, keepWithNext=1)

    S["body"] = ParagraphStyle(
        "body", parent=base, fontName=serif, fontSize=10.2, leading=14.6,
        alignment=TA_JUSTIFY, spaceAfter=7)
    S["body_first"] = ParagraphStyle("body_first", parent=S["body"])
    S["bullet"] = ParagraphStyle(
        "bullet", parent=S["body"], leftIndent=16, bulletIndent=4,
        spaceAfter=4)
    S["quote"] = ParagraphStyle(
        "quote", parent=S["body"], leftIndent=20, rightIndent=20,
        fontName=f"{serif}-Italic", spaceBefore=4, spaceAfter=8)
    S["equation"] = ParagraphStyle(
        "equation", parent=base, fontName=f"{serif}-Italic", fontSize=10.8,
        leading=16, alignment=TA_CENTER, spaceBefore=8, spaceAfter=8)
    S["caption"] = ParagraphStyle(
        "caption", parent=base, fontName=sans, fontSize=8.5, leading=11.4,
        alignment=TA_JUSTIFY, textColor=colors.HexColor("#242424"),
        spaceBefore=5, spaceAfter=12)
    S["caption_head"] = ParagraphStyle(
        "caption_head", parent=S["caption"], fontName=f"{sans}-Bold",
        spaceAfter=2, spaceBefore=5)
    S["table_head"] = ParagraphStyle(
        "table_head", parent=base, fontName=f"{sans}-Bold", fontSize=7.7,
        leading=9.6, alignment=TA_CENTER, textColor=colors.black)
    S["table_cell"] = ParagraphStyle(
        "table_cell", parent=base, fontName=sans, fontSize=7.7, leading=9.6,
        alignment=TA_CENTER)
    S["table_cell_left"] = ParagraphStyle(
        "table_cell_left", parent=S["table_cell"], alignment=TA_LEFT)
    S["note"] = ParagraphStyle(
        "note", parent=base, fontName=serif, fontSize=8.6, leading=11.6,
        alignment=TA_JUSTIFY, textColor=colors.HexColor("#333333"),
        spaceBefore=3, spaceAfter=10)
    S["code"] = ParagraphStyle(
        "code", parent=base, fontName=mono, fontSize=8.2, leading=11.4,
        leftIndent=12, spaceBefore=4, spaceAfter=8,
        textColor=colors.HexColor("#1a1a1a"))
    S["reference"] = ParagraphStyle(
        "reference", parent=base, fontName=serif, fontSize=9.4, leading=12.6,
        alignment=TA_JUSTIFY, leftIndent=18, firstLineIndent=-18,
        spaceAfter=5)
    # Front-matter headings that must look like an h1 without being picked up
    # as a table-of-contents entry.
    S["h1_plain"] = ParagraphStyle("h1_plain", parent=S["h1"])
    S["toc1"] = ParagraphStyle(
        "toc1", parent=base, fontName=f"{serif}-Bold", fontSize=10.2,
        leading=15, spaceBefore=5)
    S["toc2"] = ParagraphStyle(
        "toc2", parent=base, fontName=serif, fontSize=9.6, leading=13,
        leftIndent=16)
    S["running"] = ParagraphStyle(
        "running", parent=base, fontName=sans, fontSize=7.6, leading=9,
        textColor=colors.HexColor("#666666"))
    return S


class HorizontalRule(Flowable):
    """A plain rule used to separate the front matter from the body."""

    def __init__(self, width: float, thickness: float = 0.6,
                 colour: colors.Color = RULE, space: float = 5.0) -> None:
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.colour = colour
        self.space = space
        self.height = thickness + space

    def draw(self) -> None:
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space, self.width, self.space)


def missing_glyphs(text: str, font_name: str) -> str:
    """Characters in ``text`` that the registered font cannot render.

    ReportLab draws an unmapped codepoint as a black box with no warning, and
    a paper full of Greek and mathematical notation makes that easy to ship by
    accident. The build calls this over every string it is about to typeset.
    """
    try:
        face = pdfmetrics.getFont(font_name).face
        cmap = getattr(face, "charToGlyph", None)
    except Exception:                                  # built-in Type 1 face
        return ""
    if not cmap:
        return ""
    return "".join(sorted({c for c in text
                           if ord(c) > 127 and ord(c) not in cmap}))


def figure_flowable(path: str, max_width: float, max_height: float = 15.0 * cm
                    ) -> Image:
    """Scale a PNG to the text column, preserving aspect ratio."""
    from reportlab.lib.utils import ImageReader
    reader = ImageReader(path)
    native_w, native_h = reader.getSize()
    scale = min(max_width / native_w, max_height / native_h)
    return Image(path, width=native_w * scale, height=native_h * scale)


def make_table(rows: Sequence[Sequence[str]], styles: Dict[str, ParagraphStyle],
               col_widths: Sequence[float] | None = None,
               left_align_first: bool = True,
               font_size: float | None = None,
               highlight_rows: Iterable[int] = (),
               total_width: float = FRAME_WIDTH) -> Table:
    """A ruled academic table: header rule, single body, no vertical lines.

    ``rows[0]`` is the header. Cells are wrapped in Paragraphs so that long
    labels break rather than overflow the column, which matters because many
    of this project's row labels are full strategy names.
    """
    head_style = styles["table_head"]
    cell = styles["table_cell"]
    cell_left = styles["table_cell_left"]
    if font_size is not None:
        head_style = ParagraphStyle("h", parent=head_style, fontSize=font_size,
                                    leading=font_size * 1.25)
        cell = ParagraphStyle("c", parent=cell, fontSize=font_size,
                              leading=font_size * 1.25)
        cell_left = ParagraphStyle("cl", parent=cell_left, fontSize=font_size,
                                   leading=font_size * 1.25)

    data: List[List[Any]] = []
    for r, row in enumerate(rows):
        out: List[Any] = []
        for c, value in enumerate(row):
            text = "" if value is None else str(value)
            if r == 0:
                out.append(Paragraph(text, head_style))
            elif c == 0 and left_align_first:
                out.append(Paragraph(text, cell_left))
            else:
                out.append(Paragraph(text, cell))
        data.append(out)

    if col_widths is None:
        n = len(rows[0])
        first = total_width * (0.30 if left_align_first and n > 3 else 1.0 / n)
        rest = (total_width - first) / max(n - 1, 1)
        col_widths = [first] + [rest] * (n - 1)

    style = [
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.0),
    ]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), BAND))
    table = Table(data, colWidths=list(col_widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle(style))
    return table
