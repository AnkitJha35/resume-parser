from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.pipeline.stages.text_extraction import TextBlock


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TextStyle:
    font_size: float | None = None
    bold: bool | None = None
    italic: bool | None = None
    font_name: str | None = None


@dataclass(frozen=True)
class Span:
    span_id: str
    text: str
    bbox: BoundingBox
    font_size: float | None = None
    bold: bool | None = None
    link: str | None = None


@dataclass(frozen=True)
class Line:
    line_id: str
    page_number: int
    bbox: BoundingBox
    spans: list[Span]
    text: str
    style: TextStyle
    reading_order: int | None = None
    source_span_ids: list[str] = field(default_factory=list)
    reconstruction_method: str = "physical"


@dataclass
class Region:
    region_id: str
    kind: str
    bbox: BoundingBox
    lines: list[Line] = field(default_factory=list)
    reading_order: int | None = None
    column_id: int | None = None


@dataclass
class Page:
    page_number: int
    width: float | None = None
    height: float | None = None
    regions: list[Region] = field(default_factory=list)
    tables: list[object] = field(default_factory=list)
    headers: list[Line] = field(default_factory=list)
    footers: list[Line] = field(default_factory=list)
    drawings: list[Any] = field(default_factory=list)


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)
    document_type: str = "unknown"
    extraction_mode: str = "text"
    ocr_used: bool = False


def document_from_text_blocks(
    blocks: Iterable[TextBlock],
    drawings: Iterable[Any] | None = None,
) -> Document:
    """Build a physical-layout document view without resolving reading order."""
    if drawings is None:
        try:
            from app.pipeline.stages.text_extraction import PDFExtractor
            drawings = getattr(PDFExtractor, "_last_drawings", None) or []
        except ImportError:
            drawings = []

    drawings_by_page: dict[int, list[Any]] = {}
    for d in drawings:
        p_num = getattr(d, "page_number", 1)
        drawings_by_page.setdefault(p_num, []).append(d)

    page_blocks: dict[int, list[TextBlock]] = {}
    for block in blocks:
        page_blocks.setdefault(block.page_number, []).append(block)

    pages: list[Page] = []
    for page_number in sorted(page_blocks):
        source_blocks = page_blocks[page_number]
        lines: list[Line] = []
        for block_index, block in enumerate(source_blocks):
            line_id = f"page-{page_number}-region-0-line-{block_index}"
            span_id = f"{line_id}-span-0"
            bbox = BoundingBox(block.x0, block.y0, block.x1, block.y1)
            span = Span(
                span_id=span_id,
                text=block.text,
                bbox=bbox,
                font_size=block.font_size,
                bold=block.bold,
            )
            lines.append(
                Line(
                    line_id=line_id,
                    page_number=page_number,
                    bbox=bbox,
                    spans=[span],
                    text=block.text,
                    style=TextStyle(font_size=block.font_size, bold=block.bold),
                    reading_order=block_index,
                    source_span_ids=[span_id],
                )
            )

        x0 = min(block.x0 for block in source_blocks)
        y0 = min(block.y0 for block in source_blocks)
        x1 = max(block.x1 for block in source_blocks)
        y1 = max(block.y1 for block in source_blocks)
        pages.append(
            Page(
                page_number=page_number,
                regions=[
                    Region(
                        region_id=f"page-{page_number}-region-0",
                        kind="physical_page",
                        bbox=BoundingBox(x0, y0, x1, y1),
                        lines=lines,
                    )
                ],
                drawings=drawings_by_page.get(page_number, []),
            )
        )

    return Document(pages=pages)
