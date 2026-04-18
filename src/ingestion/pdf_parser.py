# src/ingestion/pdf_parser.py
"""
PyMuPDF-based parser that extracts clean text from lecture PDFs.
Handles multi-column layouts, slide headers, and table detection
which pypdf silently mangles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # pymupdf


@dataclass
class ParsedPage:
    """Represents one parsed PDF page with metadata."""
    page_num: int
    text: str
    has_table: bool = False
    has_image_caption: bool = False
    word_count: int = 0


@dataclass
class ParsedDocument:
    """Full parsed document ready for chunking."""
    file_path: str
    file_name: str
    lecture_name: str
    pages: list[ParsedPage] = field(default_factory=list)
    full_text: str = ""
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)


def _clean_text(raw: str) -> str:
    """
    Normalize extracted PDF text.
    - Collapse excessive whitespace/newlines
    - Remove page artifacts (lone numbers, headers repeated every page)
    - Preserve paragraph breaks as double newlines
    """
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", raw)
    # Collapse 3+ newlines to exactly 2 (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone single-digit or dual-digit lines (page numbers)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def _detect_table(page: fitz.Page) -> bool:
    """
    Heuristic table detection — checks if page has
    grid-like structure (many short lines in tabular positions).
    """
    blocks = page.get_text("blocks")
    if len(blocks) < 4:
        return False

    # If more than 40% of blocks are very short (< 30 chars), likely a table
    short_blocks = sum(1 for b in blocks if len(b[4].strip()) < 30)
    return (short_blocks / len(blocks)) > 0.4


def _extract_page(page: fitz.Page, page_num: int) -> ParsedPage:
    """
    Extract a single page using layout-preserving mode.
    'blocks' mode respects reading order better than raw text for slides.
    """
    # dict mode gives us structured blocks — better for columns
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)

    page_text_parts = []
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # type 0 = text block
            for line in block.get("lines", []):
                line_text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                if line_text.strip():
                    page_text_parts.append(line_text.strip())

    raw_text = "\n".join(page_text_parts)
    clean = _clean_text(raw_text)

    has_table = _detect_table(page)
    has_caption = bool(
        re.search(r"(figure|fig\.|table|exhibit)\s*\d", clean, re.IGNORECASE)
    )

    return ParsedPage(
        page_num=page_num,
        text=clean,
        has_table=has_table,
        has_image_caption=has_caption,
        word_count=len(clean.split()),
    )


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Main entry point. Parse a PDF file into a ParsedDocument.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with per-page content and full concatenated text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF has no extractable text (likely scanned).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    lecture_name = path.stem.replace("_", " ").replace("-", " ")

    doc_obj = fitz.open(str(path))
    parsed_pages: list[ParsedPage] = []

    for page_num in range(len(doc_obj)):
        page = doc_obj[page_num]
        parsed_page = _extract_page(page, page_num + 1)
        # Skip near-empty pages (cover slides, blank pages)
        if parsed_page.word_count > 10:
            parsed_pages.append(parsed_page)

    doc_obj.close()

    if not parsed_pages:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "This may be a scanned PDF — OCR is required."
        )

    # Concatenate all pages with clear page boundary markers
    full_text = "\n\n".join(
        f"[Page {p.page_num}]\n{p.text}" for p in parsed_pages
    )

    table_pages = sum(1 for p in parsed_pages if p.has_table)

    return ParsedDocument(
        file_path=str(path),
        file_name=path.name,
        lecture_name=lecture_name,
        pages=parsed_pages,
        full_text=full_text,
        total_pages=len(parsed_pages),
        metadata={
            "source": str(path),
            "file_name": path.name,
            "lecture": lecture_name,
            "total_pages": len(parsed_pages),
            "table_pages": table_pages,
        },
    )


def parse_txt(file_path: str | Path) -> ParsedDocument:
    """Parse a plain .txt file into the same ParsedDocument format."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    lecture_name = path.stem.replace("_", " ").replace("-", " ")
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    clean = _clean_text(raw_text)

    # Treat each 400-word block as a "page" for consistency
    words = clean.split()
    pages = []
    for i in range(0, len(words), 400):
        chunk_text = " ".join(words[i : i + 400])
        pages.append(ParsedPage(
            page_num=len(pages) + 1,
            text=chunk_text,
            word_count=len(chunk_text.split()),
        ))

    return ParsedDocument(
        file_path=str(path),
        file_name=path.name,
        lecture_name=lecture_name,
        pages=pages,
        full_text=clean,
        total_pages=len(pages),
        metadata={
            "source": str(path),
            "file_name": path.name,
            "lecture": lecture_name,
            "total_pages": len(pages),
        },
    )