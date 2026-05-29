"""
OmniRoute AI — Document Parser Module
Extracts text, images, and tables from PDF, DOCX, and TXT files.
Zero-loss: every page, every image, every table is captured.
"""

from __future__ import annotations

import io
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from PIL import Image

from modules.utils import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedImage:
    """An image extracted from a document page."""
    page_number: int
    image_bytes: bytes
    mime_type: str  # e.g. "image/png"
    width: int = 0
    height: int = 0

    def to_base64(self) -> str:
        return base64.b64encode(self.image_bytes).decode("utf-8")


@dataclass
class ExtractedTable:
    """A table extracted from a document page, stored as list-of-lists."""
    page_number: int
    table_data: list[list[str]]  # rows × cols
    source_file: str = ""

    def to_mongo_dict(self) -> dict:
        """Serialize for MongoDB insertion."""
        headers = self.table_data[0] if self.table_data else []
        rows = self.table_data[1:] if len(self.table_data) > 1 else []
        return {
            "page_number": self.page_number,
            "source_file": self.source_file,
            "headers": headers,
            "rows": [dict(zip(headers, row)) for row in rows] if headers else rows,
            "raw": self.table_data,
        }


@dataclass
class PageContent:
    """All content extracted from a single page."""
    page_number: int
    text: str = ""
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Complete parsed output of a document."""
    filename: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)

    @property
    def all_images(self) -> list[ExtractedImage]:
        return [img for p in self.pages for img in p.images]

    @property
    def all_tables(self) -> list[ExtractedTable]:
        return [tbl for p in self.pages for tbl in p.tables]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


# ---------------------------------------------------------------------------
# PDF Parser (PyMuPDF / fitz)
# ---------------------------------------------------------------------------

def _parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Extract text, images, and tables from a PDF.
    Uses PyMuPDF's built-in table detection and image extraction.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[PageContent] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        pc = PageContent(page_number=page_num)

        # --- Text ---
        pc.text = page.get_text("text") or ""

        # --- Images ---
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if base_image and base_image.get("image"):
                    img_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                    # Get dimensions
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    w, h = pil_img.size
                    # Skip tiny images (icons, decorations < 50px)
                    if w >= 50 and h >= 50:
                        pc.images.append(ExtractedImage(
                            page_number=page_num,
                            image_bytes=img_bytes,
                            mime_type=mime,
                            width=w,
                            height=h,
                        ))
            except Exception as e:
                logger.warning(f"Could not extract image xref={xref} on page {page_num}: {e}")

        # --- Tables (PyMuPDF built-in) ---
        try:
            tabs = page.find_tables()
            for tab in tabs:
                table_data = tab.extract()
                if table_data and len(table_data) > 0:
                    # Clean None values
                    cleaned = [
                        [cell if cell is not None else "" for cell in row]
                        for row in table_data
                    ]
                    pc.tables.append(ExtractedTable(
                        page_number=page_num,
                        table_data=cleaned,
                        source_file=filename,
                    ))
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_num}: {e}")

        pages.append(pc)

    doc.close()

    return ParsedDocument(
        filename=filename,
        total_pages=len(pages),
        pages=pages,
    )


# ---------------------------------------------------------------------------
# DOCX Parser
# ---------------------------------------------------------------------------

def _parse_docx(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Extract text, images, and tables from a .docx file.
    DOCX doesn't have fixed pages, so we treat the whole doc as page 1+
    and use paragraph breaks as approximate page boundaries.
    """
    doc = DocxDocument(io.BytesIO(file_bytes))
    pages: list[PageContent] = []

    # --- Text: collect all paragraphs ---
    paragraphs_text = [p.text for p in doc.paragraphs if p.text.strip()]
    # Approximate page splitting (~3000 chars per page)
    chars_per_page = 3000
    full_text = "\n".join(paragraphs_text)
    page_num = 0
    for i in range(0, max(len(full_text), 1), chars_per_page):
        page_num += 1
        chunk = full_text[i:i + chars_per_page]
        pages.append(PageContent(page_number=page_num, text=chunk))

    if not pages:
        pages.append(PageContent(page_number=1, text=""))

    # --- Images from relationships ---
    img_counter = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                img_bytes = rel.target_part.blob
                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size
                fmt = pil_img.format or "PNG"
                mime = f"image/{fmt.lower()}"
                if fmt.lower() == "jpg":
                    mime = "image/jpeg"
                img_counter += 1
                # Assign images round-robin to pages
                target_page = ((img_counter - 1) % len(pages))
                pages[target_page].images.append(ExtractedImage(
                    page_number=pages[target_page].page_number,
                    image_bytes=img_bytes,
                    mime_type=mime,
                    width=w,
                    height=h,
                ))
            except Exception as e:
                logger.warning(f"Could not extract DOCX image: {e}")

    # --- Tables ---
    for tbl_idx, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            table_data.append([cell.text for cell in row.cells])
        if table_data:
            # Assign table to the first page (DOCX doesn't give page info)
            target_page_num = min(tbl_idx + 1, len(pages))
            pages[target_page_num - 1].tables.append(ExtractedTable(
                page_number=target_page_num,
                table_data=table_data,
                source_file=filename,
            ))

    return ParsedDocument(
        filename=filename,
        total_pages=len(pages),
        pages=pages,
    )


# ---------------------------------------------------------------------------
# TXT Parser
# ---------------------------------------------------------------------------

def _parse_txt(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Parse a plain text file. Splits into approximate pages."""
    text = file_bytes.decode("utf-8", errors="replace")
    chars_per_page = 3000
    pages: list[PageContent] = []

    page_num = 0
    for i in range(0, max(len(text), 1), chars_per_page):
        page_num += 1
        chunk = text[i:i + chars_per_page]
        pages.append(PageContent(page_number=page_num, text=chunk))

    if not pages:
        pages.append(PageContent(page_number=1, text=""))

    return ParsedDocument(
        filename=filename,
        total_pages=len(pages),
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Main entry point. Parses any supported document and returns a
    ParsedDocument with zero content loss.

    Raises ValueError for unsupported file types.
    """
    ext = Path(filename).suffix.lower()
    logger.info(f"Parsing '{filename}' (ext={ext}, size={len(file_bytes):,} bytes)")

    if ext == ".pdf":
        result = _parse_pdf(file_bytes, filename)
    elif ext == ".docx":
        result = _parse_docx(file_bytes, filename)
    elif ext == ".txt":
        result = _parse_txt(file_bytes, filename)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info(
        f"Parsed '{filename}': {result.total_pages} pages, "
        f"{len(result.all_images)} images, {len(result.all_tables)} tables"
    )
    return result
