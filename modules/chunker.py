"""
OmniRoute AI — Chunker Module
Splits extracted text and image captions into semantically coherent chunks
for the Map phase. Uses recursive character splitting with overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from modules.parser import ParsedDocument, PageContent
from modules.vision import CaptionedImage
from modules.table_store import get_table_summary_text
from modules.utils import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 4000       # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 400     # Overlap between chunks for continuity
MIN_CHUNK_SIZE = 200            # Skip tiny leftover chunks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProcessedChunk:
    """A chunk ready for the Map phase agents."""
    chunk_id: int
    text: str
    source_pages: list[int]   # Which pages contributed to this chunk
    has_image_captions: bool
    has_table_refs: bool


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def _build_page_text_blocks(
    parsed_doc: ParsedDocument,
    captions: list[CaptionedImage],
) -> list[tuple[int, str]]:
    """
    Merge page text with image captions and table summaries.
    Returns list of (page_number, enriched_text) tuples.
    """
    # Index captions by page
    captions_by_page: dict[int, list[str]] = {}
    for cap in captions:
        captions_by_page.setdefault(cap.page_number, []).append(cap.caption)

    blocks: list[tuple[int, str]] = []

    for page in parsed_doc.pages:
        parts: list[str] = []

        # Page header
        parts.append(f"--- Page {page.page_number} ---")

        # Main text
        if page.text.strip():
            parts.append(page.text.strip())

        # Image captions for this page
        page_captions = captions_by_page.get(page.page_number, [])
        if page_captions:
            parts.append("\n[Image Descriptions]")
            for idx, cap in enumerate(page_captions, 1):
                parts.append(f"  Image {idx}: {cap}")

        # Table references (actual data is in MongoDB)
        if page.tables:
            table_summary = get_table_summary_text(page.tables)
            if table_summary:
                parts.append(f"\n[Table References]\n{table_summary}")

        combined = "\n".join(parts)
        if combined.strip():
            blocks.append((page.page_number, combined))

    return blocks


def chunk_document(
    parsed_doc: ParsedDocument,
    captions: list[CaptionedImage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ProcessedChunk]:
    """
    Chunk the full document text (with image captions and table refs merged)
    into overlapping segments for the Map phase.

    Strategy:
    - Build enriched text per page (text + captions + table refs).
    - Concatenate all pages.
    - Split with overlap to maintain cross-page context.
    - Track source page numbers for each chunk.

    Returns:
        List of ProcessedChunk objects, guaranteed to cover 100% of content.
    """
    # Build enriched text blocks
    page_blocks = _build_page_text_blocks(parsed_doc, captions)

    if not page_blocks:
        logger.warning("No text content to chunk.")
        return []

    # Build a single string with page markers for tracking
    full_text = ""
    # Track character offset → page number mapping
    char_to_page: list[tuple[int, int, int]] = []  # (start, end, page_num)

    for page_num, block_text in page_blocks:
        start = len(full_text)
        full_text += block_text + "\n\n"
        end = len(full_text)
        char_to_page.append((start, end, page_num))

    # Split into chunks
    chunks: list[ProcessedChunk] = []
    chunk_id = 0
    pos = 0

    while pos < len(full_text):
        end = min(pos + chunk_size, len(full_text))
        chunk_text = full_text[pos:end].strip()

        if len(chunk_text) < MIN_CHUNK_SIZE and chunks:
            # Append tiny remainder to last chunk
            chunks[-1].text += "\n" + chunk_text
            break

        if chunk_text:
            # Determine source pages for this chunk
            source_pages = set()
            for seg_start, seg_end, pg in char_to_page:
                # Check if chunk overlaps with this page's range
                if pos < seg_end and end > seg_start:
                    source_pages.add(pg)

            chunks.append(ProcessedChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_pages=sorted(source_pages),
                has_image_captions="[Image Descriptions]" in chunk_text,
                has_table_refs="[Table References]" in chunk_text,
            ))
            chunk_id += 1

        # Advance position with overlap
        pos = end - chunk_overlap if end < len(full_text) else end

    logger.info(
        f"Chunked document into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks
