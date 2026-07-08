"""
OmniRoute AI — Pipeline Orchestrator
Orchestrates the full Map-Reduce pipeline:
  1. Parse document → extract text, images, tables
  2. Store tables in MongoDB
  3. Caption images with Gemini Vision
  4. Chunk the enriched text
  5. Map Phase: run Fact + Summary agents on all chunks (async batch)
  6. Reduce Phase: Executive Agent merges → Critic validates
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from modules.parser import parse_document, ParsedDocument
from modules.vision import caption_images, CaptionedImage
from modules.table_store import store_tables
from modules.chunker import chunk_document, ProcessedChunk
from modules.agents import (
    MapOutput,
    ReduceOutput,
    run_map_phase_single,
    run_executive_agent,
    run_critic_agent,
    _get_fast_model,
)
from modules.utils import logger


# ---------------------------------------------------------------------------
# Pipeline progress tracking
# ---------------------------------------------------------------------------

@dataclass
class PipelineProgress:
    """Tracks progress across all pipeline stages."""
    stage: str = "Initializing"
    stage_progress: float = 0.0       # 0.0 – 1.0 within current stage
    overall_progress: float = 0.0     # 0.0 – 1.0 overall
    message: str = ""
    started_at: float = field(default_factory=time.time)
    stats: dict = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


# Stage weights for overall progress calculation
STAGE_WEIGHTS = {
    "parsing": 0.10,
    "tables": 0.05,
    "vision": 0.15,
    "chunking": 0.05,
    "map_phase": 0.40,
    "reduce_phase": 0.20,
    "critic": 0.05,
}


def _calc_overall(stage: str, stage_pct: float) -> float:
    """Calculate overall progress given current stage and its completion."""
    stages = list(STAGE_WEIGHTS.keys())
    completed = 0.0
    for s in stages:
        if s == stage:
            completed += STAGE_WEIGHTS[s] * stage_pct
            break
        completed += STAGE_WEIGHTS[s]
    return min(completed, 1.0)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    file_bytes: bytes,
    filename: str,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    session_id: Optional[str] = None,
) -> ReduceOutput:
    """
    Execute the full OmniRoute AI pipeline.

    Args:
        file_bytes: Raw bytes of the uploaded document.
        filename: Original filename (used for type detection and metadata).
        progress_callback: Optional callable to receive PipelineProgress updates.
        session_id: Optional session ID for MongoDB tagging.

    Returns:
        ReduceOutput with master_summary, critic_feedback, and consistency flag.
    """
    progress = PipelineProgress()

    def _update(stage: str, stage_pct: float, msg: str):
        progress.stage = stage
        progress.stage_progress = stage_pct
        progress.overall_progress = _calc_overall(stage, stage_pct)
        progress.message = msg
        if progress_callback:
            progress_callback(progress)

    # ===================================================================
    # STAGE 1: Parse Document
    # ===================================================================
    _update("parsing", 0.0, "📄 Parsing document…")
    logger.info(f"Pipeline started for '{filename}'")

    parsed_doc: ParsedDocument = parse_document(file_bytes, filename)

    progress.stats["total_pages"] = parsed_doc.total_pages
    progress.stats["total_images"] = len(parsed_doc.all_images)
    progress.stats["total_tables"] = len(parsed_doc.all_tables)

    _update("parsing", 1.0, f"✅ Parsed {parsed_doc.total_pages} pages")

    # ===================================================================
    # STAGE 2: Store Tables in MongoDB
    # ===================================================================
    _update("tables", 0.0, "🗄️ Storing tables in MongoDB…")

    tables = parsed_doc.all_tables
    tables_stored = 0
    if tables:
        try:
            tables_stored = store_tables(tables, filename, session_id)
        except Exception as e:
            logger.error(f"MongoDB table storage failed: {e}")
            # Non-fatal: continue pipeline, tables still referenced in text
            tables_stored = 0

    progress.stats["tables_stored"] = tables_stored
    _update("tables", 1.0, f"✅ {tables_stored} tables stored")

    # ===================================================================
    # STAGE 3: Caption Images with Gemini Vision
    # ===================================================================
    _update("vision", 0.0, "🖼️ Captioning images with Gemini Vision…")

    images = parsed_doc.all_images
    captions: list[CaptionedImage] = []

    if images:
        def vision_progress(completed, total):
            _update("vision", completed / total, f"🖼️ Captioned {completed}/{total} images")

        captions = await caption_images(images, progress_callback=vision_progress)
    else:
        logger.info("No images found in document.")

    progress.stats["images_captioned"] = len(captions)
    _update("vision", 1.0, f"✅ {len(captions)} images captioned")

    # ===================================================================
    # STAGE 4: Chunk Document
    # ===================================================================
    _update("chunking", 0.0, "✂️ Chunking document…")

    chunks: list[ProcessedChunk] = chunk_document(parsed_doc, captions)

    progress.stats["total_chunks"] = len(chunks)
    _update("chunking", 1.0, f"✅ {len(chunks)} chunks created")

    if not chunks:
        logger.error("No chunks produced — document may be empty.")
        return ReduceOutput(
            master_summary="⚠️ The document appears to be empty or could not be parsed.",
            critic_feedback="No content to analyze.",
            is_consistent=False,
        )

    # ===================================================================
    # STAGE 5: Map Phase — Fact + Summary Agents (Async Batch)
    # ===================================================================
    _update("map_phase", 0.0, f"🔬 Running Map phase on {len(chunks)} chunks…")

    fast_model = _get_fast_model()
    map_outputs: list[MapOutput] = []
    completed_chunks = 0

    # Process chunks sequentially — avoids asyncio.Semaphore cross-loop deadlocks
    # Sleep is handled inside run_map_phase_single to pace free-tier RPM limits
    for chunk in chunks:
        try:
            output = await run_map_phase_single(
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
                source_pages=chunk.source_pages,
                model=fast_model,
            )
            map_outputs.append(output)
        except Exception as e:
            logger.error(f"Map phase failed for chunk {chunk.chunk_id}: {e}")
            map_outputs.append(MapOutput(
                chunk_id=chunk.chunk_id,
                source_pages=chunk.source_pages,
                facts=[],
                summary=f"[Processing failed for chunk {chunk.chunk_id}]",
            ))

        completed_chunks += 1
        _update(
            "map_phase",
            completed_chunks / len(chunks),
            f"🔬 Map phase: {completed_chunks}/{len(chunks)} chunks processed",
        )

    total_facts = sum(len(mo.facts) for mo in map_outputs)
    progress.stats["total_facts_extracted"] = total_facts
    _update("map_phase", 1.0, f"✅ Map phase complete: {total_facts} facts extracted")

    # ===================================================================
    # STAGE 6: Reduce Phase — Executive Agent
    # ===================================================================
    _update("reduce_phase", 0.0, "🧠 Executive Agent synthesizing master summary…")

    master_summary = await run_executive_agent(map_outputs, filename)

    if not master_summary or not master_summary.strip():
        logger.error(
            f"Executive Agent returned empty summary! "
            f"Map outputs count: {len(map_outputs)}, "
            f"total facts: {sum(len(m.facts) for m in map_outputs)}"
        )
        # Build a fallback summary from map outputs so the user gets SOMETHING
        fallback_parts = ["## Auto-generated Fallback Summary\n",
                          "*The Executive Agent returned an empty response. "
                          "Below are the individual section summaries:*\n"]
        for mo in sorted(map_outputs, key=lambda x: x.chunk_id):
            fallback_parts.append(
                f"### Section {mo.chunk_id + 1} (Pages {', '.join(map(str, mo.source_pages))})\n"
                f"{mo.summary}\n"
            )
        master_summary = "\n".join(fallback_parts)

    progress.stats["summary_length"] = len(master_summary)
    _update("reduce_phase", 1.0, "✅ Master summary generated")

    # ===================================================================
    # STAGE 7: Critic Consistency Check
    # ===================================================================
    _update("critic", 0.0, "🔍 Critic Agent verifying consistency…")

    critic_result = await run_critic_agent(master_summary, map_outputs)

    is_consistent = critic_result.get("is_consistent", True)
    quality_score = critic_result.get("quality_score", 0)
    feedback = critic_result.get("feedback", "")

    progress.stats["quality_score"] = quality_score
    progress.stats["is_consistent"] = is_consistent

    _update("critic", 1.0, f"✅ Quality score: {quality_score}/10")

    # ===================================================================
    # DONE
    # ===================================================================
    elapsed = progress.elapsed
    logger.info(
        f"Pipeline complete for '{filename}' in {elapsed:.1f}s — "
        f"Quality: {quality_score}/10, Consistent: {is_consistent}"
    )
    progress.stats["elapsed_seconds"] = round(elapsed, 1)

    return ReduceOutput(
        master_summary=master_summary,
        critic_feedback=feedback,
        is_consistent=is_consistent,
    )
