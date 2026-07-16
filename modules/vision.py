"""
Handles multimodal capabilities — describing extracted images using Gemini Vision.
"""
import asyncio
from dataclasses import dataclass

import streamlit as st
from google.genai import types

from modules.parser import ExtractedImage
from modules.utils import logger, api_retry
from modules.gemini_client import GeminiModel


MODEL_VISION = st.secrets.get("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CaptionedImage:
    page_number: int
    caption: str
    image_id: str = ""
    occurrence: int = 0


# ---------------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "Describe this image in detail. If it is a chart or graph, summarize the key data trends, axes, "
    "and any readable numbers. If it is a diagram, explain the workflow or architecture shown. "
    "If it is a photo, describe the subjects and context. Be precise and thorough."
)


def _get_vision_model() -> GeminiModel:
    return GeminiModel(MODEL_VISION, temperature=0.1)


@api_retry
async def _caption_single_image(model: GeminiModel, image: ExtractedImage) -> CaptionedImage:
    """Sends a single image to the Vision model to generate a descriptive caption."""
    logger.info(f"Generating caption for image on page {image.page_number}...")
    response = await model.generate_content([
        types.Part.from_bytes(data=image.image_bytes, mime_type=image.mime_type),
        CAPTION_PROMPT,
    ])

    return CaptionedImage(
        page_number=image.page_number,
        caption=f"[Image on page {image.page_number}]: {response.content}",
        image_id=image.image_id,
        occurrence=image.occurrence,
    )


# ---------------------------------------------------------------------------
# Orchestrator — sequential with sleep for rate pacing
# ---------------------------------------------------------------------------

async def caption_images(images: list[ExtractedImage], progress_callback=None) -> list[CaptionedImage]:
    """
    Processes extracted images one-by-one with a sleep delay between each call.
    Sequential approach avoids asyncio.Semaphore cross-event-loop deadlocks.
    """
    if not images:
        return []

    model = _get_vision_model()
    results: list[CaptionedImage] = []
    total = len(images)
    delay = float(st.secrets.get("VISION_REQUEST_DELAY_SECONDS", 1.0))
    caption_cache: dict[str, str] = {}

    for idx, image in enumerate(images):
        if idx > 0 and delay > 0:
            await asyncio.sleep(delay)

        try:
            if image.image_id in caption_cache:
                result = CaptionedImage(
                    page_number=image.page_number,
                    caption=caption_cache[image.image_id],
                    image_id=image.image_id,
                    occurrence=image.occurrence,
                )
            else:
                result = await _caption_single_image(model, image)
                caption_cache[image.image_id] = result.caption
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to caption image on page {image.page_number}: {e}")
            results.append(CaptionedImage(
                page_number=image.page_number,
                caption=f"[Image on page {image.page_number} — captioning failed: {str(e)[:100]}]"
            ))

        if progress_callback:
            progress_callback(idx + 1, total)

    return results
