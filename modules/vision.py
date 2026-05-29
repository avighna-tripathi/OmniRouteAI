"""
OmniRoute AI — Vision Module
Uses Gemini Vision to generate descriptive captions for extracted images.
Fully async with rate-limit resilience.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from modules.parser import ExtractedImage
from modules.utils import logger, api_retry, ConcurrencyLimiter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CaptionedImage:
    """An image paired with its AI-generated caption."""
    page_number: int
    caption: str
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# Vision captioning
# ---------------------------------------------------------------------------

# Concurrency limiter for vision API calls
_vision_limiter = ConcurrencyLimiter(max_concurrent=5)

CAPTION_PROMPT = (
    "You are a precise document analysis assistant. "
    "Describe this image in detail — include all visible text, data, labels, "
    "charts, diagrams, and relationships. Be thorough and factual. "
    "If it's a chart or graph, describe the axes, data trends, and key values. "
    "If it's a diagram, describe the components and their connections."
)


def _get_vision_model() -> ChatGoogleGenerativeAI:
    """Initialize Gemini 2.0 Flash for vision (fast, cost-efficient)."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.1,
        max_output_tokens=1024,
    )


@api_retry
async def _caption_single_image(
    model: ChatGoogleGenerativeAI,
    image: ExtractedImage,
) -> CaptionedImage:
    """Generate a caption for a single image with retry logic."""
    async with _vision_limiter:
        b64_data = image.to_base64()

        message = HumanMessage(
            content=[
                {"type": "text", "text": CAPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{b64_data}"
                    },
                },
            ]
        )

        response = await model.ainvoke([message])
        caption = response.content.strip()

        logger.info(
            f"Captioned image on page {image.page_number} "
            f"({image.width}x{image.height}): {caption[:80]}…"
        )

        return CaptionedImage(
            page_number=image.page_number,
            caption=caption,
            width=image.width,
            height=image.height,
        )


async def caption_images(
    images: list[ExtractedImage],
    progress_callback=None,
) -> list[CaptionedImage]:
    """
    Caption all extracted images concurrently using Gemini Vision.

    Args:
        images: List of ExtractedImage objects.
        progress_callback: Optional callable(completed, total) for progress updates.

    Returns:
        List of CaptionedImage objects, one per input image.
    """
    if not images:
        logger.info("No images to caption.")
        return []

    model = _get_vision_model()
    total = len(images)
    logger.info(f"Captioning {total} images with Gemini Vision…")

    completed = 0
    results: list[CaptionedImage] = []

    # Process in batches to respect rate limits
    batch_size = 5
    for i in range(0, total, batch_size):
        batch = images[i:i + batch_size]
        tasks = [_caption_single_image(model, img) for img in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to caption image on page {batch[j].page_number}: {result}"
                )
                # Create a fallback caption so we don't lose track of the image
                results.append(CaptionedImage(
                    page_number=batch[j].page_number,
                    caption=f"[Image on page {batch[j].page_number} — captioning failed]",
                    width=batch[j].width,
                    height=batch[j].height,
                ))
            else:
                results.append(result)

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    logger.info(f"Captioning complete: {len(results)}/{total} images processed.")
    return results
