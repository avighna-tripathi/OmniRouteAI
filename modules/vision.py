"""
Handles multimodal capabilities — describing extracted images using Gemini Vision.
"""
import asyncio
from dataclasses import dataclass

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from modules.parser import ExtractedImage
from modules.utils import logger, api_retry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CaptionedImage:
    page_number: int
    caption: str


# ---------------------------------------------------------------------------
# Vision Agent Setup
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "Describe this image in detail. If it is a chart or graph, summarize the key data trends, axes, "
    "and any readable numbers. If it is a diagram, explain the workflow or architecture shown. "
    "If it is a photo, describe the subjects and context. Be precise and thorough, as this text "
    "will be used in a broader document summary."
)


def _get_vision_model() -> ChatGoogleGenerativeAI:
    """Initialize Gemini 2.0 Flash for Vision."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.1,
        max_retries=3,
    )


@api_retry
async def _caption_single_image(model: ChatGoogleGenerativeAI, image: ExtractedImage) -> CaptionedImage:
    """Sends a single image to the Vision model to generate a descriptive caption."""
    b64_data = image.to_base64()

    # LangChain multimodal message format
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

    logger.info(f"Generating caption for image on page {image.page_number}...")
    response = await model.ainvoke([message])

    return CaptionedImage(
        page_number=image.page_number,
        caption=f"[Image on page {image.page_number}]: {response.content}"
    )


# ---------------------------------------------------------------------------
# Orchestrator — fully sequential, no Semaphore (avoids event-loop deadlocks)
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

    for idx, image in enumerate(images):
        # Pace calls to respect Gemini free-tier 15 RPM limit
        if idx > 0:
            await asyncio.sleep(4)

        try:
            result = await _caption_single_image(model, image)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to caption image on page {image.page_number}: {e}")
            results.append(CaptionedImage(
                page_number=image.page_number,
                caption=f"[Image on page {image.page_number} — captioning failed due to API error]"
            ))

        if progress_callback:
            progress_callback(idx + 1, total)

    return results
