"""
Handles multimodal capabilities — describing extracted images using a Vision LLM.
Uses OpenRouter free vision model: meta-llama/llama-3.2-11b-vision-instruct:free
"""
import asyncio
from dataclasses import dataclass

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from modules.parser import ExtractedImage
from modules.utils import logger, api_retry


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_VISION = "meta-llama/llama-3.2-11b-vision-instruct:free"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CaptionedImage:
    page_number: int
    caption: str


# ---------------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "Describe this image in detail. If it is a chart or graph, summarize the key data trends, axes, "
    "and any readable numbers. If it is a diagram, explain the workflow or architecture shown. "
    "If it is a photo, describe the subjects and context. Be precise and thorough."
)


def _get_vision_model() -> ChatOpenAI:
    """Llama 3.2 Vision 11B via OpenRouter — free multimodal model."""
    return ChatOpenAI(
        model=MODEL_VISION,
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        max_retries=4,
        default_headers={
            "HTTP-Referer": "https://omnirouteai.streamlit.app",
            "X-Title": "OmniRoute AI",
        },
    )


@api_retry
async def _caption_single_image(model: ChatOpenAI, image: ExtractedImage) -> CaptionedImage:
    """Sends a single image to the Vision model to generate a descriptive caption."""
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

    logger.info(f"Generating caption for image on page {image.page_number}...")
    response = await model.ainvoke([message])

    return CaptionedImage(
        page_number=image.page_number,
        caption=f"[Image on page {image.page_number}]: {response.content}"
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

    for idx, image in enumerate(images):
        if idx > 0:
            await asyncio.sleep(3)  # Pace calls under OpenRouter free-tier RPM

        try:
            result = await _caption_single_image(model, image)
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
