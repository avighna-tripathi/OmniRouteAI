"""
Handles multimodal capabilities — describing extracted images using Gemini Vision.
"""
import io
import base64
import asyncio
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
        max_retries=6,
    )


# Concurrency limiter to avoid hitting Gemini rate limits
_vision_limiter = ConcurrencyLimiter(max_concurrent=2)

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
    
    async with _vision_limiter:
        await asyncio.sleep(2)  # Pace requests to respect free tier 15 RPM limit
        logger.info(f"Generating caption for image on page {image.page_number}...")
        response = await model.ainvoke([message])
        
    return CaptionedImage(
        page_number=image.page_number,
        caption=f"[Image on page {image.page_number}]: {response.content}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def caption_images(images: list[ExtractedImage], progress_callback=None) -> list[CaptionedImage]:
    """
    Processes a list of extracted images concurrently, returning their captions.
    """
    if not images:
        return []

    model = _get_vision_model()
    results: list[CaptionedImage] = []
    
    # Process in batches to manage rate limits gracefully
    batch_size = 2
    total = len(images)
    completed = 0
    
    for i in range(0, total, batch_size):
        batch = images[i:i + batch_size]
        tasks = [_caption_single_image(model, img) for img in batch]
        
        # Run batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to caption image on page {batch[j].page_number}: {result}")
                # Fallback to empty caption so pipeline doesn't break
                results.append(CaptionedImage(
                    page_number=batch[j].page_number,
                    caption=f"[Image on page {batch[j].page_number} — captioning failed due to API error]"
                ))
            else:
                results.append(result)
                
        completed += len(batch)
        if progress_callback:
            progress_callback(min(completed, total), total)
                
    return results
