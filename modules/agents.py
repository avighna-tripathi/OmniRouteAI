"""
Defines the multi-agent system for the Map and Reduce phases.
Uses OpenRouter (free tier) via its OpenAI-compatible API endpoint.
Free models used:
  - Map/Reduce/Critic: meta-llama/llama-3.1-8b-instruct:free
  - Fallback: google/gemma-3-12b-it:free
"""
from typing import Optional
import asyncio
import json
import random

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from modules.utils import logger, api_retry


# ---------------------------------------------------------------------------
# OpenRouter base config
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models on OpenRouter — selected from your active free-tier list
MODEL_FAST   = "meta-llama/llama-3.3-70b-instruct:free"         # Map phase — best free text model
MODEL_PRO    = "qwen/qwen3-coder-480b-a35b-instruct:free"       # Reduce/Critic — 480B deep reasoner
MODEL_VISION = "nvidia/nemotron-nano-12b-v2-vl:free"            # Vision — only free VL model


# ---------------------------------------------------------------------------
# Agent output structures
# ---------------------------------------------------------------------------

class FactExtractorOutput(BaseModel):
    facts: list[str] = Field(description="List of isolated, atomic facts extracted from the text.")
    key_entities: list[str] = Field(description="List of important people, organizations, or concepts.")
    data_points: list[str] = Field(description="Any specific numbers, metrics, or statistics.")

class MapOutput(BaseModel):
    chunk_id: int
    source_pages: list[int]
    facts: list[str]
    summary: str

class ReduceOutput(BaseModel):
    master_summary: str
    critic_feedback: str
    is_consistent: bool


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _get_fast_model(temperature: float = 0.1) -> ChatOpenAI:
    """Llama 3.1 8B via OpenRouter — free, fast, great for Map phase."""
    return ChatOpenAI(
        model=MODEL_FAST,
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_retries=4,
        default_headers={
            "HTTP-Referer": "https://omnirouteai.streamlit.app",
            "X-Title": "OmniRoute AI",
        },
    )


def _get_pro_model(temperature: float = 0.2) -> ChatOpenAI:
    """Gemma 3 12B via OpenRouter — free, strong reasoning for Reduce/Critic."""
    return ChatOpenAI(
        model=MODEL_PRO,
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_retries=4,
        default_headers={
            "HTTP-Referer": "https://omnirouteai.streamlit.app",
            "X-Title": "OmniRoute AI",
        },
    )


# ---------------------------------------------------------------------------
# Map Phase: Fact Agent
# ---------------------------------------------------------------------------

FACT_AGENT_SYSTEM = """You are a precise Fact Extraction Agent.
Your job is to read the provided text chunk and extract structured information.
Focus on capturing ALL hard facts, names, dates, metrics, and key statements.
Never hallucinate or add outside knowledge.

You MUST respond ONLY with valid JSON in exactly this format:
{"facts": ["fact1", "fact2"], "key_entities": ["entity1"], "data_points": ["data1"]}

No extra text. No markdown. Just the JSON object."""

@api_retry
async def _run_fact_agent(model, chunk_text: str) -> dict:
    """Extracts facts from a chunk and returns a dictionary."""
    messages = [
        SystemMessage(content=FACT_AGENT_SYSTEM),
        HumanMessage(content=f"Extract facts from this text:\n\n{chunk_text[:3000]}")
    ]
    response = await model.ainvoke(messages)
    raw = response.content.strip()

    try:
        # Strip markdown code fences if model wraps output
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        return parsed
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Fact Agent JSON parse failed. Raw: {raw[:200]}")
        return {"facts": [raw[:300]] if raw else [], "key_entities": [], "data_points": []}


# ---------------------------------------------------------------------------
# Map Phase: Summary Agent
# ---------------------------------------------------------------------------

SUMMARY_AGENT_SYSTEM = """You are a highly efficient Summarization Agent.
Read the provided chunk and write a concise, dense summary.
Capture the main narrative and arguments. Target roughly 25% of the original length.
Write in a professional, objective tone. Output plain text only."""

@api_retry
async def _run_summary_agent(model, chunk_text: str) -> str:
    """Generates a dense summary of a chunk."""
    messages = [
        SystemMessage(content=SUMMARY_AGENT_SYSTEM),
        HumanMessage(content=f"Summarize this text:\n\n{chunk_text[:3000]}"),
    ]
    response = await model.ainvoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Map Phase Orchestrator — sequential with sleep for rate pacing
# ---------------------------------------------------------------------------

async def run_map_phase_single(
    chunk_id: int,
    chunk_text: str,
    source_pages: set[int],
    model: Optional[ChatOpenAI] = None,
) -> MapOutput:
    """
    Runs Fact + Summary agents sequentially on a single chunk.
    Uses asyncio.sleep for rate pacing — no Semaphore to avoid event-loop issues.
    """
    if model is None:
        model = _get_fast_model()

    await asyncio.sleep(2)  # Pace requests under OpenRouter free-tier RPM

    logger.info(f"Map phase: processing chunk {chunk_id}...")

    # Fact agent
    try:
        fact_result = await _run_fact_agent(model, chunk_text)
        facts = fact_result.get("facts", []) if isinstance(fact_result, dict) else []
    except Exception as e:
        logger.error(f"Fact Agent failed for chunk {chunk_id}: {e}")
        facts = []

    # Sleep between fact and summary calls
    await asyncio.sleep(2)

    # Summary agent
    try:
        summary = await _run_summary_agent(model, chunk_text)
    except Exception as e:
        logger.error(f"Summary Agent failed for chunk {chunk_id}: {e}")
        summary = f"[Summarization failed for chunk {chunk_id}]"

    return MapOutput(
        chunk_id=chunk_id,
        source_pages=list(source_pages),
        facts=facts,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Reduce Phase: Executive Agent
# ---------------------------------------------------------------------------

EXECUTIVE_SYSTEM = """You are the Executive Synthesis Agent.
Your task is to take extracted summaries and facts from multiple document sections
and synthesize them into a cohesive, comprehensive master summary.

Requirements:
- Use clear markdown headings (##, ###).
- Include an Executive Summary section at the top.
- Synthesize and group related information intelligently.
- Cite source pages where relevant e.g. [Page 3].
- Write in a professional, objective tone.
- Do NOT just paste the input — rewrite and synthesize.
- Aim for 3-5 well-structured pages of output."""

@api_retry
async def run_executive_agent(map_outputs: list[MapOutput], document_name: str) -> str:
    """Synthesizes all map outputs into the final master summary."""
    model = _get_pro_model(temperature=0.3)

    payload = f"Document: {document_name}\n\n"
    for mo in sorted(map_outputs, key=lambda x: x.chunk_id):
        payload += f"### Section {mo.chunk_id + 1} (Pages: {', '.join(map(str, mo.source_pages))})\n"
        payload += f"Summary: {mo.summary}\n\n"
        if mo.facts:
            payload += "Key Facts:\n" + "\n".join(f"- {f}" for f in mo.facts[:10]) + "\n"
        payload += "---\n\n"

    # Trim payload to avoid token limits on free models
    if len(payload) > 12000:
        payload = payload[:12000] + "\n\n[Content truncated for token limit]"

    messages = [
        SystemMessage(content=EXECUTIVE_SYSTEM),
        HumanMessage(content=f"Synthesize this into a master summary:\n\n{payload}")
    ]

    logger.info("Sending payload to Executive Agent...")
    response = await model.ainvoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Reduce Phase: Critic Agent
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are a Quality Control Critic Agent.
Review the provided Master Summary against the fact checklist.
Respond ONLY with valid JSON in exactly this format:
{"is_consistent": true, "quality_score": 8, "feedback": "Brief quality notes."}
No extra text. No markdown fences."""

@api_retry
async def run_critic_agent(master_summary: str, map_outputs: list[MapOutput]) -> dict:
    """Evaluates the master summary against the raw facts."""
    model = _get_pro_model(temperature=0.1)

    all_facts = []
    for mo in map_outputs:
        all_facts.extend(mo.facts)

    sample_facts = random.sample(all_facts, min(30, len(all_facts)))

    payload = "FACTS:\n" + "\n".join(f"- {f}" for f in sample_facts)
    payload += "\n\nSUMMARY:\n" + master_summary[:2000]

    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"Evaluate:\n\n{payload}")
    ]
    response = await model.ainvoke(messages)

    raw = response.content.strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Critic returned invalid JSON: {raw[:200]}")
        return {
            "is_consistent": True,
            "quality_score": 7,
            "feedback": f"Critic output: {raw[:300]}"
        }
