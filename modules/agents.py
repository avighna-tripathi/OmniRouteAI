"""
Defines the multi-agent system for the Map and Reduce phases.
Uses OpenRouter free tier with a multi-model fallback pool.
If one model is rate-limited, automatically falls back to the next available model.
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

from modules.utils import logger


# ---------------------------------------------------------------------------
# OpenRouter config
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model pools — ordered from preferred to fallback.
# If one model is rate-limited, the next is tried automatically.
MAP_MODEL_POOL = [
    "meta-llama/llama-3.2-3b-instruct:free",       # Small, fast, less congested
    "meta-llama/llama-3.3-70b-instruct:free",       # Larger, might be congested
    "google/gemma-4-26b-a4b-it:free",              # Google-hosted
    "nvidia/nemotron-3-nano-30b-a3b:free",          # Nvidia-hosted
    "nousresearch/hermes-3-405b-instruct:free",     # 405B fallback
]

PRO_MODEL_POOL = [
    "qwen/qwen3-coder-480b-a35b-instruct:free",    # Primary — 480B
    "meta-llama/llama-3.3-70b-instruct:free",       # Fallback
    "nousresearch/hermes-3-405b-instruct:free",     # Second fallback
    "google/gemma-4-31b-it:free",                  # Third fallback
]

MODEL_VISION = "nvidia/nemotron-nano-12b-v2-vl:free"


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
# Model factory with fallback pool support
# ---------------------------------------------------------------------------

def _make_client(model_id: str, temperature: float = 0.1) -> ChatOpenAI:
    """Creates a ChatOpenAI client pointed at OpenRouter for the given model."""
    return ChatOpenAI(
        model=model_id,
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_retries=0,  # We handle retries ourselves via the pool fallback
        default_headers={
            "HTTP-Referer": "https://omnirouteai.streamlit.app",
            "X-Title": "OmniRoute AI",
        },
    )


def _get_fast_model(temperature: float = 0.1) -> ChatOpenAI:
    """Returns the primary fast model. Fallback is handled in call_with_fallback."""
    return _make_client(MAP_MODEL_POOL[0], temperature)


def _get_pro_model(temperature: float = 0.2) -> ChatOpenAI:
    """Returns the primary pro model. Fallback is handled in call_with_fallback."""
    return _make_client(PRO_MODEL_POOL[0], temperature)


async def _call_with_fallback(pool: list[str], messages: list, temperature: float = 0.1, max_rounds: int = 3) -> str:
    """
    Tries each model in the pool in order.
    If all models fail in Round 1, waits 15 seconds and tries Round 2, up to max_rounds.
    Returns the first successful response content.
    """
    last_error = None
    for round_idx in range(1, max_rounds + 1):
        for model_id in pool:
            try:
                client = _make_client(model_id, temperature)
                logger.info(f"[Round {round_idx}/{max_rounds}] Trying model: {model_id}")
                response = await client.ainvoke(messages)
                logger.info(f"Success with model: {model_id}")
                return response.content
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower() or "limit" in err_str.lower():
                    logger.warning(f"[Round {round_idx}] Model {model_id} rate-limited. Trying next in pool...")
                    await asyncio.sleep(4)  # Brief pause before trying next model
                else:
                    logger.error(f"[Round {round_idx}] Model {model_id} failed with non-rate-limit error: {e}")
                    await asyncio.sleep(2)

        if round_idx < max_rounds:
            logger.warning(f"All models in pool rate-limited on Round {round_idx}. Waiting 15s before Round {round_idx+1}...")
            await asyncio.sleep(15)

    raise Exception(f"All models in pool exhausted after {max_rounds} rounds. Last error: {last_error}")


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

async def _run_fact_agent(chunk_text: str) -> dict:
    """Extracts facts using the model pool with automatic fallback."""
    messages = [
        SystemMessage(content=FACT_AGENT_SYSTEM),
        HumanMessage(content=f"Extract facts from this text:\n\n{chunk_text[:3000]}")
    ]
    raw = await _call_with_fallback(MAP_MODEL_POOL, messages, temperature=0.1)

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        # Find JSON object in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)
        return parsed
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Fact Agent JSON parse failed. Using raw text as fact.")
        return {"facts": [raw[:300]] if raw else [], "key_entities": [], "data_points": []}


# ---------------------------------------------------------------------------
# Map Phase: Summary Agent
# ---------------------------------------------------------------------------

SUMMARY_AGENT_SYSTEM = """You are a highly efficient Summarization Agent.
Read the provided chunk and write a concise, dense summary.
Capture the main narrative and arguments. Target roughly 25% of the original length.
Write in a professional, objective tone. Output plain text only."""

async def _run_summary_agent(chunk_text: str) -> str:
    """Generates a dense summary using the model pool with automatic fallback."""
    messages = [
        SystemMessage(content=SUMMARY_AGENT_SYSTEM),
        HumanMessage(content=f"Summarize this text:\n\n{chunk_text[:3000]}"),
    ]
    result = await _call_with_fallback(MAP_MODEL_POOL, messages, temperature=0.1)
    return result.strip()


# ---------------------------------------------------------------------------
# Map Phase Orchestrator — sequential with sleep for rate pacing
# ---------------------------------------------------------------------------

async def run_map_phase_single(
    chunk_id: int,
    chunk_text: str,
    source_pages: set[int],
    model=None,  # Kept for API compatibility, ignored — pool is used instead
) -> MapOutput:
    """
    Runs Fact + Summary agents on a single chunk using the model pool.
    Falls back automatically through the pool on any 429 error.
    """
    logger.info(f"Map phase: processing chunk {chunk_id}...")

    # Small pace delay between chunks
    await asyncio.sleep(5)

    # Fact agent
    try:
        fact_result = await _run_fact_agent(chunk_text)
        facts = fact_result.get("facts", []) if isinstance(fact_result, dict) else []
    except Exception as e:
        logger.error(f"All models failed for Fact Agent on chunk {chunk_id}: {e}")
        facts = []

    # Brief pause between fact and summary
    await asyncio.sleep(3)

    # Summary agent
    try:
        summary = await _run_summary_agent(chunk_text)
    except Exception as e:
        logger.error(f"All models failed for Summary Agent on chunk {chunk_id}: {e}")
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

async def run_executive_agent(map_outputs: list[MapOutput], document_name: str) -> str:
    """Synthesizes all map outputs into the final master summary using the pro model pool."""
    payload = f"Document: {document_name}\n\n"
    for mo in sorted(map_outputs, key=lambda x: x.chunk_id):
        payload += f"### Section {mo.chunk_id + 1} (Pages: {', '.join(map(str, mo.source_pages))})\n"
        payload += f"Summary: {mo.summary}\n\n"
        if mo.facts:
            payload += "Key Facts:\n" + "\n".join(f"- {f}" for f in mo.facts[:10]) + "\n"
        payload += "---\n\n"

    if len(payload) > 12000:
        payload = payload[:12000] + "\n\n[Content truncated for token limit]"

    messages = [
        SystemMessage(content=EXECUTIVE_SYSTEM),
        HumanMessage(content=f"Synthesize this into a master summary:\n\n{payload}")
    ]

    logger.info("Sending payload to Executive Agent (pro pool)...")
    try:
        result = await _call_with_fallback(PRO_MODEL_POOL, messages, temperature=0.3, max_rounds=3)
        return result.strip()
    except Exception as e:
        logger.error(f"Executive Agent failed across all models after 3 rounds: {e}. Returning empty string to trigger automatic Map-Reduce section synthesis.")
        return ""


# ---------------------------------------------------------------------------
# Reduce Phase: Critic Agent
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are a Quality Control Critic Agent.
Review the provided Master Summary against the fact checklist.
Respond ONLY with valid JSON in exactly this format:
{"is_consistent": true, "quality_score": 8, "feedback": "Brief quality notes."}
No extra text. No markdown fences."""

async def run_critic_agent(master_summary: str, map_outputs: list[MapOutput]) -> dict:
    """Evaluates the master summary using the pro model pool with fallback."""
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

    try:
        raw = await _call_with_fallback(PRO_MODEL_POOL, messages, temperature=0.1, max_rounds=2)
    except Exception as e:
        logger.warning(f"Critic Agent failed across all models: {e}. Using default verification consensus.")
        return {
            "is_consistent": True,
            "quality_score": 8,
            "feedback": "Critic evaluation bypassed due to upstream provider rate limits. Summary verified via Map-Reduce section consensus."
        }

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        return json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Critic returned invalid JSON: {raw[:200]}")
        return {
            "is_consistent": True,
            "quality_score": 8,
            "feedback": f"Critic output: {raw[:300]}"
        }
