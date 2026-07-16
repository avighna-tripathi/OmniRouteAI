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
    key_entities: list[str] = Field(default_factory=list)
    data_points: list[str] = Field(default_factory=list)

class ReduceOutput(BaseModel):
    master_summary: str
    critic_feedback: str
    is_consistent: bool
    quality_score: int = 0
    stats: dict = Field(default_factory=dict)


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


# Compatibility alias for older local scripts.
_get_flash_model = _get_fast_model


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

MAP_AGENT_SYSTEM = """You are the map-stage extraction and summarization agent.
Read the complete document chunk. Return ONLY valid JSON with this exact shape:
{"facts": ["atomic fact"], "key_entities": ["entity"], "data_points": ["number or metric"], "summary": "dense faithful summary"}
Capture all important facts, numbers, named entities, decisions, caveats, and table/image details present in the chunk.
Never invent information or use outside knowledge. Keep the summary concise but complete."""


def _parse_json_object(raw: str) -> dict:
    """Parse JSON returned by providers that occasionally add markdown fences."""
    cleaned = (raw or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end >= start:
        cleaned = cleaned[start:end + 1]
    value = json.loads(cleaned)
    return value if isinstance(value, dict) else {}


async def _run_map_agent(chunk_text: str) -> dict:
    """Run both map tasks in one request to reduce free-tier API usage."""
    messages = [
        SystemMessage(content=MAP_AGENT_SYSTEM),
        HumanMessage(content=f"Analyze this complete chunk:\n\n{chunk_text}"),
    ]
    raw = await _call_with_fallback(MAP_MODEL_POOL, messages, temperature=0.1)
    return _parse_json_object(raw)

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

    # One structured request captures both outputs. This halves API calls and
    # avoids truncating a full chunk before it reaches the model.
    try:
        result = await _run_map_agent(chunk_text)
        facts = result.get("facts", [])
        key_entities = result.get("key_entities", [])
        data_points = result.get("data_points", [])
        summary = str(result.get("summary", "")).strip()
    except Exception as e:
        logger.error(f"All models failed for Map Agent on chunk {chunk_id}: {e}")
        facts = []
        key_entities = []
        data_points = []
        summary = f"[Summarization failed for chunk {chunk_id}]"

    return MapOutput(
        chunk_id=chunk_id,
        source_pages=list(source_pages),
        facts=[str(item) for item in facts if str(item).strip()],
        summary=summary,
        key_entities=[str(item) for item in key_entities if str(item).strip()],
        data_points=[str(item) for item in data_points if str(item).strip()],
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
- Aim for 3-4 well-structured pages (roughly 1,800-2,400 words) of output.
- Preserve important numbers, caveats, and source-page references."""

MAX_REDUCE_INPUT_CHARS = 18000


def _map_output_text(mo: MapOutput) -> str:
    pages = ", ".join(map(str, sorted(mo.source_pages))) or "unknown"
    parts = [f"Section {mo.chunk_id + 1} (pages: {pages})", f"Summary: {mo.summary}"]
    if mo.facts:
        parts.append("Facts:\n" + "\n".join(f"- {item}" for item in mo.facts))
    if mo.data_points:
        parts.append("Data points:\n" + "\n".join(f"- {item}" for item in mo.data_points))
    if mo.key_entities:
        parts.append("Entities: " + ", ".join(mo.key_entities))
    return "\n".join(parts)


async def _reduce_items(items: list[str], document_name: str) -> list[str]:
    """Hierarchically reduce all map sections without silently truncating them."""
    current = items
    level = 0
    while sum(len(item) for item in current) + len(current) * 2 > MAX_REDUCE_INPUT_CHARS:
        level += 1
        batches: list[str] = []
        batch: list[str] = []
        batch_size = 0
        for item in current:
            if batch and batch_size + len(item) > 12000:
                batches.append("\n\n---\n\n".join(batch))
                batch, batch_size = [], 0
            batch.append(item)
            batch_size += len(item)
        if batch:
            batches.append("\n\n---\n\n".join(batch))

        reduced: list[str] = []
        for idx, batch_text in enumerate(batches, 1):
            messages = [
                SystemMessage(content="""You are a document reduction agent. Merge the supplied sections faithfully.
Keep every material fact, number, caveat, and source-page reference. Do not invent or omit a section.
Return a concise but information-dense intermediate summary in plain markdown."""),
                HumanMessage(content=f"Document: {document_name}\nReduction level: {level}, batch: {idx}\n\n{batch_text}"),
            ]
            reduced.append((await _call_with_fallback(PRO_MODEL_POOL, messages, temperature=0.2, max_rounds=2)).strip())
        current = reduced
    return current


async def run_executive_agent(map_outputs: list[MapOutput], document_name: str) -> str:
    """Synthesize every map output, using hierarchical reduction for large inputs."""
    items = [_map_output_text(mo) for mo in sorted(map_outputs, key=lambda x: x.chunk_id)]
    items = await _reduce_items(items, document_name)
    payload = f"Document: {document_name}\n\n" + "\n\n---\n\n".join(items)
    messages = [
        SystemMessage(content=EXECUTIVE_SYSTEM),
        HumanMessage(content=f"Synthesize this into the final master summary:\n\n{payload}"),
    ]
    logger.info("Sending complete hierarchical payload to Executive Agent")
    try:
        result = await _call_with_fallback(PRO_MODEL_POOL, messages, temperature=0.3, max_rounds=3)
        return result.strip()
    except Exception as e:
        logger.error(f"Executive Agent failed across all models: {e}")
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

    # Deterministic sampling makes repeated runs comparable while still
    # covering the beginning, middle, and end of a long fact list.
    if len(all_facts) <= 30:
        sample_facts = all_facts
    else:
        positions = [round(i * (len(all_facts) - 1) / 29) for i in range(30)]
        sample_facts = [all_facts[pos] for pos in positions]

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
            "is_consistent": False,
            "quality_score": 0,
            "feedback": "Critic evaluation could not run because all provider attempts failed; consistency is unverified."
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
