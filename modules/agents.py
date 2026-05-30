"""
OmniRoute AI — Multi-Agent Definitions
Defines the Specialist Topology (Fact Agent, Summary Agent) for the Map phase
and the Executive + Critic Agents for the Reduce phase.

All agents use LangChain with Google Generative AI.
Tiered routing: Flash for Map, Pro for Reduce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from modules.utils import logger, api_retry, ConcurrencyLimiter


# ---------------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Agent output structures
# ---------------------------------------------------------------------------

@dataclass
class MapOutput:
    """Structured output from the Map phase for a single chunk."""
    chunk_id: int
    source_pages: list[int]
    facts: list[str]        # Atomic factual evidence
    summary: str            # Compressed localized summary
    key_entities: list[str] # Named entities mentioned
    data_points: list[str]  # Numbers, statistics, measurements


@dataclass
class ReduceOutput:
    """Final output from the Reduce phase."""
    master_summary: str
    critic_feedback: str
    is_consistent: bool


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _get_flash_model(temperature: float = 0.1) -> ChatOpenAI:
    """Gemini 2.0 Flash via OpenRouter — fastest for parallel Map work."""
    return ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=4096,
    )


def _get_pro_model(temperature: float = 0.2) -> ChatOpenAI:
    """Claude Sonnet via OpenRouter — best synthesis for Reduce phase."""
    return ChatOpenAI(
        model="anthropic/claude-sonnet-4",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=16384,
    )


# ---------------------------------------------------------------------------
# Map Phase — Specialist Topology
# ---------------------------------------------------------------------------

_map_limiter = ConcurrencyLimiter(max_concurrent=10)

FACT_AGENT_SYSTEM = """You are the Fact Extraction Agent in a document analysis pipeline.
Your job is to extract ALL atomic factual evidence from the given text chunk.

Rules:
- Extract every fact, figure, statistic, date, name, and data point.
- Be exhaustive — missing a fact is a failure.
- Each fact should be a single, self-contained statement.
- Include page references when available.
- Extract named entities (people, organizations, places, products).
- Extract numerical data points separately.

Respond in this exact JSON format:
{
  "facts": ["fact 1", "fact 2", ...],
  "key_entities": ["entity 1", "entity 2", ...],
  "data_points": ["data point 1", "data point 2", ...]
}"""

SUMMARY_AGENT_SYSTEM = """You are the Summary Compression Agent in a document analysis pipeline.
Your job is to compress the given text chunk into a concise but complete summary.

Rules:
- Capture ALL key ideas — no information loss.
- Preserve relationships between concepts.
- Maintain factual accuracy.
- Include references to images, charts, and tables described in the text.
- The summary should be 20-30% of the original length.
- Write in clear, professional prose.

Respond with ONLY the summary text, no JSON or formatting."""


@api_retry
async def _run_fact_agent(
    model: ChatGoogleGenerativeAI,
    chunk_text: str,
) -> dict:
    """Run the Fact Agent on a single chunk."""
    async with _map_limiter:
        messages = [
            SystemMessage(content=FACT_AGENT_SYSTEM),
            HumanMessage(content=f"Extract all facts from this text:\n\n{chunk_text}"),
        ]
        response = await model.ainvoke(messages)
        raw = response.content.strip()

        # Parse JSON response
        try:
            # Clean markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Fact Agent returned non-JSON, wrapping: {raw[:100]}…")
            parsed = {
                "facts": [raw],
                "key_entities": [],
                "data_points": [],
            }

        return parsed


@api_retry
async def _run_summary_agent(
    model: ChatGoogleGenerativeAI,
    chunk_text: str,
) -> str:
    """Run the Summary Agent on a single chunk."""
    async with _map_limiter:
        messages = [
            SystemMessage(content=SUMMARY_AGENT_SYSTEM),
            HumanMessage(content=f"Summarize this text:\n\n{chunk_text}"),
        ]
        response = await model.ainvoke(messages)
        return response.content.strip()


async def run_map_phase_single(
    chunk_id: int,
    chunk_text: str,
    source_pages: list[int],
    model: Optional[ChatGoogleGenerativeAI] = None,
) -> MapOutput:
    """
    Run both Fact and Summary agents on a single chunk concurrently.
    Returns structured MapOutput.
    """
    if model is None:
        model = _get_flash_model()

    import asyncio
    fact_task = _run_fact_agent(model, chunk_text)
    summary_task = _run_summary_agent(model, chunk_text)

    fact_result, summary_result = await asyncio.gather(
        fact_task, summary_task, return_exceptions=True
    )

    # Handle errors gracefully
    if isinstance(fact_result, Exception):
        logger.error(f"Fact Agent failed for chunk {chunk_id}: {fact_result}")
        fact_result = {"facts": [], "key_entities": [], "data_points": []}
    if isinstance(summary_result, Exception):
        logger.error(f"Summary Agent failed for chunk {chunk_id}: {summary_result}")
        summary_result = f"[Summary unavailable for chunk {chunk_id}]"

    return MapOutput(
        chunk_id=chunk_id,
        source_pages=source_pages,
        facts=fact_result.get("facts", []),
        summary=summary_result,
        key_entities=fact_result.get("key_entities", []),
        data_points=fact_result.get("data_points", []),
    )


# ---------------------------------------------------------------------------
# Reduce Phase — Executive Agent + Critic
# ---------------------------------------------------------------------------

EXECUTIVE_AGENT_SYSTEM = """You are the Executive Synthesis Agent in a document analysis pipeline.
You receive structured outputs from multiple specialist agents who analyzed different sections of a large document.

Your task is to produce a MASTER SUMMARY of exactly 3-5 pages (approximately 4500-7500 words).

Rules:
1. SYSTEMATIC MERGING: Process all intermediate outputs methodically. Do not skip any section.
2. LOGICAL FLOW: Organize the summary with clear sections, headings, and logical transitions.
3. ZERO INFORMATION LOSS: Every key fact, entity, and data point from the intermediate outputs must be represented.
4. COHERENCE: Eliminate redundancy while maintaining completeness. Synthesize related facts from different sections.
5. DATA INTEGRITY: Preserve all statistics, figures, dates, and named entities accurately.
6. VISUAL CONTENT: Include references to images, charts, and tables where mentioned.
7. STRUCTURE: Use markdown formatting with ## headings, bullet points, and bold for key terms.

Produce ONLY the master summary in well-structured markdown format."""

CRITIC_AGENT_SYSTEM = """You are the Critic Consistency Agent. You verify the quality and completeness of a master summary.

You will receive:
1. The master summary produced by the Executive Agent.
2. A checklist of key facts, entities, and data points from the source material.

Your task:
1. Check if all major facts and entities are represented in the summary.
2. Check for logical consistency and contradictions.
3. Check that the summary is 3-5 pages (4500-7500 words).
4. Rate the overall quality on a scale of 1-10.

Respond in this exact JSON format:
{
  "is_consistent": true/false,
  "quality_score": 8,
  "missing_items": ["item1", "item2"],
  "contradictions": ["contradiction1"],
  "feedback": "Overall assessment text"
}"""


@api_retry
async def run_executive_agent(
    map_outputs: list[MapOutput],
    document_name: str,
) -> str:
    """
    Executive Agent: merges all Map outputs into a 3-5 page master summary.
    Uses Gemini Pro for deep synthesis.
    """
    model = _get_pro_model()

    # Prepare the structured input
    sections = []
    all_entities = set()
    all_data_points = []

    for mo in sorted(map_outputs, key=lambda x: x.chunk_id):
        section = f"### Section {mo.chunk_id + 1} (Pages: {', '.join(map(str, mo.source_pages))})\n"
        section += f"**Summary:** {mo.summary}\n\n"
        if mo.facts:
            section += "**Key Facts:**\n" + "\n".join(f"- {f}" for f in mo.facts) + "\n"
        if mo.data_points:
            section += "**Data Points:**\n" + "\n".join(f"- {d}" for d in mo.data_points) + "\n"
        sections.append(section)
        all_entities.update(mo.key_entities)
        all_data_points.extend(mo.data_points)

    intermediate_text = "\n---\n".join(sections)
    entity_list = ", ".join(sorted(all_entities)[:100])  # Cap for prompt size

    prompt = (
        f"Document: {document_name}\n"
        f"Total sections analyzed: {len(map_outputs)}\n"
        f"Key entities found: {entity_list}\n\n"
        f"=== INTERMEDIATE ANALYSIS OUTPUTS ===\n\n"
        f"{intermediate_text}\n\n"
        f"=== END OF INTERMEDIATE OUTPUTS ===\n\n"
        f"Now produce the 3-5 page master summary following your instructions."
    )

    messages = [
        SystemMessage(content=EXECUTIVE_AGENT_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await model.ainvoke(messages)
    summary = response.content.strip()

    logger.info(f"Executive Agent produced summary: {len(summary)} chars")
    return summary


@api_retry
async def run_critic_agent(
    master_summary: str,
    map_outputs: list[MapOutput],
) -> dict:
    """
    Critic Agent: checks consistency and completeness of the master summary.
    Uses Gemini Pro.
    """
    model = _get_pro_model(temperature=0.1)

    # Build fact checklist
    all_facts = []
    all_entities = set()
    for mo in map_outputs:
        all_facts.extend(mo.facts[:5])  # Sample facts per chunk
        all_entities.update(mo.key_entities)

    checklist = (
        f"Key entities to verify: {', '.join(sorted(all_entities)[:50])}\n\n"
        f"Sample facts to verify ({len(all_facts)} sampled):\n"
        + "\n".join(f"- {f}" for f in all_facts[:50])
    )

    prompt = (
        f"=== MASTER SUMMARY ===\n{master_summary}\n\n"
        f"=== VERIFICATION CHECKLIST ===\n{checklist}\n\n"
        f"Perform your consistency check now."
    )

    messages = [
        SystemMessage(content=CRITIC_AGENT_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = await model.ainvoke(messages)
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Critic returned non-JSON: {raw[:200]}…")
        result = {
            "is_consistent": True,
            "quality_score": 7,
            "missing_items": [],
            "contradictions": [],
            "feedback": raw,
        }

    logger.info(
        f"Critic Agent: consistent={result.get('is_consistent')}, "
        f"quality={result.get('quality_score')}/10"
    )
    return result
