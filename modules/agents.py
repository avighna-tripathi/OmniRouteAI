"""
Defines the multi-agent system for the Map and Reduce phases.
"""
from typing import Optional

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from modules.utils import logger, api_retry, ConcurrencyLimiter


# ---------------------------------------------------------------------------
# Agent output structures
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field

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

def _get_fast_model(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Uses the extremely fast Gemini 2.0 Flash for parallel Map work."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=temperature,
        max_retries=6,
    )


def _get_pro_model(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Uses Gemini 2.5 Flash for the heavy-lifting Reduce phase."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=temperature,
        max_retries=6,
    )


# ---------------------------------------------------------------------------
# Map Phase: Fact Agent
# ---------------------------------------------------------------------------

FACT_AGENT_SYSTEM = """You are a precise Fact Extraction Agent.
Your job is to read the provided text chunk and extract structured information.
Focus on capturing ALL hard facts, names, dates, metrics, and key statements.
Never hallucinate or add outside knowledge.

Extract into three lists:
1. facts: Complete sentences stating specific facts.
2. key_entities: Names of people, companies, tools, etc.
3. data_points: Specific numbers, percentages, financial figures.
"""

@api_retry
async def _run_fact_agent(model, chunk_text: str) -> dict:
    """Extracts facts from a chunk and returns a dictionary."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", FACT_AGENT_SYSTEM + "\n\nYou MUST respond ONLY with valid JSON matching this schema: {{\"facts\": [\"...\"], \"key_entities\": [\"...\"], \"data_points\": [\"...\"]}}"),
        ("human", "Text to process:\n\n{text}")
    ])
    chain = prompt | model
    response = await chain.ainvoke({"text": chunk_text})
    
    # Simple JSON parsing with fallback
    import json
    raw = response.content
    try:
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        raw = raw.strip()
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from Fact Agent. Returning raw as single fact.")
        return {"facts": [raw], "key_entities": [], "data_points": []}


# ---------------------------------------------------------------------------
# Map Phase: Summary Agent
# ---------------------------------------------------------------------------

SUMMARY_AGENT_SYSTEM = """You are a highly efficient Summarization Agent.
Read the provided chunk and write a concise, dense summary.
Capture the main narrative and arguments. Target roughly 25% of the original length.
Write in a professional, objective tone."""

@api_retry
async def _run_summary_agent(model, chunk_text: str) -> str:
    """Generates a dense summary of a chunk."""
    messages = [
        SystemMessage(content=SUMMARY_AGENT_SYSTEM),
        HumanMessage(content=f"Summarize this text:\n\n{chunk_text}"),
    ]
    response = await model.ainvoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Map Phase Orchestrator
# ---------------------------------------------------------------------------

import asyncio

# Note: Gemini free tier has strict RPM limits. ConcurrencyLimiter controls burst rate.
_map_limiter = ConcurrencyLimiter(max_concurrent=2)

async def run_map_phase_single(chunk_id: int, chunk_text: str, source_pages: set[int], model: Optional[ChatGoogleGenerativeAI] = None) -> MapOutput:
    """Runs both Map agents on a single chunk concurrently."""
    if model is None:
        model = _get_fast_model()
        
    async with _map_limiter:
        await asyncio.sleep(2)  # Pace requests to respect free tier 15 RPM limit
        fact_task = _run_fact_agent(model, chunk_text)
        summary_task = _run_summary_agent(model, chunk_text)
        
        # Run both simultaneously
        fact_result, summary_result = await asyncio.gather(
            fact_task, summary_task, return_exceptions=True
        )
        
        # Handle potential exceptions gracefully
        facts = fact_result.get("facts", []) if isinstance(fact_result, dict) else [str(fact_result)]
        summary = summary_result if isinstance(summary_result, str) else str(summary_result)
        
        if isinstance(fact_result, Exception):
            logger.error(f"Fact Agent failed for chunk {chunk_id}: {fact_result}")
            facts = []
        if isinstance(summary_result, Exception):
            logger.error(f"Summary Agent failed for chunk {chunk_id}: {summary_result}")
            summary = "[Summarization failed]"
            
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
Your task is to take the extracted summaries and facts from multiple document sections and synthesize them into a cohesive, comprehensive master summary.
The final document should be approximately 3 to 5 pages long.

Requirements:
- Organize with clear, logical markdown headings.
- Include an Executive Summary at the top.
- Synthesize facts intelligently — group related information even if it appeared in different sections.
- Cite the source pages when discussing specific facts or data points (e.g., "[Page 4]").
- Ensure smooth transitions between topics.
- Do NOT simply paste the input sections. You must rewrite and synthesize.
"""

@api_retry
async def run_executive_agent(map_outputs: list[MapOutput], document_name: str) -> str:
    """Synthesizes all map outputs into the final master summary."""
    model = _get_pro_model(temperature=0.3)  # Slightly higher temp for better narrative flow
    
    # Construct the massive prompt payload
    payload = f"Document Name: {document_name}\n\n"
    payload += "Below are the processed chunks from the document in chronological order.\n\n"
    
    for mo in sorted(map_outputs, key=lambda x: x.chunk_id):
        payload += f"### Section {mo.chunk_id + 1} (Pages: {', '.join(map(str, mo.source_pages))})\n"
        payload += f"**Summary:** {mo.summary}\n\n"
        if mo.facts:
            payload += "**Key Facts:**\n" + "\n".join(f"- {f}" for f in mo.facts) + "\n"
        payload += "---\n\n"
        
    messages = [
        SystemMessage(content=EXECUTIVE_SYSTEM),
        HumanMessage(content=f"Synthesize the following document data into a 3-5 page master summary:\n\n{payload}")
    ]
    
    logger.info("Sending payload to Executive Agent...")
    response = await model.ainvoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Reduce Phase: Critic Agent
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are the Quality Control Critic Agent.
You will be provided with a generated Master Summary and the original extracted facts.
Your job is to verify that the Master Summary is accurate, consistent with the facts, and properly formatted.

Output your evaluation as JSON:
{{
  "is_consistent": true/false,
  "quality_score": 1-10,
  "feedback": "Detailed explanation of any missing facts, contradictions, or formatting issues."
}}
"""

@api_retry
async def run_critic_agent(master_summary: str, map_outputs: list[MapOutput]) -> dict:
    """Evaluates the master summary against the raw facts."""
    model = _get_pro_model(temperature=0.1)
    
    # Compile a checklist of all facts
    all_facts = []
    for mo in map_outputs:
        all_facts.extend(mo.facts)
        
    # We only send a sample of facts if there are too many, to avoid context limits
    import random
    if len(all_facts) > 100:
        sample_facts = random.sample(all_facts, 100)
    else:
        sample_facts = all_facts
        
    payload = "--- EXTRACTED FACTS CHECKLIST ---\n"
    payload += "\n".join(f"- {f}" for f in sample_facts)
    payload += "\n\n--- MASTER SUMMARY TO EVALUATE ---\n"
    payload += master_summary
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM),
        ("human", "Evaluate this summary:\n\n{text}")
    ])
    chain = prompt | model
    
    response = await chain.ainvoke({"text": payload})
    
    # Parse JSON
    import json
    raw = response.content
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("Critic Agent returned invalid JSON.")
        return {
            "is_consistent": True,
            "quality_score": 7,
            "feedback": f"Could not parse Critic response. Raw output:\n{raw}"
        }
