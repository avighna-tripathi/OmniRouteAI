# OmniRoute AI — Project Blueprint

## Overview
OmniRoute AI is a high-performance, zero-loss, agentic RAG (Retrieval-Augmented Generation) application that synthesizes documents up to 200 pages into a strict 3–5 page master summary using a multi-agent Map-Reduce pipeline. It uses **Forced Routing** — every chunk is processed, nothing is dropped.

---

## Core Architecture

### Input
- **Streamlit file upload** accepting PDF, Word (.docx), and Text (.txt) files.
- Supports documents up to **200 pages**.

### Data Preservation (Zero-Loss)
- Extract **text**, **images**, and **tables** from every page.
- No content, page, or image may be dropped at any stage.

### Table Storage
- Extracted tables are pushed directly to **MongoDB** as structured JSON with page metadata.
- This preserves structural integrity that would be lost in plain text.

### UI/UX
- **Streamlit** interface with:
  - Dynamic progress bar for background tasks.
  - Token streaming (`st.write_stream`) for final output.

### Secrets Management
- All API keys (Gemini API, MongoDB URI) accessed via **`st.secrets`** dictionary.
- No `.env` files or `os.getenv`.

---

## Multi-Agent Map-Reduce Pipeline

### Phase 1: Parser & Vision Module
1. Parse the uploaded document.
2. Separate **tables** → push to MongoDB as JSON.
3. Extract **text** per page.
4. Extract **images** → pass to **Gemini Vision** for descriptive text captions.

### Phase 2: Forced Routing — Map Phase
1. Chunk the combined text + image captions.
2. Route **every single chunk** to a Specialist Topology:
   - **Fact Agent:** Extracts atomic factual evidence (key facts, figures, entities).
   - **Summary Agent:** Compresses the chunk into a concise localized summary.
3. Each chunk produces a structured output: `{ facts: [...], summary: "..." }`.

### Phase 3: Executive Aggregation — Reduce Phase
1. **Executive Agent** (Gemini Pro) receives all structured Map outputs.
2. Systematically merges intermediate outputs into a cohesive **3–5 page master summary**.
3. **Critic Agent** performs a consistency check on the final output.

---

## Performance & Latency Engineering

### Asynchronous Batching
- Use `asyncio` and LangChain's `.abatch()` to run the Map Phase concurrently.
- Process dozens of chunks in parallel.

### Tiered Model Routing
| Phase | Model | Reason |
|-------|-------|--------|
| Map Phase (Fact/Summary Agents) | `gemini-1.5-flash` | Speed + cost-efficiency for parallel work |
| Reduce Phase (Executive/Critic) | `gemini-1.5-pro` | Deep logical synthesis |

### Rate Limit Resilience
- **Exponential backoff** via `tenacity` library on all API calls.
- Smooth handling of RPM/TPM limits during async Map phase.

---

## Folder Structure

```
OmniRoute AI/
├── PROJECT_BLUEPRINT.md
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── secrets.toml            # Secrets template (Gemini API, MongoDB URI)
├── modules/
│   ├── __init__.py
│   ├── parser.py               # Document parsing (PDF, DOCX, TXT)
│   ├── vision.py               # Gemini Vision image captioning
│   ├── table_store.py          # MongoDB table storage
│   ├── chunker.py              # Text chunking logic
│   ├── agents.py               # Fact Agent, Summary Agent, Executive Agent, Critic
│   ├── pipeline.py             # Orchestrates the Map-Reduce pipeline
│   └── utils.py                # Retry logic, helpers
```

---

## Secrets Template (`.streamlit/secrets.toml`)

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
MONGODB_URI = "your-mongodb-connection-string-here"
MONGODB_DB_NAME = "omniroute_ai"
MONGODB_COLLECTION = "extracted_tables"
```

---

## Project State & Progress

- [x] PROJECT_BLUEPRINT.md created
- [x] Folder structure initialized
- [x] `requirements.txt` written
- [x] `.streamlit/secrets.toml` template created
- [x] `modules/utils.py` — Retry logic, helpers
- [x] `modules/parser.py` — Document parsing
- [x] `modules/vision.py` — Gemini Vision captioning
- [x] `modules/table_store.py` — MongoDB table storage
- [x] `modules/chunker.py` — Text chunking
- [x] `modules/agents.py` — Multi-agent definitions
- [x] `modules/pipeline.py` — Map-Reduce orchestration
- [x] `app.py` — Streamlit UI
- [ ] End-to-end testing
