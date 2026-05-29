"""
OmniRoute AI — Streamlit Application
Main entry point. Premium UI with file upload, dynamic progress tracking,
and streaming summary output.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import pandas as pd
import streamlit as st

from modules.parser import SUPPORTED_EXTENSIONS
from modules.pipeline import run_pipeline, PipelineProgress
from modules.table_store import (
    get_tables_for_session,
    get_document_names_for_session,
    delete_tables_for_document_session,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OmniRoute AI — Intelligent Document Summarization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with glassmorphism
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Root variables ---- */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255, 255, 255, 0.04);
        --bg-glass: rgba(255, 255, 255, 0.06);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --text-primary: #e8e8ed;
        --text-secondary: #9898a6;
        --accent-primary: #6c5ce7;
        --accent-secondary: #a29bfe;
        --accent-gradient: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 50%, #74b9ff 100%);
        --success: #00b894;
        --warning: #fdcb6e;
        --error: #e17055;
        --glow-purple: 0 0 30px rgba(108, 92, 231, 0.3);
    }

    /* ---- Global resets ---- */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Header area ---- */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(108,92,231,0.2), rgba(162,155,254,0.15));
        border: 1px solid rgba(108,92,231,0.3);
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent-secondary);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ---- Glass cards ---- */
    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(108, 92, 231, 0.25);
        box-shadow: var(--glow-purple);
    }

    /* ---- Stats row ---- */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    .stat-item {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stat-item:hover { transform: translateY(-2px); }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }

    /* ---- Stage indicator ---- */
    .stage-indicator {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    .stage-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--accent-primary);
        animation: pulse-dot 1.5s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(108,92,231,0.4); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(108,92,231,0); }
    }
    .stage-text {
        font-size: 0.9rem;
        color: var(--text-primary);
        font-weight: 500;
    }

    /* ---- Pipeline architecture diagram ---- */
    .pipeline-viz {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }
    .pipeline-node {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-primary);
        text-align: center;
        min-width: 90px;
        transition: all 0.3s ease;
    }
    .pipeline-node:hover {
        border-color: var(--accent-primary);
        box-shadow: var(--glow-purple);
        transform: translateY(-2px);
    }
    .pipeline-node.active {
        border-color: var(--accent-primary);
        background: rgba(108,92,231,0.15);
        box-shadow: var(--glow-purple);
    }
    .pipeline-arrow {
        color: var(--accent-secondary);
        font-size: 1.1rem;
        opacity: 0.6;
    }

    /* ---- Sidebar styling ---- */
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    /* ---- Success / error banners ---- */
    .result-banner {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
    }
    .result-banner.success {
        background: rgba(0,184,148,0.1);
        border: 1px solid rgba(0,184,148,0.3);
        color: var(--success);
    }
    .result-banner.warning {
        background: rgba(253,203,110,0.1);
        border: 1px solid rgba(253,203,110,0.3);
        color: var(--warning);
    }

    /* ---- Footer ---- */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary);
        opacity: 0.6;
    }

    /* ---- Hide default Streamlit elements ---- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization — persistent token via URL query params
# ---------------------------------------------------------------------------

# Generate or restore a persistent session token.
# Stored in the URL as ?token=xxxx so it survives page refreshes.
# Each user gets a unique token → their tables are isolated in MongoDB.
params = st.query_params
if "token" in params:
    _session_token = params["token"]
else:
    _session_token = uuid.uuid4().hex[:16]  # 16-char cryptographic token
    st.query_params["token"] = _session_token

if "session_id" not in st.session_state:
    st.session_state.session_id = _session_token
else:
    st.session_state.session_id = _session_token

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_stats" not in st.session_state:
    st.session_state.pipeline_stats = {}
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Agentic Multi-Agent RAG</div>
    <div class="hero-title">OmniRoute AI</div>
    <div class="hero-subtitle">
        Zero-loss document intelligence — transforms 200-page documents into
        precise 3–5 page summaries using a multi-agent Map-Reduce pipeline.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline architecture visualization
# ---------------------------------------------------------------------------

st.markdown("""
<div class="pipeline-viz">
    <div class="pipeline-node">📄 Parser</div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-node">🖼️ Vision</div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-node">🗄️ Tables</div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-node">✂️ Chunker</div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-node">🔬 Map Phase</div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-node">🧠 Executive</div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-node">🔍 Critic</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Configuration & Info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">📎 Document Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt"],
        help="Supports PDF, Word (.docx), and plain text files up to 200 pages.",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">⚙️ Pipeline Configuration</div>', unsafe_allow_html=True)

    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=2000,
        max_value=8000,
        value=4000,
        step=500,
        help="Larger chunks = fewer API calls but less granularity.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=100,
        max_value=1000,
        value=400,
        step=100,
        help="Overlap ensures cross-chunk context continuity.",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🏗️ Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    | Phase | Model |
    |-------|-------|
    | Map (Fact/Summary) | `Gemini 2.0 Flash` |
    | Reduce (Executive) | `Gemini 2.5 Pro` |
    | Critic | `Gemini 2.5 Pro` |
    | Vision | `Gemini 2.0 Flash` |
    """)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📊 Session Info</div>', unsafe_allow_html=True)
    st.code(f"Session: {st.session_state.session_id}", language=None)


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

if uploaded_file is not None and not st.session_state.is_processing:

    # File info card
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(f"""
    <div class="glass-card">
        <strong>📄 {uploaded_file.name}</strong>
        <span style="color: var(--text-secondary); margin-left: 1rem;">
            {file_size_mb:.2f} MB · {uploaded_file.type}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_btn = st.button(
            "🚀  Launch OmniRoute Pipeline",
            use_container_width=True,
            type="primary",
        )

    if process_btn:
        st.session_state.is_processing = True
        st.session_state.pipeline_result = None
        st.session_state.pipeline_stats = {}

        # Read file bytes
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        # Progress tracking UI
        progress_bar = st.progress(0.0)
        stage_container = st.empty()
        stats_container = st.empty()

        def update_progress(p: PipelineProgress):
            """Callback to update Streamlit progress UI."""
            progress_bar.progress(
                min(p.overall_progress, 1.0),
                text=p.message,
            )
            stage_container.markdown(f"""
            <div class="stage-indicator">
                <div class="stage-dot"></div>
                <span class="stage-text">{p.message}</span>
            </div>
            """, unsafe_allow_html=True)

        # Run the async pipeline
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                run_pipeline(
                    file_bytes=file_bytes,
                    filename=filename,
                    progress_callback=update_progress,
                    session_id=st.session_state.session_id,
                )
            )
            loop.close()

            st.session_state.pipeline_result = result
            progress_bar.progress(1.0, text="✅ Pipeline complete!")
            stage_container.empty()
            st.session_state.is_processing = False
            st.rerun()

        except Exception as e:
            st.session_state.is_processing = False
            progress_bar.empty()
            stage_container.empty()
            st.error(f"❌ Pipeline failed: {str(e)}")
            st.exception(e)

elif st.session_state.is_processing:
    st.info("⏳ Pipeline is running… please wait.")


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if st.session_state.pipeline_result is not None:
    result = st.session_state.pipeline_result

    # Consistency banner
    if result.is_consistent:
        st.markdown("""
        <div class="result-banner success">
            ✅ Critic Verification Passed — Summary is consistent and complete.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-banner warning">
            ⚠️ Critic flagged potential inconsistencies — review recommended.
        </div>
        """, unsafe_allow_html=True)

    # Master Summary — Streamed output
    st.markdown("## 📋 Master Summary")
    st.markdown("---")

    # Use st.write_stream for token-level streaming effect
    def _stream_summary():
        """Generator that yields the summary in word-sized chunks for streaming."""
        words = result.master_summary.split(" ")
        buffer = ""
        for i, word in enumerate(words):
            buffer += word + " "
            if len(buffer) >= 15 or i == len(words) - 1:
                yield buffer
                buffer = ""
                time.sleep(0.02)  # Small delay for visual streaming effect

    st.write_stream(_stream_summary())

    # Critic feedback
    if result.critic_feedback:
        with st.expander("🔍 Critic Agent Feedback", expanded=False):
            st.markdown(result.critic_feedback)

    # Download button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥  Download Summary as Markdown",
            data=result.master_summary,
            file_name="omniroute_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Table Viewer — Session-scoped, user-isolated
# ---------------------------------------------------------------------------

st.markdown("")
st.markdown("## 🗄️ Extracted Tables Viewer")
st.markdown(
    '<p style="color: var(--text-secondary); font-size: 0.9rem;">'
    'Tables extracted from your documents are stored securely in MongoDB. '
    'Only you can see tables from your session.'
    '</p>',
    unsafe_allow_html=True,
)

try:
    user_documents = get_document_names_for_session(st.session_state.session_id)
except Exception as e:
    user_documents = []
    st.warning(f"⚠️ Could not connect to MongoDB: {e}")

if user_documents:
    # Document selector
    selected_doc = st.selectbox(
        "Select a document",
        options=user_documents,
        format_func=lambda x: f"📄 {x}",
        key="table_viewer_doc_select",
    )

    if selected_doc:
        tables = get_tables_for_session(st.session_state.session_id, selected_doc)

        if tables:
            st.markdown(f"""
            <div class="glass-card">
                <strong>📊 {len(tables)} table(s)</strong> extracted from
                <strong>{selected_doc}</strong>
                <span style="color: var(--text-secondary); margin-left: 0.5rem;">
                    🔒 Session-scoped · Only visible to you
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Display each table
            for idx, tbl in enumerate(tables):
                page_num = tbl.get("page_number", "?")
                headers = tbl.get("headers", [])
                raw = tbl.get("raw", [])

                with st.expander(
                    f"📋 Table {idx + 1} — Page {page_num} "
                    f"({len(raw) - 1 if len(raw) > 1 else 0} rows, "
                    f"{len(headers)} columns)",
                    expanded=(idx == 0),
                ):
                    if raw and len(raw) > 1:
                        df = pd.DataFrame(raw[1:], columns=raw[0] if raw[0] else None)
                        st.dataframe(df, use_container_width=True)
                    elif raw:
                        st.dataframe(pd.DataFrame(raw), use_container_width=True)
                    else:
                        st.info("Empty table.")

            # Download & Delete actions
            st.markdown("")
            act_col1, act_col2, act_col3 = st.columns([1, 1, 1])

            with act_col1:
                # Download as CSV
                all_rows = []
                for tbl in tables:
                    raw = tbl.get("raw", [])
                    if raw and len(raw) > 1:
                        for row in raw[1:]:
                            row_dict = dict(zip(raw[0], row)) if raw[0] else {}
                            row_dict["_page"] = tbl.get("page_number", "")
                            all_rows.append(row_dict)
                if all_rows:
                    csv_df = pd.DataFrame(all_rows)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_df.to_csv(index=False),
                        file_name=f"{selected_doc}_tables.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            with act_col2:
                # Download as JSON
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(tables, indent=2, default=str),
                    file_name=f"{selected_doc}_tables.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with act_col3:
                # Delete tables for this document
                if st.button(
                    "🗑️ Delete Tables",
                    use_container_width=True,
                    key=f"delete_{selected_doc}",
                ):
                    deleted = delete_tables_for_document_session(
                        st.session_state.session_id, selected_doc
                    )
                    st.success(f"Deleted {deleted} table(s) for '{selected_doc}'.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No tables found for this document.")

else:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 0.75rem;">🗄️</div>
        <div style="color: var(--text-secondary); font-size: 0.9rem;">
            No tables extracted yet. Upload and process a document to see extracted tables here.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if uploaded_file is None and st.session_state.pipeline_result is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
            Upload a document to get started
        </div>
        <div style="color: var(--text-secondary); font-size: 0.9rem;">
            Supports PDF, Word (.docx), and plain text files.<br/>
            Documents up to 200 pages will be processed with zero content loss.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔬</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Zero-Loss Extraction</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                Every page, image, and table is captured. Nothing is dropped or filtered.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Async Parallel Processing</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                Dozens of chunks processed concurrently with batched API calls and rate-limit resilience.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">Multi-Agent Synthesis</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                Specialist Fact & Summary agents feed an Executive Agent for deep, coherent summaries.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div class="footer">
    OmniRoute AI · Built with Streamlit, LangChain, and Gemini · Multi-Agent Map-Reduce Architecture
</div>
""", unsafe_allow_html=True)
