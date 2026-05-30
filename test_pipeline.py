"""
OmniRoute AI — Local Test Script
Runs the pipeline on a given PDF without the Streamlit UI.
Prints detailed step-by-step output to diagnose issues.
"""

import asyncio
import sys
import os

# Streamlit secrets emulation — load from .streamlit/secrets.toml
# We need to set this up BEFORE importing any module that uses st.secrets
os.environ["STREAMLIT_SECRETS_PATH"] = os.path.join(
    os.path.dirname(__file__), ".streamlit", "secrets.toml"
)

# Mock st.secrets and st.cache_resource for non-Streamlit usage
import toml
import streamlit as st

# Load secrets
secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
with open(secrets_path) as f:
    _secrets = toml.loads(f.read())

# Patch st.secrets if running outside streamlit
# Actually, we'll use streamlit's own mechanism by just importing properly

def main():
    pdf_path = r"C:\Users\hp\Downloads\Use2.pdf"
    
    print("=" * 60)
    print("OmniRoute AI — Local Pipeline Test")
    print("=" * 60)
    
    # Step 1: Read file
    print(f"\n[1/7] Reading file: {pdf_path}")
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    print(f"      File size: {len(file_bytes):,} bytes ({len(file_bytes)/1024/1024:.2f} MB)")
    
    # Step 2: Parse
    print(f"\n[2/7] Parsing document...")
    from modules.parser import parse_document
    parsed = parse_document(file_bytes, os.path.basename(pdf_path))
    print(f"      Pages: {parsed.total_pages}")
    print(f"      Images: {len(parsed.all_images)}")
    print(f"      Tables: {len(parsed.all_tables)}")
    print(f"      Text length: {len(parsed.full_text):,} chars")
    if parsed.full_text:
        print(f"      First 200 chars: {parsed.full_text[:200]}...")
    else:
        print("      ⚠️ NO TEXT EXTRACTED!")
    
    # Step 3: Store tables
    print(f"\n[3/7] Storing tables in MongoDB...")
    from modules.table_store import store_tables
    try:
        stored = store_tables(parsed.all_tables, os.path.basename(pdf_path), "test-session")
        print(f"      Stored: {stored} tables")
    except Exception as e:
        print(f"      ⚠️ MongoDB failed (non-fatal): {e}")
    
    # Step 4: Caption images
    print(f"\n[4/7] Captioning images...")
    from modules.vision import caption_images
    
    async def do_caption():
        if not parsed.all_images:
            print("      No images to caption.")
            return []
        captions = await caption_images(parsed.all_images)
        for cap in captions:
            print(f"      Page {cap.page_number}: {cap.caption[:80]}...")
        return captions
    
    captions = asyncio.run(do_caption())
    
    # Step 5: Chunk
    print(f"\n[5/7] Chunking document...")
    from modules.chunker import chunk_document
    chunks = chunk_document(parsed, captions)
    print(f"      Chunks created: {len(chunks)}")
    for c in chunks:
        print(f"      Chunk {c.chunk_id}: {len(c.text)} chars, pages {c.source_pages}")
    
    # Step 6: Map phase
    print(f"\n[6/7] Running Map phase...")
    from modules.agents import run_map_phase_single, _get_flash_model
    
    async def do_map():
        model = _get_flash_model()
        results = []
        for c in chunks:
            print(f"      Processing chunk {c.chunk_id}...", end=" ", flush=True)
            try:
                result = await run_map_phase_single(c.chunk_id, c.text, c.source_pages, model)
                print(f"[OK] facts={len(result.facts)}, summary={len(result.summary)} chars")
                results.append(result)
            except Exception as e:
                print(f"[FAIL] {e}")
        return results
    
    map_outputs = asyncio.run(do_map())
    total_facts = sum(len(m.facts) for m in map_outputs)
    print(f"      Total facts extracted: {total_facts}")
    
    # Step 7: Reduce phase
    print(f"\n[7/7] Running Reduce phase (Executive Agent)...")
    from modules.agents import run_executive_agent, run_critic_agent
    
    async def do_reduce():
        print("      Calling Executive Agent...")
        summary = await run_executive_agent(map_outputs, os.path.basename(pdf_path))
        print(f"      Summary length: {len(summary)} chars")
        if summary:
            print(f"      First 300 chars:\n      {summary[:300]}...")
        else:
            print("      ⚠️ EMPTY SUMMARY RETURNED!")
        
        print("\n      Calling Critic Agent...")
        critic = await run_critic_agent(summary, map_outputs)
        print(f"      Critic result: {critic}")
        
        return summary, critic
    
    summary, critic = asyncio.run(do_reduce())
    
    # Final output
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Summary length: {len(summary)} chars")
    print(f"Critic consistent: {critic.get('is_consistent', 'N/A')}")
    print(f"Quality score: {critic.get('quality_score', 'N/A')}/10")
    
    if summary:
        print(f"\n--- FULL SUMMARY ---\n")
        print(summary)
    else:
        print("\n⚠️ NO SUMMARY GENERATED!")


if __name__ == "__main__":
    main()
