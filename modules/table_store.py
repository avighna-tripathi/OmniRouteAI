"""
OmniRoute AI — Table Store Module
Pushes extracted tables to MongoDB as structured JSON with page metadata.
Preserves structural integrity that would be lost in plain text.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from modules.parser import ExtractedTable
from modules.utils import logger


# ---------------------------------------------------------------------------
# MongoDB connection (cached per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_mongo_client() -> MongoClient:
    """Create a cached MongoDB client from st.secrets."""
    uri = st.secrets["MONGODB_URI"]
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    # Verify connection
    try:
        client.admin.command("ping")
        logger.info("MongoDB connection established.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    return client


def _get_collection():
    """Get the target MongoDB collection."""
    client = _get_mongo_client()
    db_name = st.secrets["MONGODB_DB_NAME"]
    coll_name = st.secrets["MONGODB_COLLECTION"]
    return client[db_name][coll_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_tables(
    tables: list[ExtractedTable],
    document_name: str,
    session_id: Optional[str] = None,
) -> int:
    """
    Store all extracted tables to MongoDB.

    Each table is stored as a document with:
    - page_number: source page
    - source_file: original filename
    - headers: column headers
    - rows: list of {header: value} dicts
    - raw: original list-of-lists
    - document_name: the uploaded file name
    - session_id: optional Streamlit session identifier

    Returns:
        Number of tables successfully inserted.
    """
    if not tables:
        logger.info("No tables to store.")
        return 0

    collection = _get_collection()

    documents = []
    for table in tables:
        doc = table.to_mongo_dict()
        doc["document_name"] = document_name
        if session_id:
            doc["session_id"] = session_id
        documents.append(doc)

    try:
        result = collection.insert_many(documents, ordered=False)
        count = len(result.inserted_ids)
        logger.info(f"Stored {count} tables to MongoDB (collection: {collection.name})")
        return count
    except OperationFailure as e:
        logger.error(f"MongoDB insert failed: {e}")
        raise


def get_tables_for_document(document_name: str) -> list[dict]:
    """Retrieve all stored tables for a given document name."""
    collection = _get_collection()
    cursor = collection.find(
        {"document_name": document_name},
        {"_id": 0},
    ).sort("page_number", 1)
    return list(cursor)


def get_tables_for_session(session_id: str, document_name: str = None) -> list[dict]:
    """
    Retrieve tables scoped to a specific session token.
    This ensures user isolation — each user only sees their own tables.

    Args:
        session_id: The user's unique session token.
        document_name: Optional filter by document name.

    Returns:
        List of table documents belonging to this session.
    """
    collection = _get_collection()
    query = {"session_id": session_id}
    if document_name:
        query["document_name"] = document_name

    cursor = collection.find(
        query,
        {"_id": 0},
    ).sort("page_number", 1)
    return list(cursor)


def get_document_names_for_session(session_id: str) -> list[str]:
    """
    Get all unique document names uploaded by a specific session.
    Used to populate the table viewer's document selector.
    """
    collection = _get_collection()
    return collection.distinct("document_name", {"session_id": session_id})


def delete_tables_for_document_session(session_id: str, document_name: str) -> int:
    """
    Delete all tables for a specific document within a session.
    Returns the number of documents deleted.
    """
    collection = _get_collection()
    result = collection.delete_many({
        "session_id": session_id,
        "document_name": document_name,
    })
    logger.info(f"Deleted {result.deleted_count} tables for '{document_name}' in session {session_id}")
    return result.deleted_count


def get_table_summary_text(tables: list[ExtractedTable]) -> str:
    """
    Generate a text summary of tables for inclusion in the text pipeline.
    This ensures table content is represented in the summary even though
    the structured data lives in MongoDB.
    """
    if not tables:
        return ""

    summaries = []
    for tbl in tables:
        headers = tbl.table_data[0] if tbl.table_data else []
        num_rows = max(len(tbl.table_data) - 1, 0)
        header_str = ", ".join(headers) if headers else "no headers"
        summaries.append(
            f"[Table on page {tbl.page_number}: {num_rows} rows, "
            f"columns: {header_str}]"
        )

    return "\n".join(summaries)

