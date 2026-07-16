"""
OmniRoute AI — Table Store Module
Pushes extracted tables to MongoDB as structured JSON with page metadata.
Preserves structural integrity that would be lost in plain text.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
from pymongo import MongoClient
from gridfs import GridFSBucket
from pymongo.errors import ConnectionFailure, OperationFailure

from modules.parser import ExtractedImage, ExtractedTable
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


def _get_database():
    client = _get_mongo_client()
    return client[st.secrets["MONGODB_DB_NAME"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_tables(
    tables: list[ExtractedTable],
    document_name: str,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
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
        if document_id:
            doc["document_id"] = document_id
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


def store_images(
    images: list[ExtractedImage],
    document_name: str,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
) -> int:
    """Store extracted image bytes in MongoDB GridFS with page metadata.

    GridFS avoids MongoDB's 16 MB document limit and keeps the binary asset
    separate from the table collection. Every occurrence is represented in
    metadata even when the same image bytes appear on multiple pages.
    """
    if not images:
        return 0

    database = _get_database()
    bucket_name = st.secrets.get("MONGODB_IMAGE_BUCKET", "extracted_images")
    bucket = GridFSBucket(database, bucket_name=bucket_name)
    stored = 0
    for image in images:
        metadata = {
            "document_name": document_name,
            "page_number": image.page_number,
            "image_id": image.image_id,
            "occurrence": image.occurrence,
            "mime_type": image.mime_type,
            "width": image.width,
            "height": image.height,
        }
        if session_id:
            metadata["session_id"] = session_id
        if document_id:
            metadata["document_id"] = document_id
        try:
            bucket.upload_from_stream(
                f"{document_name}-page-{image.page_number}-{image.occurrence}.{image.mime_type.split('/')[-1]}",
                image.image_bytes,
                metadata=metadata,
            )
            stored += 1
        except Exception as exc:
            logger.warning(
                f"Could not store image {image.image_id} on page {image.page_number}: {exc}"
            )
    logger.info(f"Stored {stored}/{len(images)} images in MongoDB GridFS")
    return stored


def get_tables_for_document(document_name: str) -> list[dict]:
    """Retrieve all stored tables for a given document name."""
    collection = _get_collection()
    cursor = collection.find(
        {"document_name": document_name},
        {"_id": 0},
    ).sort([("page_number", 1), ("table_index", 1)])
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
    ).sort([("page_number", 1), ("table_index", 1)])
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


def get_images_for_session(session_id: str, document_name: Optional[str] = None) -> list[dict]:
    """Return GridFS image metadata for the current session."""
    database = _get_database()
    bucket_name = st.secrets.get("MONGODB_IMAGE_BUCKET", "extracted_images")
    query = {"metadata.session_id": session_id}
    if document_name:
        query["metadata.document_name"] = document_name
    return [
        {
            "file_id": file_doc._id,
            **(file_doc.metadata or {}),
            "filename": file_doc.filename,
            "length": file_doc.length,
        }
        for file_doc in GridFSBucket(database, bucket_name=bucket_name).find(query)
    ]


def get_table_summary_text(tables: list[ExtractedTable]) -> str:
    """
    Generate a text summary of tables for inclusion in the text pipeline.
    This ensures table content is represented in the summary even though
    the structured data lives in MongoDB.
    """
    if not tables:
        return ""

    summaries = []
    for tbl_idx, tbl in enumerate(tables, 1):
        rows = tbl.table_data or []
        summaries.append(f"[Table {tbl_idx} on page {tbl.page_number}]")
        if not rows:
            summaries.append("(empty table)")
            continue
        # Keep the complete extracted matrix in the model context. MongoDB
        # retains the canonical `raw` matrix; this text representation makes
        # cell values available to the summarizer as well.
        for row in rows:
            summaries.append(" | ".join(str(cell).replace("\n", " ").strip() for cell in row))

    return "\n".join(summaries)
