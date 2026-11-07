"""
vector_store.py
---------------
Service layer for the document feature:

    add_document()      -> extract, chunk, embed, store in Neon
    search()            -> semantic search over all stored chunks
    list_documents()    -> documents currently stored
    delete_document()   -> remove a document and its vectors

Embeddings use the Gemini embedding model through the same GEMINI_API_KEY
you already use for chat (no extra package needed, plain HTTPS via httpx).
"""

import hashlib
import os
import time
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from database import EMBEDDING_DIM, ensure_schema, get_conn
from document_processor import build_chunks

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBEDDING_MODEL}:batchEmbedContents"
)

MAX_FILE_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))  # cosine similarity, 0..1
EMBED_BATCH = 50

# One shared HTTP client: the connection to Google stays open between
# questions, so each search skips a new TLS handshake.
_http = httpx.Client(timeout=60)


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------

def _embed(texts: List[str], task_type: str) -> List[List[float]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the .env file")

    vectors: List[List[float]] = []

    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": EMBEDDING_DIM,
                }
                for t in batch
            ]
        }

        last_error = None
        for attempt in range(4):
            try:
                resp = _http.post(
                    EMBED_URL,
                    json=payload,
                    headers={"x-goog-api-key": api_key},
                )
                if resp.status_code in (429, 500, 503):
                    last_error = f"HTTP {resp.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                vectors.extend(e["values"] for e in resp.json()["embeddings"])
                break
            except httpx.HTTPError as exc:
                last_error = str(exc)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Embedding request failed: {last_error}")

    return vectors


def _to_pgvector(values: List[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


# ------------------------------------------------------------------
# Add / list / delete
# ------------------------------------------------------------------

def add_document(filename: str, data: bytes, content_type: Optional[str] = None) -> dict:
    """
    Store a document. Returns a dict with status "created" or "duplicate".
    Raises ValueError for user-fixable problems (type, size, empty file).
    """
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_FILE_MB} MB.")

    ensure_schema()

    file_hash = hashlib.sha256(data).hexdigest()

    # Same file uploaded before? Skip re-embedding (saves API calls + DB space).
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, chunk_count FROM documents WHERE file_hash = %s",
            (file_hash,),
        )
        row = cur.fetchone()
    if row:
        return {
            "status": "duplicate",
            "id": row[0],
            "filename": row[1],
            "chunk_count": row[2],
        }

    chunks, page_count = build_chunks(filename, data)
    embeddings = _embed([c["content"] for c in chunks], "RETRIEVAL_DOCUMENT")

    # One transaction: a document is either fully stored or not stored at all.
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (filename, content_type, file_size, file_hash, file_data,
                 page_count, chunk_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (filename, content_type, len(data), file_hash, data,
             page_count, len(chunks)),
        )
        doc_id = cur.fetchone()[0]

        execute_values(
            cur,
            """
            INSERT INTO document_chunks
                (document_id, chunk_index, page_number, content, embedding)
            VALUES %s
            """,
            [
                (doc_id, c["chunk_index"], c["page_number"], c["content"],
                 _to_pgvector(vec))
                for c, vec in zip(chunks, embeddings)
            ],
            template="(%s, %s, %s, %s, %s::vector)",
        )

    return {
        "status": "created",
        "id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "page_count": page_count,
    }


def list_documents() -> List[dict]:
    ensure_schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, file_size, page_count, chunk_count, uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC
            """
        )
        return [
            {
                "id": r[0],
                "filename": r[1],
                "file_size": r[2],
                "page_count": r[3],
                "chunk_count": r[4],
                "uploaded_at": r[5].isoformat(),
            }
            for r in cur.fetchall()
        ]


def delete_document(doc_id: int) -> bool:
    ensure_schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return cur.rowcount > 0


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

def search(query: str, top_k: int = 5, document_ids: Optional[List[int]] = None) -> List[dict]:
    """Return the most relevant chunks for a question."""
    query = (query or "").strip()
    if not query:
        return []

    ensure_schema()
    query_vec = _to_pgvector(_embed([query], "RETRIEVAL_QUERY")[0])

    sql = """
        SELECT d.id, d.filename, c.page_number, c.content,
               1 - (c.embedding <=> %s::vector) AS score
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    params: list = [query_vec]
    where = ""
    if document_ids:
        where = "WHERE d.id = ANY(%s)"
        params.append(document_ids)
    params.extend([query_vec, top_k])

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql.format(where=where), params)
        rows = cur.fetchall()

    return [
        {
            "document_id": r[0],
            "filename": r[1],
            "page": r[2],
            "content": r[3],
            "score": round(float(r[4]), 4),
        }
        for r in rows
        if float(r[4]) >= MIN_SCORE
    ]
