"""
database.py
-----------
Neon (PostgreSQL + pgvector) connection and schema setup.

Neon scales to zero when idle, so a long-lived connection can go stale.
To stay safe we open a short-lived connection for every operation
(use Neon's *pooled* connection string, the host containing "-pooler").

Required .env value:
    NEON_DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx-pooler.REGION.aws.neon.tech/neondb?sslmode=require
"""

import os
import threading
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Must match the output size used in vector_store.py
EMBEDDING_DIM = 768

_schema_ready = False
_schema_lock = threading.Lock()


def _get_url() -> str:
    url = os.getenv("NEON_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "NEON_DATABASE_URL is missing from the .env file. "
            "Add your Neon connection string to enable document upload."
        )
    return url


@contextmanager
def get_conn():
    """Open a connection, commit on success, roll back on error, always close."""
    conn = psycopg2.connect(_get_url(), connect_timeout=20)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    filename      TEXT        NOT NULL,
    content_type  TEXT,
    file_size     BIGINT,
    file_hash     TEXT        NOT NULL UNIQUE,
    file_data     BYTEA,
    page_count    INTEGER,
    chunk_count   INTEGER     NOT NULL DEFAULT 0,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    page_number  INTEGER,
    content      TEXT    NOT NULL,
    embedding    vector({EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""


def ensure_schema() -> None:
    """Create the extension, tables and indexes once per process."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        _schema_ready = True


def is_configured() -> bool:
    return bool(os.getenv("NEON_DATABASE_URL"))
