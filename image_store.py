"""
image_store.py
--------------
Saves every uploaded / generated / edited image in Neon (table `images`)
so nothing is lost when the Streamlit session restarts.

    save_image()      -> store one image, returns its id
    get_image()       -> load bytes back
    list_images()     -> gallery for ONE visitor (never other visitors' images)
    delete_image()    -> remove an image

The table is created automatically by database.ensure_schema().
Storing is best-effort: if the database is down or not configured,
the chat still works and the image simply is not saved.
"""

import hashlib
from typing import List, Optional

from database import ensure_schema, get_conn, is_configured


def save_image(
    session_id: str,
    kind: str,                      # "upload" | "generated" | "edited"
    data: bytes,
    mime: str,
    prompt: Optional[str] = None,
    filename: Optional[str] = None,
    parent_id: Optional[int] = None,
) -> Optional[int]:
    """Returns the new image id, or None if saving was not possible."""
    if not is_configured() or not data:
        return None
    try:
        ensure_schema()
        digest = hashlib.sha256(data).hexdigest()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO images
                    (session_id, kind, filename, mime_type, file_size,
                     file_hash, file_data, prompt, parent_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, kind, filename, mime, len(data), digest,
                 data, (prompt or "")[:4000], parent_id),
            )
            return cur.fetchone()[0]
    except Exception as exc:
        print(f"[images] could not save image: {exc}")
        return None


def get_image(image_id: int, session_id: str) -> Optional[dict]:
    try:
        ensure_schema()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, kind, filename, mime_type, file_data, prompt, created_at
                FROM images WHERE id = %s AND session_id = %s
                """,
                (image_id, session_id),
            )
            r = cur.fetchone()
    except Exception:
        return None
    if not r:
        return None
    return {
        "id": r[0], "kind": r[1], "filename": r[2], "mime": r[3],
        "data": bytes(r[4]), "prompt": r[5], "created_at": r[6],
    }


def list_images(session_id: str, limit: int = 30) -> List[dict]:
    """Newest first. Does not load the image bytes (fast)."""
    try:
        ensure_schema()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, kind, filename, mime_type, file_size, prompt, created_at
                FROM images WHERE session_id = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
    except Exception:
        return []
    return [
        {"id": r[0], "kind": r[1], "filename": r[2], "mime": r[3],
         "file_size": r[4], "prompt": r[5], "created_at": r[6]}
        for r in rows
    ]


def delete_image(image_id: int, session_id: str) -> bool:
    try:
        ensure_schema()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM images WHERE id = %s AND session_id = %s",
                (image_id, session_id),
            )
            return cur.rowcount > 0
    except Exception:
        return False
