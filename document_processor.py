"""
document_processor.py
---------------------
Turns an uploaded file (bytes) into page-aware text chunks.

Supported: PDF, DOCX, TXT, MD, CSV
"""

import io
import os
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def get_extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def _extract_pdf(data: bytes) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def _extract_docx(data: bytes) -> List[Tuple[int, str]]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    # Include table contents as well
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts).strip()
    # DOCX has no fixed pages, so page number is None
    return [(None, text)] if text else []


def _extract_text(data: bytes) -> List[Tuple[int, str]]:
    text = data.decode("utf-8", errors="replace").strip()
    return [(None, text)] if text else []


def extract_pages(filename: str, data: bytes) -> List[Tuple[int, str]]:
    """Return a list of (page_number | None, text)."""
    ext = get_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        pages = _extract_pdf(data)
    elif ext == ".docx":
        pages = _extract_docx(data)
    else:
        pages = _extract_text(data)

    if not pages:
        raise ValueError(
            "No readable text was found in this file. "
            "If it is a scanned PDF (images only), it needs OCR first."
        )
    return pages


def build_chunks(filename: str, data: bytes) -> Tuple[List[dict], int]:
    """
    Returns (chunks, page_count).
    Each chunk: {"chunk_index": int, "page_number": int | None, "content": str}
    """
    pages = extract_pages(filename, data)

    chunks = []
    index = 0
    for page_number, text in pages:
        for piece in _splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_index": index,
                    "page_number": page_number,
                    "content": piece,
                }
            )
            index += 1

    if not chunks:
        raise ValueError("The document produced no text chunks.")

    page_count = max((p for p, _ in pages if p is not None), default=len(pages))
    return chunks, page_count
