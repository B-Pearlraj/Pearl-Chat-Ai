# Pearl.Ai - AI chat with document Q&A

- `app.py` - Streamlit UI (chat + Documents panel in the sidebar)
- `main.py` - Deep Agent + FastAPI endpoints
- `database.py` - Neon connection and schema (pgvector)
- `document_processor.py` - PDF/DOCX/TXT/MD/CSV text extraction and chunking
- `vector_store.py` - embeddings, storage, semantic search

## Setup
1. Neon SQL Editor: `CREATE EXTENSION IF NOT EXISTS vector;`
2. Copy `.env.example` to `.env` and fill in the keys.
3. `pip install -r requirements.txt`
4. `streamlit run app.py` (tables are created automatically on first use)

## API (optional, `uvicorn main:app --reload`)
- `POST /documents/upload`
- `GET /documents`
- `DELETE /documents/{id}`
- `GET /documents/search?query=...`
