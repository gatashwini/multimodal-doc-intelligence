"""
FastAPI Backend
----------------
Endpoints:
  POST /ingest          — Upload and ingest a PDF
  POST /ask             — Q&A over ingested documents
  GET  /documents       — List ingested documents
  GET  /health          — Health check
  POST /benchmark       — Run RAGAS evaluation on a test set
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import logging
import time
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.pipeline import ingest_pdf
from src.retrieval.qa_chain import answer_question
from src.utils.config import settings
from src.utils.ragas_eval import run_ragas_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Document Intelligence API",
    description="RAG pipeline over PDFs, invoices, and chart-heavy reports using GPT-4o Vision + Weaviate",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
# Serve the custom UI
@app.get("/ui", include_in_schema=False)
def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    return FileResponse(ui_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: int = 6
    doc_type_filter: Literal["invoice", "report", "generic"] | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    context_chunks: int
    latency_ms: float


class IngestResponse(BaseModel):
    file: str
    pages: int
    chunks: int
    doc_type: str
    latency_ms: float


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form("generic"),
):
    """Upload a PDF and run two-pass ingestion (OCR + GPT-4o Vision)."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Save file temporarily
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    t0 = time.time()
    try:
        stats = ingest_pdf(str(dest), doc_type=doc_type)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion error: {str(e)}")
    finally:
        dest.unlink(missing_ok=True)   # clean up temp file

    return IngestResponse(
        **stats,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question over all ingested documents."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    t0 = time.time()
    result = answer_question(
        question=req.question,
        top_k=req.top_k,
        doc_type_filter=req.doc_type_filter,
    )

    return AskResponse(
        question=req.question,
        answer=result.answer,
        sources=result.sources,
        context_chunks=result.context_chunks,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/documents")
def list_documents():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": settings.OPENAI_API_KEY}
    )
    try:
        col = client.collections.get("DocumentChunk")
        
        # Get all objects and group manually
        result = col.query.fetch_objects(
            limit=1000,
            return_properties=["source_file", "doc_type"]
        )
        
        # Group by source file manually
        docs = {}
        for obj in result.objects:
            fname = obj.properties.get("source_file", "unknown")
            if fname not in docs:
                docs[fname] = {
                    "file": fname,
                    "chunks": 0,
                    "doc_type": obj.properties.get("doc_type", "generic")
                }
            docs[fname]["chunks"] += 1
        
        doc_list = list(docs.values())
        return {"documents": doc_list, "total": len(doc_list)}
    except Exception as e:
        return {"documents": [], "total": 0, "error": str(e)}
    finally:
        client.close()


@app.post("/benchmark")
def benchmark(test_file: str = "tests/ragas_testset.json"):
    """Run RAGAS evaluation. Provide a JSON file with question/answer/context triples."""
    scores = run_ragas_benchmark(test_file)
    return {"ragas_scores": scores}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
