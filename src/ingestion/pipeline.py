"""
Two-Pass Document Ingestion Pipeline
-------------------------------------
Pass 1: pytesseract OCR  → extracts raw text from each page
Pass 2: GPT-4o Vision   → generates semantic page description (tables, charts, diagrams)
Both signals are stored as hybrid vectors in Weaviate for retrieval.
"""
# Add this at the top of pipeline.py — Windows only
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import base64
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytesseract

import weaviate.auth
from src.utils.config import settings
import weaviate
from openai import OpenAI
from PIL import Image

from src.utils.config import settings
from src.utils.chunker import smart_chunk_text

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)


# ─────────────────────────────────────────────
# 1. PDF → per-page images
# ─────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[tuple[int, Image.Image]]:
    """Convert every page of a PDF to a PIL Image."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append((page_num + 1, img))
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages


# ─────────────────────────────────────────────
# 2. Pass 1 — pytesseract OCR
# ─────────────────────────────────────────────

def ocr_page(image: Image.Image) -> str:
    """Extract raw text from a page image via Tesseract."""
    text = pytesseract.image_to_string(image, config="--psm 6")
    return text.strip()


# ─────────────────────────────────────────────
# 3. Pass 2 — GPT-4o Visual Description
# ─────────────────────────────────────────────

VISION_PROMPT = """You are a document analyst. Carefully examine this page and extract ALL information.

Provide a structured description covering:
1. Page type (invoice / chart / table / text / mixed / form / letter)
2. ALL text content visible on the page — headings, paragraphs, labels, values
3. ALL numbers, dates, names, amounts, percentages visible
4. Table structure if present — column headers and every row of data
5. Chart or diagram description — type, axes, all data points and values
6. Key entities — company names, person names, addresses, IDs, codes
7. Layout description — single column, multi-column, form layout etc.

Be exhaustive. Extract every piece of information visible on the page.
Do NOT summarize or skip any data. Every number and name matters."""


def vision_describe_page(image: Image.Image) -> str:
    """Send page image to GPT-4o and return structured visual description."""
    # Encode image to base64 PNG
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 4. Hybrid ingestion → Weaviate
# ─────────────────────────────────────────────

def get_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": settings.OPENAI_API_KEY}
    )

def ensure_schema(client: weaviate.Client):
    """Create Weaviate collection if it doesn't exist."""
    collection_name = "DocumentChunk"
    existing = [c.name for c in client.collections.list_all().values()]
    if collection_name not in existing:
        client.collections.create(
            name=collection_name,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small"
            ),
            generative_config=weaviate.classes.config.Configure.Generative.openai(
                model="gpt-4o"
            ),
        )
        logger.info(f"Created Weaviate collection: {collection_name}")


def ingest_pdf(pdf_path: str, doc_type: str = "generic") -> dict[str, Any]:
    """
    Full ingestion pipeline for a single PDF.
    Returns ingestion statistics.
    """
    path = Path(pdf_path)
    assert path.exists(), f"File not found: {pdf_path}"

    client = get_weaviate_client()
    ensure_schema(client)
    collection = client.collections.get("DocumentChunk")

    pages = pdf_to_images(pdf_path)
    total_chunks = 0

    for page_num, image in pages:
        logger.info(f"  Processing page {page_num}/{len(pages)}")

        # Pass 1: OCR
        ocr_text = ocr_page(image)

        # Pass 2: Vision
        visual_desc = vision_describe_page(image)

        # Combine both signals
        combined_content = f"[OCR TEXT]\n{ocr_text}\n\n[VISUAL DESCRIPTION]\n{visual_desc}"

        # Smart chunk the combined content
        chunks = smart_chunk_text(combined_content, chunk_size=512, overlap=64)

        for chunk_idx, chunk_text in enumerate(chunks):
            collection.data.insert({
                "content": chunk_text,
                "source_file": path.name,
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "doc_type": doc_type,
                "has_visual": bool(visual_desc),
                "ocr_text": ocr_text[:500],           # preview only
                "visual_description": visual_desc[:500],
            })
            total_chunks += 1

    logger.info(f"Ingested {path.name}: {len(pages)} pages → {total_chunks} chunks")
    return {
        "file": path.name,
        "pages": len(pages),
        "chunks": total_chunks,
        "doc_type": doc_type,
    }
