"""
RAG Q&A Chain
--------------
Takes a user question, retrieves relevant chunks via hybrid search,
and generates a grounded answer with source page attribution.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI

from src.retrieval.retriever import hybrid_search, deduplicate_chunks, build_context, RetrievedChunk
from src.utils.config import settings

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a precise document analyst with expertise in invoices, financial reports, and chart-heavy documents.

Rules:
1. Answer ONLY from the provided context. Never hallucinate.
2. Cite sources using the format [Source N, Page X] inline.
3. If a table or chart is described, extract exact values when asked.
4. If the answer cannot be found, say: "I could not find this information in the provided documents."
5. Be concise but complete. Prefer bullet points for multi-part answers."""


@dataclass
class QAResponse:
    answer: str
    sources: list[dict]          # [{file, page, score}]
    context_chunks: int
    model: str = "gpt-4o"


def answer_question(
    question: str,
    top_k: int = 6,
    doc_type_filter: str | None = None,
) -> QAResponse:
    """
    Full RAG pipeline: retrieve → deduplicate → generate → attribute.
    """
    # 1. Retrieve
    chunks = hybrid_search(question, top_k=top_k, doc_type_filter=doc_type_filter)
    chunks = deduplicate_chunks(chunks)

    if not chunks:
        return QAResponse(
            answer="No relevant documents found. Please ingest documents first.",
            sources=[],
            context_chunks=0,
        )

    # 2. Build context
    context = build_context(chunks)

    # 3. Generate answer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000,
        temperature=0.1,    # low temp for factual extraction
    )

    answer = response.choices[0].message.content.strip()

    # 4. Build source attribution
    sources = [
        {
            "file": c.source_file,
            "page": c.page_number,
            "doc_type": c.doc_type,
            "score": round(c.score, 4),
            "has_visual": c.has_visual,
        }
        for c in chunks
    ]

    return QAResponse(
        answer=answer,
        sources=sources,
        context_chunks=len(chunks),
    )
