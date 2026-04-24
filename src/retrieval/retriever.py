"""
Hybrid Semantic Retrieval
--------------------------
Combines dense vector search (OpenAI embeddings via Weaviate)
with BM25 keyword search for robust retrieval across:
  • Text-heavy documents
  • Scanned invoices (OCR-dominant)
  • Chart/table-heavy reports (vision-dominant)
"""

import logging
from dataclasses import dataclass
import weaviate.auth
from src.utils.config import settings
import weaviate
import weaviate.classes as wvc

from src.utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    page_number: int
    chunk_index: int
    doc_type: str
    score: float
    has_visual: bool


def get_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": settings.OPENAI_API_KEY}
    )

def hybrid_search(
    query: str,
    top_k: int = 6,
    alpha: float = 0.6,          # 0 = pure BM25, 1 = pure vector
    doc_type_filter: str | None = None,
) -> list[RetrievedChunk]:
    """
    Hybrid search combining BM25 + dense embeddings.
    alpha=0.6 favors semantic similarity while preserving keyword signal.
    """
    client = get_client()
    collection = client.collections.get("DocumentChunk")

    filters = None
    if doc_type_filter:
        filters = wvc.query.Filter.by_property("doc_type").equal(doc_type_filter)

    results = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=top_k,
        filters=filters,
        return_metadata=wvc.query.MetadataQuery(score=True),
        return_properties=[
            "content", "source_file", "page_number",
            "chunk_index", "doc_type", "has_visual",
        ],
    )

    chunks = []
    for obj in results.objects:
        p = obj.properties
        chunks.append(RetrievedChunk(
            content=p["content"],
            source_file=p["source_file"],
            page_number=p["page_number"],
            chunk_index=p["chunk_index"],
            doc_type=p.get("doc_type", "generic"),
            score=obj.metadata.score or 0.0,
            has_visual=p.get("has_visual", False),
        ))

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}'")
    return chunks


def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove near-duplicate chunks (same file + page + adjacent chunk_index)."""
    seen: set[tuple[str, int, int]] = set()
    deduped = []
    for chunk in chunks:
        key = (chunk.source_file, chunk.page_number, chunk.chunk_index)
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)
    return deduped


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a structured context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk.source_file}, Page {chunk.page_number}]\n{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)
