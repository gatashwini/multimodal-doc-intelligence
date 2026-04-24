# 🧠 Multimodal Document Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GPT-4o](https://img.shields.io/badge/GPT--4o-Vision-412991?style=for-the-badge&logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Weaviate](https://img.shields.io/badge/Weaviate-1.24-FF6D00?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-87%25_Accuracy-4CAF50?style=for-the-badge)

**Multimodal RAG pipeline enabling Q&A over PDFs, scanned invoices, and chart-heavy reports.**  
Two-pass ingestion (OCR + GPT-4o Vision) · Hybrid semantic retrieval · Source page attribution · 87% RAGAS accuracy

[🌐 **Live Demo**](https://YOUR_USERNAME.github.io/multimodal-doc-intelligence) · [📖 API Docs](http://localhost:8000/docs) · [📊 Architecture](#-architecture)

</div>

---

## 🎯 What This Project Does

Most RAG pipelines fail on real enterprise documents because they only handle clean text.  
This system handles **three hard document types**:

| Document Type | Challenge | Solution |
|---|---|---|
| 📄 PDF Reports | Charts, tables, mixed layouts | GPT-4o page-level visual description |
| 🧾 Scanned Invoices | No selectable text, rotated | pytesseract OCR → structured extraction |
| 📊 Chart-Heavy Reports | Data locked in images | Vision model extracts chart values |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
│                                                                     │
│  PDF Input                                                          │
│     │                                                               │
│     ▼                                                               │
│  PyMuPDF → Page Images (200 DPI)                                    │
│     │                                                               │
│     ├──────────────────────────────────┐                            │
│     │                                  │                            │
│     ▼                                  ▼                            │
│  [PASS 1]                          [PASS 2]                         │
│  pytesseract OCR                   GPT-4o Vision                    │
│  Raw text + numbers               Tables, charts, diagrams          │
│     │                                  │                            │
│     └──────────────┬─────────────────┘                             │
│                    ▼                                                │
│             Combined Content                                        │
│             Smart Chunking (512t / 64t overlap)                     │
│                    │                                                │
│                    ▼                                                │
│             Weaviate Vector DB                                      │
│             (OpenAI text-embedding-3-small)                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                               │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  Hybrid Search (BM25 + Dense, α=0.6)                               │
│       │                                                             │
│       ▼                                                             │
│  Deduplicate + Rerank (top-6 chunks)                                │
│       │                                                             │
│       ▼                                                             │
│  GPT-4o Answer Generation                                           │
│  (grounded, temperature=0.1)                                        │
│       │                                                             │
│       ▼                                                             │
│  Response + Source Attribution [Source N, Page X]                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
multimodal-doc-intelligence/
├── src/
│   ├── ingestion/
│   │   └── pipeline.py          # Two-pass ingestion: OCR + GPT-4o Vision
│   ├── retrieval/
│   │   ├── retriever.py         # Hybrid BM25 + dense search
│   │   └── qa_chain.py          # RAG chain with source attribution
│   ├── api/
│   │   └── main.py              # FastAPI: /ingest, /ask, /documents, /benchmark
│   └── utils/
│       ├── config.py            # Pydantic settings
│       ├── chunker.py           # Sentence-boundary chunker
│       └── ragas_eval.py        # RAGAS benchmarking
├── demo/
│   └── index.html               # ← Interactive GitHub Pages demo
├── tests/
│   └── ragas_testset.json       # 50 Q&A pairs for evaluation
├── docker/
│   └── Dockerfile
├── docker-compose.yml           # API + Weaviate in one command
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-doc-intelligence.git
cd multimodal-doc-intelligence

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Run with Docker (recommended)

```bash
docker-compose up --build
```

This starts:
- **Weaviate** at `http://localhost:8080`
- **FastAPI** at `http://localhost:8000`
- Interactive API docs at `http://localhost:8000/docs`

### 3. Manual setup (without Docker)

```bash
# Install system deps
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Install Python deps
pip install -r requirements.txt

# Start Weaviate (requires Docker)
docker run -d -p 8080:8080 semitechnologies/weaviate:1.24.4

# Run API
uvicorn src.api.main:app --reload
```

---

## 🔌 API Usage

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@invoice.pdf" \
  -F "doc_type=invoice"
```

```json
{
  "file": "invoice.pdf",
  "pages": 3,
  "chunks": 24,
  "doc_type": "invoice",
  "latency_ms": 2840
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total invoice amount including tax?",
    "doc_type_filter": "invoice"
  }'
```

```json
{
  "question": "What is the total invoice amount including tax?",
  "answer": "The total invoice amount is $4,320.00 including 12.5% tax [Source 1, Page 2].",
  "sources": [
    {"file": "invoice.pdf", "page": 2, "score": 0.934}
  ],
  "context_chunks": 6,
  "latency_ms": 842
}
```

### List documents

```bash
curl http://localhost:8000/documents
```

### Run RAGAS benchmark

```bash
curl -X POST "http://localhost:8000/benchmark?test_file=tests/ragas_testset.json"
```

```json
{
  "ragas_scores": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.91,
    "context_precision": 0.84,
    "context_recall": 0.82,
    "overall": 0.8625
  }
}
```

---

## 📊 Evaluation Results

Benchmarked on **50 question-answer pairs** across all three document types.

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | **87%** | Answer grounded in retrieved context |
| **Answer Relevancy** | **91%** | Answer addresses the question |
| **Context Precision** | **84%** | Retrieved chunks are relevant |
| **Context Recall** | **82%** | All relevant chunks are retrieved |
| **Overall** | **86.25%** | Macro average across all metrics |

### Key Design Choices That Improved Accuracy

| Decision | Why |
|---|---|
| Two-pass ingestion (OCR + Vision) | OCR alone misses chart values; Vision alone misses printed numbers |
| Hybrid search α=0.6 | Dense embeddings for semantic questions; BM25 for specific numbers/dates |
| Low temperature (0.1) | Factual extraction needs determinism |
| Chunk size 512 / overlap 64 | Balances context window vs retrieval granularity for financial tables |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Vision Model** | GPT-4o | Page-level visual description |
| **OCR** | pytesseract | Text extraction from scanned pages |
| **PDF Processing** | PyMuPDF (fitz) | High-quality page rendering |
| **Vector DB** | Weaviate 1.24 | Hybrid BM25 + dense search |
| **Embeddings** | text-embedding-3-small | Dense vector representation |
| **API** | FastAPI + uvicorn | REST API with OpenAPI docs |
| **Evaluation** | RAGAS | RAG-specific benchmark metrics |
| **Deployment** | Docker Compose | One-command setup |

---

## 🔭 Future Improvements

- [ ] Multi-document cross-referencing (e.g., compare two invoices)
- [ ] Table-aware chunking (preserve row/column relationships)
- [ ] Streaming API responses for faster perceived latency
- [ ] Support for XLSX and DOCX input formats
- [ ] Self-hosted embedding model to reduce API costs

---

## 📋 Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...          # Required: GPT-4o + embeddings
WEAVIATE_HOST=localhost         # Default: localhost
WEAVIATE_PORT=8080             # Default: 8080
```

---

<div align="center">
Built as part of personal AI engineering R&D · January 2026
</div>
