# CLAUDE.md — Production RAG Application
# "Ask My Docs" — Domain-Specific Retrieval-Augmented Generation System

## Project Overview

A production-grade, domain-specific RAG system featuring hybrid retrieval (BM25 + dense
vector search), cross-encoder reranking, citation enforcement, and a CI-gated evaluation
pipeline. This is the most common pattern in enterprise AI and the gold standard for
portfolio-ready RAG work.

---

## Tech Stack

| Layer              | Tool / Library                                          | Status      |
|--------------------|----------------------------------------------------------|-------------|
| Language           | Python 3.11+                                             | ✅ Done     |
| Orchestration      | LangChain (LCEL chains)                                  | 🔄 Partial  |
| LLM                | Claude (claude-sonnet-4-6) via Anthropic SDK             | 🔄 Partial  |
| Embeddings         | BGE-large-en-v1.5 via sentence-transformers (local)      | ✅ Done     |
| Vector Store       | Qdrant (local Docker for dev, cloud for prod)            | ✅ Done     |
| Sparse Retrieval   | BM25 via rank_bm25                                       | ✅ Done     |
| Reranker           | BGE-reranker-v2-m3 via sentence-transformers (local)     | ✅ Done     |
| API Layer          | FastAPI + uvicorn                                        | ✅ Done     |
| Evaluation         | RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) | ⏳ Pending |
| Observability      | Langfuse (tracing) + LangSmith (prompt versioning)       | ⏳ Pending  |
| Config / Secrets   | pydantic-settings + python-dotenv (.env)                 | ✅ Done     |
| Testing            | pytest + pytest-asyncio                                  | ✅ Done     |
| CI/CD              | GitHub Actions                                           | ✅ Done     |
| Containerization   | Docker + docker-compose                                  | ✅ Done     |

> **Note:** OpenAI and Cohere API keys are **not required**. Embeddings and reranking
> run locally using BAAI open-source models.

---

## Project Structure

```
production-rag/
├── CLAUDE.md                   # ← You are here
├── pyproject.toml              # ✅ All deps declared
├── .env.example                # ✅ All env vars documented
├── docker-compose.yml          # ✅ Qdrant + app services
├── Dockerfile                  # ✅ Multi-stage build
│
├── app/
│   ├── main.py                 # ✅ FastAPI entrypoint + /health
│   ├── config.py               # ✅ pydantic-settings — all env vars
│   ├── exceptions.py           # ✅ Custom exception hierarchy
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py        # 🔄 Stub — needs generation wired up
│   │   │   └── ingest.py       # ✅ POST /ingest → run_ingestion()
│   │   └── schemas.py          # ✅ QueryRequest/Response, IngestRequest/Response
│   │
│   ├── ingestion/
│   │   ├── loader.py           # ✅ PDF, HTML, Markdown — file, directory, URL
│   │   ├── chunker.py          # ✅ RecursiveCharacterTextSplitter
│   │   ├── embedder.py         # ✅ BGE-large-en-v1.5 (local, no API key)
│   │   └── pipeline.py         # ✅ load → chunk → embed → Qdrant + BM25
│   │
│   ├── retrieval/
│   │   ├── vector_store.py     # ✅ AsyncQdrantClient — upsert + similarity_search
│   │   ├── bm25_store.py       # ✅ BM25Okapi — add + search + disk persistence
│   │   ├── hybrid.py           # ✅ Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py         # ✅ BGE-reranker-v2-m3 (local, no API key)
│   │   └── pipeline.py         # ✅ embed → dense+sparse → RRF → rerank
│   │
│   ├── generation/
│   │   ├── chain.py            # 🔄 LCEL chain skeleton — needs wiring
│   │   ├── prompts.py          # ✅ System + user prompt templates
│   │   └── citations.py        # ✅ Citation extraction & validation
│   │
│   └── observability/
│       ├── langfuse_client.py  # ✅ Langfuse handler (optional, skips if no key)
│       └── metrics.py          # ✅ Latency + cost tracking helpers
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py           # ✅ 3 tests
│   │   ├── test_citations.py         # ✅ 6 tests
│   │   ├── test_hybrid_retrieval.py  # ✅ 3 tests
│   │   ├── test_ingestion_pipeline.py# ✅ 5 tests
│   │   ├── test_loader.py            # ✅ 5 tests
│   │   └── test_reranker.py          # ✅ 5 tests
│   ├── integration/
│   │   └── test_rag_pipeline.py      # ⏳ Placeholder
│   └── eval/
│       ├── eval_dataset.json         # ⏳ Sample only — needs real Q&A pairs
│       └── run_ragas_eval.py         # ⏳ Scaffold — needs pipeline wired
│
└── .github/
    └── workflows/
        └── ci.yml              # ✅ lint → unit → integration → RAGAS gate
```

---

## Implementation Status

### ✅ Completed

#### Ingestion Pipeline (`app/ingestion/`)
- **`loader.py`** — detects source type (file, directory, URL); directory mode
  recursively finds all supported files and loads them concurrently via
  `asyncio.gather`; skips failed files gracefully
- **`chunker.py`** — `RecursiveCharacterTextSplitter` with configurable
  `chunk_size` (512) and `chunk_overlap` (50); prefers paragraph → sentence breaks
- **`embedder.py`** — `BAAI/bge-large-en-v1.5` via `sentence-transformers`;
  model cached as singleton (`@lru_cache`); encoding offloaded to thread-pool
  executor; `normalize_embeddings=True` for cosine similarity
- **`pipeline.py`** — full orchestration: load → chunk → embed → store;
  assigns `chunk_id` (UUID), `source`, `page`, `ingested_at` per chunk;
  upserts to Qdrant and BM25 concurrently via `asyncio.gather`

#### Retrieval Pipeline (`app/retrieval/`)
- **`vector_store.py`** — `AsyncQdrantClient`; `ensure_collection` creates
  collection on first use (1024-dim, cosine); `upsert_chunks` + `similarity_search`
- **`bm25_store.py`** — `BM25Okapi` in-memory singleton; `asyncio.Lock` for
  safe concurrent writes; persisted to `data/bm25_index.pkl` and auto-loaded
  at import time
- **`hybrid.py`** — Reciprocal Rank Fusion (RRF, k=60) merges dense and sparse
  ranked lists; deduplicates by `chunk_id`
- **`reranker.py`** — `BAAI/bge-reranker-v2-m3` CrossEncoder; singleton via
  `@lru_cache`; scoring in thread-pool executor; adds `rerank_score` to each chunk
- **`pipeline.py`** — full orchestration: embed query + BM25 search in parallel
  → RRF fusion → reranker → top-5 chunks

#### API & Config
- **`app/main.py`** — FastAPI app with `/health`, `/ingest`, `/query` routes
- **`app/config.py`** — all settings via `pydantic-settings`; no required API
  keys except `ANTHROPIC_API_KEY`
- **`app/exceptions.py`** — custom exception hierarchy
- **`app/api/schemas.py`** — typed request/response models

#### Generation (partial)
- **`prompts.py`** — system + user prompt templates enforcing citation format
- **`citations.py`** — `extract_cited_ids` + `validate_citations` (UUID pattern matching)
- **`chain.py`** — LCEL chain skeleton wired to Claude; needs retrieval connected

#### Infrastructure
- **`docker-compose.yml`** — Qdrant on 6333/6334 with health check
- **`Dockerfile`** — multi-stage build, non-root user
- **`pyproject.toml`** — all deps declared; `sentence-transformers`, `torch`,
  `qdrant-client`, `rank-bm25`, `fastapi`, `langchain-anthropic`, etc.
- **`.github/workflows/ci.yml`** — 4-stage CI pipeline

#### Tests (27 passing)
| File | Tests |
|------|-------|
| `test_chunker.py` | 3 |
| `test_citations.py` | 6 |
| `test_hybrid_retrieval.py` | 3 |
| `test_ingestion_pipeline.py` | 5 |
| `test_loader.py` | 5 |
| `test_reranker.py` | 5 |

---

### ⏳ Remaining Work

1. **Generation pipeline** — wire `retrieval/pipeline.py` output into `generation/chain.py`;
   implement citation retry logic in `query.py`
2. **Query route** — `app/api/routes/query.py` currently raises `NotImplementedError`
3. **Observability** — integrate Langfuse callback into the RAG chain; wire token cost logging
4. **RAGAS evaluation** — populate `eval_dataset.json` with real Q&A pairs; implement
   `run_ragas_eval.py` to call `ragas.evaluate()`
5. **Integration tests** — replace placeholder in `test_rag_pipeline.py` with real
   Qdrant-backed tests
6. **Lint / type-check** — fix outstanding `ruff` and `mypy` issues before CI passes

---

## Core Architectural Decisions

### 1. Hybrid Retrieval (BM25 + Dense Vector)
- Dense retrieval handles semantic similarity; BM25 handles exact keyword matching.
- Results are fused using **Reciprocal Rank Fusion (RRF)** before reranking.
- Never use dense-only retrieval in production — keyword queries degrade badly.

### 2. Local Models — No API Keys for Embeddings or Reranking
- **Embedder:** `BAAI/bge-large-en-v1.5` (~1.3 GB, downloads once to `~/.cache/huggingface/`)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (~568 MB, same cache)
- Both run on CPU by default; set `EMBEDDING_DEVICE=mps` or `RERANKER_DEVICE=cuda` for GPU.
- CPU-bound inference is offloaded to `asyncio.run_in_executor` to keep the event loop free.

### 3. Cross-Encoder Reranking
- After hybrid fusion, top-20 candidates are scored by the cross-encoder to get final top-5.
- Reranking is the single highest-leverage improvement in a RAG pipeline.

### 4. Citation Enforcement
- Every LLM response **must** include inline citations referencing specific retrieved chunks.
- The system prompt enforces a structured citation format `[chunk_id]`.
- Post-generation, `citations.py` validates that all cited chunk IDs exist in the context.
- Uncited claims should trigger a retry or a structured fallback.

### 5. Async Throughout
- All I/O-bound operations (vector DB queries, LLM calls) are `async`.
- CPU-bound operations (embedding, reranking) run in thread-pool executors.
- `asyncio.gather()` used for parallel operations (embed + BM25 search, Qdrant + BM25 upsert).

### 6. Evaluation-First Development
- Every retrieval change must show metric delta in RAGAS before merging.
- Tracked metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- Eval runs are gated in CI — PRs that regress metrics by >5% are blocked.

---

## Chunking Strategy

```python
# Default chunking config — tune per domain
CHUNK_SIZE = 512          # tokens
CHUNK_OVERLAP = 50        # tokens
SEPARATORS = ["\n\n", "\n", ". ", " "]  # Prefer paragraph → sentence → word breaks
```

Metadata stored per chunk:
- `source` (filename / URL)
- `page` or `section`
- `chunk_id` (UUID4)
- `ingested_at` (ISO 8601 UTC timestamp)

---

## Prompt Templates

### System Prompt (enforces grounding + citation)
```
You are a precise, grounded Q&A assistant. Answer ONLY using the provided context.
For every claim you make, cite the source chunk using [chunk_id].
If the answer is not in the context, reply: "I don't have enough information to answer this."
Do NOT use prior knowledge outside the provided context.
```

### User Prompt Template
```
Context:
{context}

Question: {question}

Answer with citations:
```

---

## Environment Variables

```bash
# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── Embeddings (local — no API key required) ──────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu          # cpu | cuda | mps

# ── Reranker (local — no API key required) ────────────────────────────────────
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu

# ── Vector Store ──────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...              # Only for Qdrant Cloud
QDRANT_COLLECTION_NAME=documents

# ── Observability (all optional) ──────────────────────────────────────────────
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGSMITH_API_KEY=...

# ── App ───────────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
ENV=development                 # development | production
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=20
RERANK_TOP_N=5
```

---

## Coding Conventions

- **Typing:** Full type hints on all functions and class attributes.
- **Docstrings:** Google-style docstrings on all public functions.
- **Error handling:** Use custom exception classes in `app/exceptions.py`.
- **Logging:** Use Python `structlog` for structured JSON logs with correlation IDs.
- **Config:** All values from environment — no hardcoded strings in business logic.
- **Async:** Prefer `async def` for all I/O; use `run_in_executor` for CPU-bound work.
- **Tests:** Every new module must have a corresponding unit test file.

---

## CI/CD Pipeline (GitHub Actions)

On every PR:
1. `ruff` lint + `mypy` type check
2. `pytest tests/unit/` — must pass 100%
3. `pytest tests/integration/` — runs against a local Qdrant Docker container
4. `python tests/eval/run_ragas_eval.py` — RAGAS scores must meet thresholds below

### Eval Thresholds (block merge if below)
| Metric             | Minimum Score |
|--------------------|---------------|
| faithfulness       | 0.85          |
| answer_relevancy   | 0.80          |
| context_precision  | 0.75          |
| context_recall     | 0.75          |

---

## Key Reference Links

- LangChain RAG Tutorial: https://docs.langchain.com/oss/python/langchain/rag
- BAAI BGE Models: https://huggingface.co/BAAI
- RAGAS Docs: https://docs.ragas.io/
- Langfuse Tracing Docs: https://langfuse.com/docs
- LangSmith Docs: https://docs.smith.langchain.com/
- Eugene Yan — LLM Patterns: https://eugeneyan.com/writing/llm-patterns/

---

## Development Quickstart

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd production-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy env vars (no API keys needed for embeddings/reranking)
cp .env.example .env
# Fill in ANTHROPIC_API_KEY at minimum

# 3. Start Qdrant locally
docker-compose up -d qdrant

# 4. Ingest sample documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "data/sample_docs/"}'

# 5. Run the API
uvicorn app.main:app --reload --port 8000

# 6. Run tests
pytest tests/unit/ -v

# 7. Run RAGAS eval
python tests/eval/run_ragas_eval.py
```

---

## What "Done" Looks Like

- [x] Ingestion pipeline handles PDF, HTML, and Markdown sources
- [x] Directory ingestion — recursively loads all supported files concurrently
- [x] BGE-large-en-v1.5 embeddings (local, no OpenAI key required)
- [x] Qdrant vector store — upsert and similarity search
- [x] BM25 sparse retrieval with disk persistence
- [x] Hybrid retrieval (BM25 + dense) with RRF fusion
- [x] BGE-reranker-v2-m3 cross-encoder (local, no Cohere key required)
- [x] Full retrieval pipeline: embed → dense+sparse → RRF → rerank
- [x] Citation extraction and validation (`citations.py`)
- [x] Prompt templates enforcing grounded, cited answers
- [x] FastAPI entrypoint with `/health` and `/ingest` routes
- [x] Docker-compose with Qdrant service
- [x] Multi-stage Dockerfile
- [x] GitHub Actions CI pipeline (lint → unit → integration → RAGAS gate)
- [x] 27 unit tests passing
- [ ] Generation pipeline wired end-to-end (chain + citation retry)
- [ ] `POST /query` route fully implemented
- [ ] Langfuse tracing integrated into RAG chain
- [ ] RAGAS eval suite with real Q&A pairs passing all thresholds
- [ ] Integration tests against live Qdrant
- [ ] README with architecture decisions and before/after metrics
