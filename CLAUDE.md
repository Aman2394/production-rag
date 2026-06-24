# CLAUDE.md — Production RAG Application
# "Ask My Docs" — Domain-Specific Retrieval-Augmented Generation System

## Project Overview

A production-grade, domain-specific RAG system featuring hybrid retrieval (BM25 + dense
vector search), cross-encoder reranking, citation enforcement, streaming responses,
namespace-based multi-tenancy, and a CI-gated RAGAS evaluation pipeline. Supports three
LLM providers — Ollama (local), Anthropic Claude, and Groq (free tier).

---

## Tech Stack

| Layer              | Tool / Library                                                         | Status   |
|--------------------|------------------------------------------------------------------------|----------|
| Language           | Python 3.11+                                                           | ✅ Done  |
| Orchestration      | LangChain (LCEL chains)                                                | ✅ Done  |
| LLM                | Ollama (local) · Anthropic Claude · Groq (free tier) via LangChain    | ✅ Done  |
| Embeddings         | BGE-large-en-v1.5 via sentence-transformers (local)                    | ✅ Done  |
| Vector Store       | Qdrant (local Docker for dev, cloud for prod)                          | ✅ Done  |
| Sparse Retrieval   | BM25 via rank_bm25 (per-namespace indexes)                             | ✅ Done  |
| Reranker           | BGE-reranker-v2-m3 via sentence-transformers (local)                   | ✅ Done  |
| API Layer          | FastAPI + uvicorn (REST + SSE streaming)                               | ✅ Done  |
| Streaming          | Server-Sent Events via `POST /query/stream`                            | ✅ Done  |
| Multi-tenancy      | Namespace isolation — Qdrant filter + per-namespace BM25               | ✅ Done  |
| Conversation Memory| Redis (short-term, 24h TTL) + PostgreSQL (long-term, optional)         | ✅ Done  |
| Evaluation         | RAGAS (faithfulness, answer_relevancy, context_precision/recall)       | ✅ Done  |
| Observability      | Langfuse (tracing, wired into all chains) + LangSmith + structlog      | ✅ Done  |
| Config / Secrets   | pydantic-settings + python-dotenv (.env)                               | ✅ Done  |
| Testing            | pytest + pytest-asyncio (49 unit tests)                                | ✅ Done  |
| CI/CD              | GitHub Actions                                                         | ✅ Done  |
| Containerization   | Docker + docker-compose                                                | ✅ Done  |

> **Note:** OpenAI and Cohere API keys are **not required**. Embeddings and reranking
> run locally using BAAI open-source models. For a fully free production deployment,
> use `LLM_PROVIDER=groq` (14,400 requests/day free at console.groq.com).

---

## Project Structure

```
production-rag/
├── CLAUDE.md                   # ← You are here
├── pyproject.toml              # ✅ All deps declared (incl. langchain-groq)
├── .env.example                # ✅ All env vars documented
├── docker-compose.yml          # ✅ Qdrant + Redis + app services
├── Dockerfile                  # ✅ Multi-stage build
│
├── app/
│   ├── main.py                 # ✅ FastAPI entrypoint + /health
│   ├── config.py               # ✅ pydantic-settings — all env vars incl. Groq
│   ├── exceptions.py           # ✅ Custom exception hierarchy
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py        # ✅ POST /query + POST /query/stream (SSE)
│   │   │   └── ingest.py       # ✅ POST /ingest → run_ingestion(namespace)
│   │   └── schemas.py          # ✅ All models include namespace field
│   │
│   ├── ingestion/
│   │   ├── loader.py           # ✅ PDF, HTML, Markdown — file, directory, URL
│   │   ├── chunker.py          # ✅ RecursiveCharacterTextSplitter
│   │   ├── embedder.py         # ✅ BGE-large-en-v1.5 (local, no API key)
│   │   └── pipeline.py         # ✅ load → chunk(+namespace) → embed → Qdrant + BM25
│   │
│   ├── retrieval/
│   │   ├── vector_store.py     # ✅ AsyncQdrantClient — namespace-filtered search
│   │   ├── bm25_store.py       # ✅ Per-namespace BM25 — lazy load, per-namespace pickle
│   │   ├── hybrid.py           # ✅ Reciprocal Rank Fusion (RRF, k=60)
│   │   ├── reranker.py         # ✅ BGE-reranker-v2-m3 (local, no API key)
│   │   └── pipeline.py         # ✅ embed → dense+sparse(namespace) → RRF → rerank
│   │
│   ├── generation/
│   │   ├── chain.py            # ✅ Ollama/Anthropic/Groq · streaming · Langfuse · retry
│   │   ├── prompts.py          # ✅ System + user prompt templates
│   │   └── citations.py        # ✅ Citation extraction & validation
│   │
│   ├── memory/
│   │   ├── manager.py          # ✅ Coordinates Redis + PostgreSQL layers
│   │   ├── redis_history.py    # ✅ Short-term history (24h TTL, LPUSH/LTRIM)
│   │   └── postgres_store.py   # ✅ Long-term store (optional — no-op if unconfigured)
│   │
│   └── observability/
│       ├── langfuse_client.py  # ✅ CallbackHandler — wired into all LCEL chains
│       └── metrics.py          # ✅ track_request() latency context manager
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py           # ✅ 3 tests
│   │   ├── test_citations.py         # ✅ 6 tests
│   │   ├── test_generation_chain.py  # ✅ 11 tests (incl. Groq + streaming)
│   │   ├── test_hybrid_retrieval.py  # ✅ 3 tests
│   │   ├── test_ingestion_pipeline.py# ✅ 5 tests
│   │   ├── test_loader.py            # ✅ 5 tests
│   │   ├── test_memory.py            # ✅ 7 tests
│   │   └── test_reranker.py          # ✅ 5 tests
│   ├── integration/
│   │   └── test_rag_pipeline.py      # ⏳ Placeholder — needs real Qdrant-backed tests
│   └── eval/
│       ├── eval_dataset.json         # ✅ 15 real Q&A pairs (RAG/ML/vector-search domain)
│       └── run_ragas_eval.py         # ✅ Full RAGAS implementation (static + pipeline modes)
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
  assigns `chunk_id` (UUID), `source`, `page`, `ingested_at`, `namespace` per chunk;
  upserts to Qdrant and BM25 concurrently via `asyncio.gather`

#### Retrieval Pipeline (`app/retrieval/`)
- **`vector_store.py`** — `AsyncQdrantClient`; `ensure_collection` creates
  collection on first use (1024-dim, cosine); `upsert_chunks` stores namespace in
  payload; `similarity_search(query_vector, namespace)` filters by namespace via
  Qdrant `Filter(must=[FieldCondition(key="namespace", ...)])`
- **`bm25_store.py`** — per-namespace `BM25Okapi` instances stored in
  `_indexes: dict[str, {corpus, bm25}]`; lazy on-demand load from
  `data/bm25_indexes/<namespace>.pkl`; `asyncio.Lock` for safe concurrent writes;
  `delete_namespace()` clears in-memory state and pickle file
- **`hybrid.py`** — Reciprocal Rank Fusion (RRF, k=60) merges dense and sparse
  ranked lists; deduplicates by `chunk_id`
- **`reranker.py`** — `BAAI/bge-reranker-v2-m3` CrossEncoder; singleton via
  `@lru_cache`; scoring in thread-pool executor; adds `rerank_score` to each chunk
- **`pipeline.py`** — full orchestration: embed query + BM25 search in parallel
  (both scoped to namespace) → RRF fusion → reranker → top-5 chunks

#### Generation Pipeline (`app/generation/`)
- **`chain.py`** — three LLM providers via `_get_llm()`:
  - `ollama` — local Ollama server, no API key
  - `anthropic` — Claude via Anthropic SDK
  - `groq` — Llama 3.3 70B via Groq API (free tier); lazy import
  - `temperature=0` enforced on all providers for citation compliance
  - `_langfuse_config()` — passes Langfuse `CallbackHandler` to every
    `chain.ainvoke()` and `chain.astream()` call when keys are configured
  - `generate()` — citation validation with up to 2 retries; raises `GenerationError`
    after max retries
  - `stream_generate()` — async generator yielding SSE event dicts:
    `{type: token, content}` per token, then `{type: done, answer, citations}`;
    citation validation runs post-stream with graceful degradation (no retry,
    valid IDs only)
  - `_stream_chain()` — extracted async generator for `chain.astream()`; patchable
    in tests
  - `contextualize_question()` — reformulates follow-up questions into standalone
    retrieval queries using conversation history; also traced via Langfuse
- **`prompts.py`** — system + user prompt templates enforcing citation format;
  history-aware variants; CONTEXTUALIZE_PROMPT for multi-turn support
- **`citations.py`** — `extract_cited_ids` + `validate_citations` (UUID pattern matching)

#### API & Config
- **`app/main.py`** — FastAPI app with `/health`, `/ingest`, `/query`, `/query/stream`
- **`app/config.py`** — all settings via `pydantic-settings`; three LLM providers
  (`ollama`, `anthropic`, `groq`); no required API keys for embeddings/reranking
- **`app/api/schemas.py`** — `namespace` field (pattern `^[a-zA-Z0-9_-]{1,64}$`,
  default `"default"`) on `QueryRequest`, `IngestRequest`, `IngestResponse`
- **`app/api/routes/query.py`**:
  - `POST /query` — full RAG pipeline with memory; passes `namespace` to `retrieve()`
  - `POST /query/stream` — SSE streaming; pre-steps synchronous (history + retrieve);
    streams generation tokens; saves turn to memory after last token; headers:
    `Cache-Control: no-cache`, `X-Accel-Buffering: no`

#### Conversation Memory (`app/memory/`)
- **`manager.py`** — coordinates Redis (short-term) and PostgreSQL (long-term)
- **`redis_history.py`** — LPUSH/LTRIM with 24h TTL; max 20 messages; non-fatal errors
- **`postgres_store.py`** — optional long-term store; no-op when `POSTGRES_URL` unset;
  `conversations` table with JSONB citations column

#### Observability (`app/observability/`)
- **`langfuse_client.py`** — `get_langfuse_handler()` returns a configured
  `CallbackHandler` when `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` are set;
  wired into `_invoke_chain()`, `_stream_chain()`, and `contextualize_question()`
- **`metrics.py`** — `track_request(operation)` async context manager logs
  latency with structlog; used at every pipeline stage

#### RAGAS Evaluation (`tests/eval/`)
- **`eval_dataset.json`** — 15 Q&A samples covering RRF, BM25, reranking, HNSW,
  hybrid search, context precision/recall, faithfulness, embeddings, chunking,
  structured logging; each sample has `question`, `ground_truth`, `reference_contexts`
- **`run_ragas_eval.py`** — full RAGAS 0.2+ implementation:
  - `static` mode (default) — uses `reference_contexts` from JSON; no Qdrant needed;
    evaluates `faithfulness` + `answer_relevancy`
  - `pipeline` mode — runs live retrieval per question; evaluates all 4 metrics
  - RAGAS judge LLM: Groq first (free), Anthropic fallback
  - RAGAS embeddings: `BAAI/bge-small-en-v1.5` (65 MB, configurable via
    `RAGAS_EMBEDDING_MODEL`)
  - `--samples N` flag for fast iteration; CI uses `--mode static`
  - Exits non-zero if any active metric is below threshold

#### Infrastructure
- **`docker-compose.yml`** — Qdrant (6333/6334), Redis (6379), app with health checks
- **`Dockerfile`** — multi-stage build, non-root user
- **`pyproject.toml`** — all deps declared; includes `langchain-groq`
- **`.github/workflows/ci.yml`** — 4-stage CI pipeline

#### Tests (49 passing)
| File | Tests | Coverage |
|------|-------|----------|
| `test_chunker.py` | 3 | Text splitting, chunk size |
| `test_citations.py` | 6 | UUID extraction, validation, dedup |
| `test_generation_chain.py` | 11 | Context format, citations, retries, Groq provider, streaming |
| `test_hybrid_retrieval.py` | 3 | RRF fusion, deduplication |
| `test_ingestion_pipeline.py` | 5 | Chunk metadata (incl. namespace), empty handling |
| `test_loader.py` | 5 | File/directory/URL loading |
| `test_memory.py` | 7 | Redis history, PostgreSQL no-op, manager |
| `test_reranker.py` | 5 | Ranking, scoring, error handling |

---

### ⏳ Remaining Work

1. **Integration tests** — replace placeholder in `test_rag_pipeline.py` with real
   end-to-end tests against live Qdrant + populated namespace
2. **README** — architecture diagram, before/after RAGAS metric numbers, deployment
   guide, API reference
3. **Rate limiting** — `slowapi` with per-namespace or per-IP limiter
4. **API key auth** — Bearer token middleware mapping keys → allowed namespaces
5. **Document management routes** — `DELETE /namespaces/{id}`, `GET /namespaces/{id}/docs`
   (`delete_namespace()` is already implemented in `bm25_store.py`; needs Qdrant + route)
6. **Prometheus `/metrics` endpoint** — for Grafana dashboards in production

---

## Core Architectural Decisions

### 1. Hybrid Retrieval (BM25 + Dense Vector)
- Dense retrieval handles semantic similarity; BM25 handles exact keyword matching.
- Results are fused using **Reciprocal Rank Fusion (RRF, k=60)** before reranking.
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
- The system prompt enforces `[chunk_id]` format using the actual UUID from context.
- Post-generation, `citations.py` validates all cited IDs exist in the retrieved set.
- Non-streaming: retries up to 2 times with escalating citation reminder in the prompt.
- Streaming: validates post-stream; degrades gracefully (filters to valid IDs) rather
  than retrying, since tokens are already sent.

### 5. Async Throughout
- All I/O-bound operations (vector DB, Redis, LLM calls) are `async def`.
- CPU-bound operations (embedding, reranking) run in thread-pool executors.
- `asyncio.gather()` used for parallel operations (embed + BM25 search, Qdrant + BM25 upsert).

### 6. Namespace Multi-Tenancy
- Every document is tagged with a `namespace` at ingest time (stored in Qdrant payload
  and in the BM25 corpus entry).
- **Dense retrieval:** Qdrant `Filter(must=[FieldCondition(key="namespace", ...)])` scopes
  every `query_points` call — no cross-namespace leakage is possible at the DB layer.
- **Sparse retrieval:** each namespace has its own `BM25Okapi` instance in a dict keyed
  by namespace; persisted to `data/bm25_indexes/<namespace>.pkl`; loaded lazily on first
  access.
- `namespace` defaults to `"default"` — existing clients that omit it continue to work.
- Pattern validation `^[a-zA-Z0-9_-]{1,64}$` prevents path traversal and keeps pickle
  filenames filesystem-safe.
- Analogous to a **NotebookLM notebook**: each namespace is an isolated knowledge base.

### 7. Streaming (Server-Sent Events)
- `POST /query/stream` returns `text/event-stream` via FastAPI `StreamingResponse`.
- History load + question contextualisation + retrieval run synchronously first (fast).
- Generation streams token-by-token via LangChain `chain.astream()`.
- Three SSE event types: `token` (per LLM token), `done` (full answer + citations +
  session_id), `error` (on failure).
- Memory save (`save_turn`) runs after the last token, before the generator closes.
- `X-Accel-Buffering: no` header disables nginx buffering for real-time delivery.

### 8. Multi-Provider LLM Support
- Provider selected via `LLM_PROVIDER` env var; `temperature=0` enforced on all.
- `ollama` — local, free, no API key; good for dev and air-gapped deployments.
- `anthropic` — Claude; best quality; requires `ANTHROPIC_API_KEY`.
- `groq` — Llama 3.3 70B via Groq LPU; free tier (14,400 req/day); best for
  production without spend; requires `GROQ_API_KEY`; lazy import.

### 9. Evaluation-First Development
- Every retrieval change must show metric delta in RAGAS before merging.
- Tracked metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- Eval runs are gated in CI — PRs that fall below thresholds are blocked.

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
- `namespace` (used for multi-tenant isolation in Qdrant and BM25)

---

## Streaming API

### `POST /query/stream`

Request body: same as `POST /query` (includes `namespace`, `session_id`, `top_k`).

Response: `text/event-stream` — one JSON object per `data:` line.

```
data: {"type": "token",  "content": "Reciprocal Rank Fusion "}
data: {"type": "token",  "content": "combines results from "}
...
data: {"type": "done",   "session_id": "abc-123", "answer": "<full answer>", "citations": [...]}
data: {"type": "error",  "detail": "<message>"}   ← only on failure
```

```bash
# curl example
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RRF?", "namespace": "my-notebook"}'
```

---

## Prompt Templates

### System Prompt (enforces grounding + citation)
```
You are a precise, grounded Q&A assistant. Answer ONLY using the provided context.
Each context chunk starts with its ID in square brackets, e.g. [550e8400-...].
For every claim you make, you MUST cite the exact chunk ID from the context.
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
# ── LLM provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER=ollama              # ollama | anthropic | groq

# Ollama (local, no API key)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Anthropic (optional)
# ANTHROPIC_API_KEY=sk-ant-...
# ANTHROPIC_MODEL=claude-sonnet-4-6

# Groq — free tier, 14 400 req/day (recommended for production without spend)
# Get key at: https://console.groq.com — no credit card required
# GROQ_API_KEY=gsk_...
# GROQ_MODEL=llama-3.3-70b-versatile

# ── Embeddings (local — no API key required) ──────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu              # cpu | cuda | mps

# ── Reranker (local — no API key required) ────────────────────────────────────
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu

# ── Vector Store ──────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                   # Required for Qdrant Cloud; leave blank for local
QDRANT_COLLECTION_NAME=documents

# ── Observability (all optional) ──────────────────────────────────────────────
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com

LANGSMITH_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=production-rag

# ── Memory — short-term (Redis) ───────────────────────────────────────────────
REDIS_URL=redis://localhost:6379
REDIS_HISTORY_TTL=86400           # seconds — 24h session expiry
REDIS_MAX_HISTORY=20              # max messages stored per session

# ── Memory — long-term (PostgreSQL, optional) ─────────────────────────────────
# POSTGRES_URL=postgresql://user:password@localhost:5432/ragdb

# ── App ───────────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
ENV=development                   # development | production

# ── Retrieval Tuning ──────────────────────────────────────────────────────────
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=20                # Candidates passed to reranker
RERANK_TOP_N=5                    # Final docs returned to LLM
```

---

## Coding Conventions

- **Typing:** Full type hints on all functions and class attributes.
- **Docstrings:** Google-style docstrings on all public functions.
- **Error handling:** Use custom exception classes in `app/exceptions.py`.
- **Logging:** Use Python `structlog` for structured JSON logs.
- **Config:** All values from environment — no hardcoded strings in business logic.
- **Async:** Prefer `async def` for all I/O; use `run_in_executor` for CPU-bound work.
- **Tests:** Every new module must have a corresponding unit test file.
- **temperature=0** on all LLM providers — required for deterministic citation compliance.

---

## CI/CD Pipeline (GitHub Actions)

On every PR:
1. `ruff` lint + `mypy` type check
2. `pytest tests/unit/` — must pass 100% (49 tests)
3. `pytest tests/integration/` — runs against a local Qdrant Docker container
4. `python tests/eval/run_ragas_eval.py --mode static` — RAGAS scores must meet thresholds

### Eval Thresholds (block merge if below)
| Metric             | Minimum | Mode evaluated in CI |
|--------------------|---------|----------------------|
| faithfulness       | 0.85    | static + pipeline    |
| answer_relevancy   | 0.80    | static + pipeline    |
| context_precision  | 0.75    | pipeline only        |
| context_recall     | 0.75    | pipeline only        |

---

## Development Quickstart

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd production-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy env vars
cp .env.example .env
# Minimum required: set LLM_PROVIDER and its key (Groq recommended — free)

# 3. Start services
docker-compose up -d qdrant redis

# 4. Run the API
uvicorn app.main:app --reload --port 8000

# 5. Ingest documents into a namespace
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "data/sample_docs/", "namespace": "my-notebook"}'

# 6. Query (standard)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RRF?", "namespace": "my-notebook"}'

# 7. Query (streaming)
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RRF?", "namespace": "my-notebook"}'

# 8. Run unit tests
pytest tests/unit/ -v

# 9. Run RAGAS eval (quick — 5 samples, no Qdrant needed)
GROQ_API_KEY=gsk_... python tests/eval/run_ragas_eval.py --samples 5

# 10. Run RAGAS eval (full static mode)
python tests/eval/run_ragas_eval.py --mode static

# 11. Run RAGAS eval (pipeline mode — requires ingested Qdrant data)
python tests/eval/run_ragas_eval.py --mode pipeline
```

---

## What "Done" Looks Like

- [x] Ingestion pipeline handles PDF, HTML, and Markdown sources
- [x] Directory ingestion — recursively loads all supported files concurrently
- [x] BGE-large-en-v1.5 embeddings (local, no OpenAI key required)
- [x] Qdrant vector store — upsert and namespace-filtered similarity search
- [x] BM25 sparse retrieval — per-namespace indexes with disk persistence
- [x] Hybrid retrieval (BM25 + dense) with RRF fusion
- [x] BGE-reranker-v2-m3 cross-encoder (local, no Cohere key required)
- [x] Full retrieval pipeline: embed → dense+sparse(namespace) → RRF → rerank
- [x] Namespace multi-tenancy — complete isolation per namespace in Qdrant + BM25
- [x] Citation extraction and validation (`citations.py`)
- [x] Prompt templates enforcing grounded, cited answers
- [x] `POST /query` — fully implemented with memory, contextualisation, citations
- [x] `POST /query/stream` — SSE streaming with token-by-token delivery
- [x] Multi-turn conversation memory (Redis short-term + PostgreSQL long-term optional)
- [x] Three LLM providers: Ollama (local) · Anthropic Claude · Groq (free tier)
- [x] Langfuse tracing wired into all LCEL chain invocations
- [x] RAGAS eval suite — 15 Q&A samples, static + pipeline modes, CI-gated thresholds
- [x] FastAPI entrypoint with `/health`, `/ingest`, `/query`, `/query/stream`
- [x] Docker-compose with Qdrant + Redis services
- [x] Multi-stage Dockerfile
- [x] GitHub Actions CI pipeline (lint → unit → integration → RAGAS gate)
- [x] 49 unit tests passing
- [ ] Integration tests against live Qdrant (placeholder only)
- [ ] README with architecture diagram and before/after RAGAS metrics
- [ ] Rate limiting + API key authentication
- [ ] Document management routes (DELETE /namespaces/{id}, GET /namespaces/{id}/docs)
- [ ] Prometheus /metrics endpoint
