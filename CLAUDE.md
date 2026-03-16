# CLAUDE.md — Production RAG Application
# "Ask My Docs" — Domain-Specific Retrieval-Augmented Generation System

## Project Overview

A production-grade, domain-specific RAG system featuring hybrid retrieval (BM25 + dense
vector search), cross-encoder reranking, citation enforcement, and a CI-gated evaluation
pipeline. This is the most common pattern in enterprise AI and the gold standard for
portfolio-ready RAG work.

---

## Tech Stack

| Layer              | Tool / Library                                      |
|--------------------|-----------------------------------------------------|
| Language           | Python 3.11+                                        |
| Orchestration      | LangChain (LCEL chains)                             |
| LLM                | Claude (claude-sonnet-4-6) via Anthropic SDK        |
| Embeddings         | text-embedding-3-small (OpenAI) or BGE-large-en     |
| Vector Store       | Qdrant (local Docker for dev, cloud for prod)       |
| Sparse Retrieval   | BM25 via rank_bm25                                  |
| Reranker           | Cohere Rerank API or BGE-reranker-v2 (cross-encoder)|
| API Layer          | FastAPI + uvicorn                                   |
| Evaluation         | RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) |
| Observability      | Langfuse (tracing) + LangSmith (prompt versioning)  |
| Config / Secrets   | pydantic-settings + python-dotenv (.env)            |
| Testing            | pytest + pytest-asyncio                             |
| CI/CD              | GitHub Actions                                      |
| Containerization   | Docker + docker-compose                             |

---

## Project Structure

```
production-rag/
├── CLAUDE.md                   # ← You are here
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile
│
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # pydantic-settings config
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py        # POST /query endpoint
│   │   │   └── ingest.py       # POST /ingest endpoint
│   │   └── schemas.py          # Request / Response Pydantic models
│   │
│   ├── ingestion/
│   │   ├── loader.py           # PDF, HTML, Markdown loaders
│   │   ├── chunker.py          # RecursiveCharacterTextSplitter
│   │   ├── embedder.py         # Embedding model wrapper
│   │   └── pipeline.py         # Async ingestion orchestration
│   │
│   ├── retrieval/
│   │   ├── vector_store.py     # Qdrant client wrapper
│   │   ├── bm25_store.py       # BM25 index (rank_bm25)
│   │   ├── hybrid.py           # RRF fusion of dense + sparse results
│   │   ├── reranker.py         # Cross-encoder reranking (Cohere / BGE)
│   │   └── pipeline.py         # Full retrieval orchestration
│   │
│   ├── generation/
│   │   ├── chain.py            # LangChain LCEL RAG chain
│   │   ├── prompts.py          # System + user prompt templates
│   │   └── citations.py        # Citation extraction & enforcement
│   │
│   └── observability/
│       ├── langfuse_client.py  # Tracing setup
│       └── metrics.py          # Latency, cost-per-request tracking
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_hybrid_retrieval.py
│   │   └── test_citations.py
│   ├── integration/
│   │   └── test_rag_pipeline.py
│   └── eval/
│       ├── eval_dataset.json   # Ground-truth Q&A pairs
│       └── run_ragas_eval.py   # RAGAS evaluation script
│
└── .github/
    └── workflows/
        └── ci.yml              # Lint + test + eval gate on every PR
```

---

## Core Architectural Decisions

### 1. Hybrid Retrieval (BM25 + Dense Vector)
- Dense retrieval handles semantic similarity; BM25 handles exact keyword matching.
- Results are fused using **Reciprocal Rank Fusion (RRF)** before reranking.
- Never use dense-only retrieval in production — keyword queries degrade badly.

### 2. Cross-Encoder Reranking
- After hybrid fusion, pass top-20 candidates through a cross-encoder (Cohere Rerank or
  BGE-reranker-v2) to get the final top-5.
- Reranking is the single highest-leverage improvement in a RAG pipeline.
- Reference: https://docs.cohere.com/docs/rerank-guide

### 3. Citation Enforcement
- Every LLM response **must** include inline citations referencing specific retrieved chunks.
- The system prompt enforces a structured citation format.
- Post-generation, `citations.py` validates that all cited chunk IDs exist in the context.
- Uncited claims should trigger a retry or a structured fallback.

### 4. Async Throughout
- All I/O-bound operations (embedding calls, vector DB queries, LLM calls) must be `async`.
- Use `asyncio.gather()` for parallel embedding batches during ingestion.

### 5. Evaluation-First Development
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

Always store the following metadata per chunk:
- `source` (filename / URL)
- `page` or `section`
- `chunk_id` (UUID)
- `ingested_at` (ISO timestamp)

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
# .env.example
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...           # For embeddings (if using OpenAI)
COHERE_API_KEY=...              # For Cohere Rerank
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...              # For Qdrant Cloud
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGSMITH_API_KEY=...
LOG_LEVEL=INFO
ENV=development                 # development | production
```

---

## Coding Conventions

- **Typing:** Full type hints on all functions and class attributes.
- **Docstrings:** Google-style docstrings on all public functions.
- **Error handling:** Use custom exception classes in `app/exceptions.py`.
- **Logging:** Use Python `structlog` for structured JSON logs with correlation IDs.
- **Config:** All values from environment — no hardcoded strings in business logic.
- **Async:** Prefer `async def` for all functions that touch I/O.
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
- Cohere Rerank Guide: https://docs.cohere.com/docs/rerank-guide
- RAGAS Docs: https://docs.ragas.io/
- Langfuse Tracing Docs: https://langfuse.com/docs
- LangSmith Docs: https://docs.smith.langchain.com/
- Eugene Yan — LLM Patterns: https://eugeneyan.com/writing/llm-patterns/

---

## Development Quickstart

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd production-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy and fill env vars
cp .env.example .env

# 3. Start Qdrant locally
docker-compose up -d qdrant

# 4. Ingest sample documents
python -m app.ingestion.pipeline --source ./data/sample_docs/

# 5. Run the API
uvicorn app.main:app --reload --port 8000

# 6. Run tests
pytest tests/unit/ -v

# 7. Run RAGAS eval
python tests/eval/run_ragas_eval.py
```

---

## What "Done" Looks Like

- [ ] Ingestion pipeline handles PDF, HTML, and Markdown sources
- [ ] Hybrid retrieval (BM25 + dense) with RRF fusion working end-to-end
- [ ] Cross-encoder reranker integrated and benchmarked vs. no-rerank baseline
- [ ] Citations enforced and validated in every LLM response
- [ ] RAGAS eval suite passing all thresholds
- [ ] CI pipeline blocks PRs on eval regression
- [ ] Langfuse tracing showing per-request latency and cost
- [ ] Docker-compose spins up the full stack in one command
- [ ] README documents architecture decisions with before/after metrics