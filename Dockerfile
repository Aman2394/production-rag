# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some packages (e.g. lxml, unstructured)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install project + runtime deps into a prefix we can copy later
RUN pip install --upgrade pip && \
    pip install --prefix=/install -e ".[dev]" --no-deps || true && \
    pip install --prefix=/install . 2>/dev/null || \
    pip install --prefix=/install \
        "langchain>=0.3.0" \
        "langchain-community>=0.3.0" \
        "langchain-openai>=0.2.0" \
        "langchain-anthropic>=0.3.0" \
        "anthropic>=0.40.0" \
        "openai>=1.50.0" \
        "qdrant-client>=1.12.0" \
        "rank-bm25>=0.2.2" \
        "cohere>=5.0.0" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.32.0" \
        "pydantic>=2.9.0" \
        "pydantic-settings>=2.6.0" \
        "python-dotenv>=1.0.0" \
        "langfuse>=2.50.0" \
        "langsmith>=0.2.0" \
        "structlog>=24.4.0" \
        "pypdf>=5.0.0" \
        "beautifulsoup4>=4.12.0" \
        "lxml>=5.3.0" \
        "httpx>=0.28.0" \
        "aiofiles>=24.1.0"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
