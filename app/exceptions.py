"""Custom exception classes for the production-rag application."""


class RAGException(Exception):
    """Base exception for all application errors."""


class IngestionError(RAGException):
    """Raised when document ingestion fails."""


class RetrievalError(RAGException):
    """Raised when the retrieval pipeline fails."""


class GenerationError(RAGException):
    """Raised when LLM generation fails."""


class CitationError(GenerationError):
    """Raised when a response contains uncited claims."""


class RerankerError(RAGException):
    """Raised when the reranker fails."""


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
