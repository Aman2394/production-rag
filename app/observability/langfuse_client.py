"""Langfuse tracing setup."""

from app.config import get_settings

_settings = get_settings()


def get_langfuse_handler() -> object | None:
    """Return a configured Langfuse callback handler, or None if not configured.

    Returns:
        A LangfuseCallbackHandler instance if credentials are set, else None.
    """
    if not _settings.langfuse_secret_key or not _settings.langfuse_public_key:
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import]

        return CallbackHandler(
            secret_key=_settings.langfuse_secret_key.get_secret_value(),
            public_key=_settings.langfuse_public_key,
            host=_settings.langfuse_host,
        )
    except ImportError:
        return None
