from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from langfuse import get_client
    _LANGFUSE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional runtime dependency fallback
    get_client = None  # type: ignore[assignment]
    _LANGFUSE_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)


def get_langfuse_base_url() -> str | None:
    return os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")


def is_langfuse_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and get_langfuse_base_url()
        and get_client is not None
    )


def get_langfuse_import_error() -> Exception | None:
    return _LANGFUSE_IMPORT_ERROR


def get_langfuse() -> Any | None:
    if not is_langfuse_enabled():
        return None
    return get_client()


def log_langfuse_connection_status() -> bool:
    if not is_langfuse_enabled():
        if get_client is None and _LANGFUSE_IMPORT_ERROR is not None:
            logger.warning("Langfuse SDK import failed: %s", _LANGFUSE_IMPORT_ERROR)
        else:
            logger.info("Langfuse tracing disabled: missing configuration or SDK dependency")
        return False

    client = get_langfuse()
    if client is None:
        logger.warning("Langfuse tracing unavailable: client could not be created")
        return False

    try:
        client.auth_check()
    except Exception as exc:
        logger.warning(
            "Langfuse connection check failed for %s: %s",
            get_langfuse_base_url(),
            exc,
        )
        return False

    logger.info("Langfuse connection established for %s", get_langfuse_base_url())
    return True


@contextmanager
def observe(
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    output: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    client = get_langfuse()
    if client is None:
        yield None
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if model is not None:
        kwargs["model"] = model
    if metadata is not None:
        kwargs["metadata"] = metadata

    with client.start_as_current_observation(**kwargs) as observation:
        yield observation


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()
