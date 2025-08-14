import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from functools import wraps
from typing import Any

from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from aana.utils.json import jsonify

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

Yieldable = ServerSentEvent | dict | str | bytes | Any


def _to_sse(item: Yieldable, default_event: str | None) -> ServerSentEvent:
    if isinstance(item, ServerSentEvent):
        return item

    # If it's a preformatted dict with SSE keys, respect them
    if isinstance(item, dict) and any(
        k in item for k in ("event", "data", "id", "retry")
    ):
        data = item.get("data", "")
        if not isinstance(data, str | bytes):
            data = jsonify(data)
        elif isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return ServerSentEvent(
            data=data,
            event=item.get("event") or default_event,
            id=item.get("id"),
            retry=item.get("retry"),
        )

    # Otherwise treat it as a payload for the default event
    if isinstance(item, bytes):
        data = item.decode("utf-8", errors="replace")
    elif isinstance(item, str):
        data = item
    else:
        data = jsonify(item)
    return ServerSentEvent(data=data, event=default_event)


def sse(
    *,
    headers: dict[str, str] | None = None,
    heartbeat: float = 15.0,
    default_event: str | None = "message",
    done_event: Yieldable | None = None,
) -> Callable[
    [Callable[..., AsyncIterable[Yieldable]]], Callable[..., EventSourceResponse]
]:
    """Decorate an async generator to stream SSE using sse-starlette.

    Example usage:
    ```
    @app.get("/tokens")
    @sse(done_event={"event": "done", "data": {}})
    async def tokens(limit: int = 5):
        for i in range(limit):
            await asyncio.sleep(0.2)
            yield {"token": f"token-{i}"}  # auto-JSON + event: "message"
        yield ServerSentEvent(data="1.0", event="progress", id="final")  # explicit SSE
    ```

    Args:
        headers (dict[str, str] | None): Additional headers to include in the response.
        heartbeat (float): Interval in seconds to send a ping to keep the connection alive.
        default_event (str | None): Default event type for the streamed data.
        done_event (Yieldable | None): Optional event to send when the stream is done. E.g., {"event": "done", "data": {}}.
        logger (logging.Logger | None): Optional logger to log cancellation events.

    Returns:
        Callable: Decorated function that returns an EventSourceResponse.
    """

    def deco(fn: Callable[..., AsyncIterable[Yieldable]]):
        @wraps(fn)
        async def wrapper(*args, **kwargs) -> EventSourceResponse:
            async def gen() -> AsyncIterator[ServerSentEvent]:
                try:
                    async for item in fn(*args, **kwargs):
                        yield _to_sse(item, default_event)
                    if done_event is not None:
                        yield _to_sse(done_event, default_event)
                except asyncio.CancelledError:
                    logger.info("SSE stream cancelled by client")
                    return

            return EventSourceResponse(
                gen(),
                ping=heartbeat,
                headers={**_DEFAULT_HEADERS, **(headers or {})},
            )

        return wrapper

    return deco
