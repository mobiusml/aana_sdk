import asyncio
from queue import Empty


async def async_streamer_adapter(streamer):
    """Adapt the TextIteratorStreamer to an async generator."""
    while True:
        try:
            for item in streamer:
                yield item
            break
        except Empty:
            # wait for the next item
            await asyncio.sleep(0.01)
