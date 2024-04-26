import asyncio
import threading
from typing import Any


class RunThread(threading.Thread):
    """Run a coroutine in a thread."""

    def __init__(self, coro):
        """Initialize the thread."""
        self.coro = coro
        self.result = None
        super().__init__()

    def run(self):
        """Run the coroutine."""
        self.result = asyncio.run(self.coro)


def run_async(coro: asyncio.coroutine) -> Any:
    """Run an async function in a thread.

    Usefull when you want to run an async function in a non-async context.

    From: https://stackoverflow.com/a/75094151

    Args:
        coro (Coroutine): The coroutine to run.

    Returns:
        Any: The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(coro)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(coro)
