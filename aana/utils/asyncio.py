import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

__all__ = ["run_async"]

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine in a thread if the current thread is running an event loop.

    Otherwise, run the coroutine in the current asyncio loop.

    Useful when you want to run an async function in a non-async context.

    From: https://stackoverflow.com/a/75094151

    Args:
        coro (Coroutine): The coroutine to run.

    Returns:
        T: The result of the coroutine.
    """

    class RunThread(threading.Thread):
        """Run a coroutine in a thread."""

        def __init__(self, coro: Coroutine[Any, Any, T]):
            """Initialize the thread."""
            self.coro = coro
            self.result: T | None = None
            self.exception: Exception | None = None
            super().__init__()

        def run(self):
            """Run the coroutine."""
            try:
                self.result = asyncio.run(self.coro)
            except Exception as e:
                self.exception = e

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        thread = RunThread(coro)
        thread.start()
        thread.join()
        if thread.exception:
            raise thread.exception
        return thread.result
    else:
        return asyncio.run(coro)
