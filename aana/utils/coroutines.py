from asyncio import coroutine, is_coroutine, run

from typing_extensions import Any


def run_sync(x: coroutine | Any) -> Any:
    """Resolves a function return value if it is a coroutine.

    Arguments:
        x (coroutine | Any): a return value from a function that may be async

    Returns:
        Any: x, if x is not awaitable, otherwise the final result of x
    """
    if is_coroutine(x):
        return run(x)
    return x
