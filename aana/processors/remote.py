import inspect
from collections.abc import Callable

import ray


def run_remote(func: Callable) -> Callable:
    """Wrap a function to run it remotely on Ray.

    Args:
        func (Callable): the function to wrap

    Returns:
        Callable: the wrapped function
    """

    async def generator_wrapper(*args, **kwargs):
        async for item in ray.remote(func).remote(*args, **kwargs):
            yield await item

    if inspect.isgeneratorfunction(func):
        return generator_wrapper
    else:
        return ray.remote(func).remote
