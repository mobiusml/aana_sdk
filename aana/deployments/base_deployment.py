import inspect
import pickle
from functools import wraps
from importlib import resources
from pathlib import Path
from typing import Any

import rapidfuzz

from aana.configs.settings import settings
from aana.utils.core import get_object_hash
from aana.utils.json import jsonify


def test_cache(func):  # noqa: C901
    """Decorator for caching and loading the results of a deployment function in testing mode.

    Keep in mind that this decorator only works for async functions and async generator functions.

    Use this decorator to annotate deployment functions that you want to cache in testing mode.

    There are 3 environment variables that control the behavior of the decorator:
    - TEST_MODE: set to "true" to enable testing mode
                (default is "false", should only be set to "true" if you are running tests)
    - USE_DEPLOYMENT_CACHE: set to "true" to enable cache usage
    - SAVE_DEPLOYMENT_CACHE: set to "true" to enable cache saving

    The decorator behaves differently in testing and production modes.

    In production mode, the decorator is a no-op.
    In testing mode, the behavior of the decorator is controlled by the environment variables USE_DEPLOYMENT_CACHE and SAVE_DEPLOYMENT_CACHE.

    If USE_DEPLOYMENT_CACHE is set to "true", the decorator will load the result from the cache if it exists. SAVE_DEPLOYMENT_CACHE is ignored.
    The decorator takes a hash of the deployment configuration and the function arguments and keyword arguments (args and kwargs) to locate the cache file.
    If the cache file exists, the decorator will load the result from the cache and return it.
    If the cache file does not exist, the decorator will try to find the cache file with the closest args and load the result from that cache file
    (function name and deployment configuration should match exactly, fuzzy matching only applies to args and kwargs).

    If USE_DEPLOYMENT_CACHE is set to "false", the decorator will execute the function and save the result to the cache if SAVE_DEPLOYMENT_CACHE is set to "true".
    """
    if not settings.test.test_mode:
        # If we are in production, the decorator is a no-op
        return func

    def get_cache_path(args, kwargs):
        """Get the path to the cache file."""
        self = args[0]

        func_name = func.__name__
        deployment_name = self.__class__.__name__

        config = args[0].config
        config_hash = get_object_hash(config)

        args_hash = get_object_hash({"args": args[1:], "kwargs": kwargs})

        return (
            resources.path("aana.tests.files.cache", "")
            / Path(deployment_name)
            / Path(f"{func_name}_{config_hash}_{args_hash}.pkl")
        )

    def save_cache(cache_path, cache, args, kwargs):
        """Save the cache to a file."""
        cache_obj = {
            "args": jsonify({"args": args[1:], "kwargs": kwargs}),
        }
        if "exception" in cache:
            cache_obj["exception"] = cache[
                "exception"
            ]  # if the cache contains an exception, save it
        else:
            cache_obj["cache"] = cache  # otherwise, cache the result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.open("wb").write(pickle.dumps(cache_obj))

    def find_matching_cache(cache_path, args, kwargs):
        """Find the cache file with the closest args."""

        def get_args(path):
            cache = pickle.loads(path.open("rb").read())  # noqa: S301
            print(cache)
            return cache["args"]

        args_str = jsonify({"args": args[1:], "kwargs": kwargs})
        pattern = cache_path.name.replace(cache_path.name.split("_")[-1], "*")
        candidate_cache_files = list(cache_path.parent.glob(pattern))

        if len(candidate_cache_files) == 0:
            raise FileNotFoundError(f"{cache_path.parent}/{pattern}")

        # find the cache with the closest args
        path = min(
            candidate_cache_files,
            key=lambda path: rapidfuzz.distance.Levenshtein.distance(
                args_str, get_args(path)
            ),
        )
        return Path(path)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        """Wrapper for the deployment function."""
        cache_path = get_cache_path(args, kwargs)

        if settings.test.use_deployment_cache:
            # load from cache
            if not cache_path.exists():
                print(args, kwargs)
                # raise FileNotFoundError(cache_path)
                cache_path = find_matching_cache(cache_path, args, kwargs)
            cache = pickle.loads(cache_path.open("rb").read())  # noqa: S301
            # raise exception if the cache contains an exception
            if "exception" in cache:
                raise cache["exception"]
            return cache["cache"]
        else:
            # execute the function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                result = {"exception": e}
                raise
            finally:
                if settings.test.save_deployment_cache and not cache_path.exists():
                    # save to cache
                    save_cache(cache_path, result, args, kwargs)
            return result

    @wraps(func)
    async def wrapper_generator(*args, **kwargs):
        """Wrapper for the deployment generator function."""
        cache_path = get_cache_path(args, kwargs)

        if settings.test.use_deployment_cache:
            # load from cache
            if not cache_path.exists():
                # raise FileNotFoundError(cache_path)
                cache_path = find_matching_cache(cache_path, args, kwargs)

            cache = pickle.loads(cache_path.open("rb").read())  # noqa: S301
            # raise exception if the cache contains an exception
            if "exception" in cache:
                raise cache["exception"]
            for item in cache["cache"]:
                yield item
        else:
            cache = []
            try:
                # execute the function
                async for item in func(*args, **kwargs):
                    yield item
                    if settings.test.save_deployment_cache:
                        cache.append(item)
            except Exception as e:
                cache = {"exception": e}
                raise
            finally:
                if settings.test.save_deployment_cache and not cache_path.exists():
                    # save to cache
                    save_cache(cache_path, cache, args, kwargs)

    wrapper_generator.test_cache_enabled = True
    wrapper.test_cache_enabled = True

    if inspect.isasyncgenfunction(func):
        return wrapper_generator
    else:
        return wrapper


class BaseDeployment:
    """Base class for all deployments.

    We can use this class to define common methods for all deployments.
    For example, we can connect to the database here or download artifacts.
    """

    def __init__(self):
        """Inits to unconfigured state."""
        self.config = None
        self.configured = False

    async def reconfigure(self, config: dict[str, Any]):
        """Reconfigure the deployment.

        The method is called when the deployment is updated.
        """
        self.config = config
        if (
            settings.test.test_mode
            and settings.test.use_deployment_cache
            and self.check_test_cache_enabled()
        ):
            # If we are in testing mode and we want to use the cache,
            # we don't need to load the model
            self.configured = True
            return
        else:
            await self.apply_config(config)
            self.configured = True

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        Args:
            config (dict): the configuration
        """
        raise NotImplementedError

    async def get_methods(self) -> dict:
        """Returns the methods of the deployment.

        Returns:
            dict: the methods of the deployment with annotations and docstrings
        """
        cls = self.__class__
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        methods_info = {}
        for name, method in methods:
            # Skip private methods
            if name.startswith("_"):
                continue
            # Skip non-asynchronous methods
            if not (
                inspect.iscoroutinefunction(method)
                or inspect.isasyncgenfunction(method)
            ):
                continue

            methods_info[name] = {}
            if method.__annotations__:
                methods_info[name]["annotations"] = method.__annotations__
            if method.__doc__:
                methods_info[name]["doc"] = method.__doc__
        return methods_info

    def check_test_cache_enabled(self):
        """Check if the deployment has any methods decorated with test_cache."""
        for method in self.__class__.__dict__.values():
            if callable(method) and getattr(method, "test_cache_enabled", False):
                return True
        return False
