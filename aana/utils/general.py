import hashlib
import inspect
import json
import os
import pickle
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import rapidfuzz
import requests
from pydantic import BaseModel

from aana.api.api_generation import Endpoint
from aana.configs.endpoints import endpoints as all_endpoints
from aana.configs.settings import settings
from aana.exceptions.general import DownloadException, EndpointNotFoundException
from aana.utils.json import json_serializer_default

OptionType = TypeVar("OptionType", bound=BaseModel)


def merged_options(default_options: OptionType, options: OptionType) -> OptionType:
    """Merge options into default_options.

    Args:
        default_options (OptionType): default options
        options (OptionType): options to be merged into default_options

    Returns:
        OptionType: merged options
    """
    # if options is None, return default_options
    if options is None:
        return default_options.copy()
    # options and default_options have to be of the same type
    if type(default_options) != type(options):
        raise ValueError("Option type mismatch.")  # noqa: TRY003
    default_options_dict = default_options.dict()
    for k, v in options.dict().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__.parse_obj(default_options_dict)


def download_file(url: str) -> bytes:
    """Download a file from a URL.

    Args:
        url (str): the URL of the file to download

    Returns:
        bytes: the file content

    Raises:
        DownloadException: Request does not succeed.
    """
    # TODO: add retries, check status code, etc.
    try:
        response = requests.get(url)  # noqa: S113 TODO
    except Exception as e:
        raise DownloadException(url) from e
    return response.content


def pydantic_to_dict(data: Any) -> Any:
    """Convert all Pydantic objects in the structured data.

    Args:
        data (Any): the structured data

    Returns:
        Any: the same structured data with Pydantic objects converted to dictionaries
    """
    if isinstance(data, BaseModel):
        return data.dict()
    elif isinstance(data, list):
        return [pydantic_to_dict(item) for item in data]
    elif isinstance(data, dict):
        return {key: pydantic_to_dict(value) for key, value in data.items()}
    else:
        return data  # return as is for non-Pydantic types


def get_endpoint(target: str, endpoint: str) -> Endpoint:
    """Get endpoint from endpoints config.

    Args:
        target (str): the name of the target deployment
        endpoint (str): the endpoint path

    Returns:
        Endpoint: the endpoint

    Raises:
        EndpointNotFoundException: If the endpoint is not found
    """
    for e in all_endpoints[target]:
        if e.path == endpoint:
            return e
    raise EndpointNotFoundException(target=target, endpoint=endpoint)


def is_testing() -> bool:
    """Check if the code is running in testing mode.

    Returns:
        bool: True if the code is running in testing mode, False otherwise
    """
    return os.environ.get("TEST_MODE", "false").lower() in ["true", "1"]


def jsonify(data: Any) -> str:
    """Convert data to JSON string.

    Args:
        data (Any): the data

    Returns:
        str: the JSON string
    """
    return json.dumps(data, default=json_serializer_default)


def get_object_hash(obj: Any) -> str:
    """Get the MD5 hash of an object.

    Objects are converted to JSON strings before hashing.

    Args:
        obj (Any): the object to hash

    Returns:
        str: the MD5 hash of the object
    """
    return hashlib.md5(
        jsonify(obj).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


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
    if not is_testing():
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
            return cache["args"]

        args_str = jsonify({"args": args[1:], "kwargs": kwargs})
        pattern = cache_path.parent.glob(
            cache_path.name.replace(cache_path.name.split("_")[-1], "*")
        )

        # find the cache with the closest args
        path = min(
            pattern,
            key=lambda path: rapidfuzz.distance.Levenshtein.distance(
                args_str, get_args(path)
            ),
        )
        return Path(path)

    async def wrapper(*args, **kwargs):
        """Wrapper for the deployment function."""
        cache_path = get_cache_path(args, kwargs)

        if settings.use_deployment_cache:
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
                if settings.save_deployment_cache and not cache_path.exists():
                    # save to cache
                    save_cache(cache_path, result, args, kwargs)
            return result

    async def wrapper_generator(*args, **kwargs):
        """Wrapper for the deployment generator function."""
        cache_path = get_cache_path(args, kwargs)

        if settings.use_deployment_cache:
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
                    if settings.save_deployment_cache:
                        cache.append(item)
            except Exception as e:
                cache = {"exception": e}
                raise
            finally:
                if settings.save_deployment_cache and not cache_path.exists():
                    # save to cache
                    save_cache(cache_path, cache, args, kwargs)

    if inspect.isasyncgenfunction(func):
        return wrapper_generator
    else:
        return wrapper
