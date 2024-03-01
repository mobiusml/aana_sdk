import hashlib
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

from aana.api.api_generation import Endpoint
from aana.configs.endpoints import endpoints as all_endpoints
from aana.exceptions.general import DownloadException, EndpointNotFoundException
from aana.utils.json import jsonify

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
    default_options_dict = default_options.model_dump()
    for k, v in options.model_dump().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__.model_validate(default_options_dict)


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
        return data.model_dump()
    elif isinstance(data, list):
        return [pydantic_to_dict(item) for item in data]
    elif isinstance(data, dict):
        return {key: pydantic_to_dict(value) for key, value in data.items()}
    else:
        return data  # return as is for non-Pydantic types


def get_endpoint(target: str, endpoint: str) -> Endpoint:
    """Get endpoint from endpoints config.

    #TODO: make EndpointList a class and make this a method.

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


def get_gpu_memory(gpu: int = 0) -> int:
    """Get the total memory of a GPU in bytes."""
    import torch
    return torch.cuda.get_device_properties(gpu).total_memory