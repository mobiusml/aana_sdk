from typing import Any, TypeVar

from pydantic import BaseModel
import requests

from aana.exceptions.general import DownloadException


OptionType = TypeVar("OptionType", bound=BaseModel)


def merged_options(default_options: OptionType, options: OptionType) -> OptionType:
    """
    Merge options into default_options.

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
    assert type(default_options) == type(options)
    default_options_dict = default_options.dict()
    for k, v in options.dict().items():
        if v is not None:
            default_options_dict[k] = v
    return options.__class__.parse_obj(default_options_dict)


def download_file(url: str) -> bytes:
    """
    Download a file from a URL.

    Args:
        url (str): the URL of the file to download

    Returns:
        bytes: the file content

    Raises:

    """
    # TODO: add retries, check status code, etc.
    try:
        response = requests.get(url)
    except Exception as e:
        raise DownloadException(url) from e
    return response.content


def pydantic_to_dict(data: Any) -> Any:
    """
    Convert all Pydantic objects in the structured data.

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
