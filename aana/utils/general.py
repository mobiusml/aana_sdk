import hashlib
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

from aana.api.api_generation import Endpoint
from aana.configs.endpoints.endpoints import endpoints as all_endpoints
from aana.exceptions.general import DownloadException, EndpointNotFoundException
from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
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


def get_attribute(obj: Any, attribute: str) -> Any:
    """Get an attribute of an object.

    Args:
        obj (Any): the object
        attribute (str): the attribute name

    Returns:
        Any: the attribute value
    """
    return getattr(obj, attribute)


def format_prompt_as_chat_dialog(
    system_prompt: str, prompt_template: str, text: str
) -> ChatDialog:
    """Format a prompt template with text.

    Args:
        system_prompt (str): the system prompt
        prompt_template (str): the prompt template
        text (str): the text to insert into the template

    Returns:
        ChatDialog: the formatted prompt as a ChatDialog with two ChatMessage objects:
            one that contains the system prompt
            one that contains the formatted prompt
    """
    system_prompt_message = ChatMessage(
        content=system_prompt,
        role="system",
    )
    formatted_prompt_message = ChatMessage(
        content=prompt_template.format(text=text),
        role="user",
    )
    dialog = ChatDialog(
        messages=[
            system_prompt_message,
            formatted_prompt_message,
        ]
    )
    print(dialog)
    return dialog

    # """Format a prompt template with text.

    # Args:
    #     prompt_template (str): the prompt template
    #     text (str): the text to insert into the template

    # Returns:
    #     str: the formatted prompt
    # """
    # print(prompt_template)
    # print(text)
    # print(prompt_template.format(text=text))
    # return prompt_template.format(text=text)
