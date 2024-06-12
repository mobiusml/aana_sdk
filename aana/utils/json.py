import json
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel
from sqlalchemy import Engine


def json_serializer_default(obj: object) -> object:
    """Default function for json serializer to handle custom objects.

    If json serializer does not know how to serialize an object, it calls the default function.

    For example, if we see that the object is a pydantic model,
    we call the dict method to get the dictionary representation of the model
    that json serializer can deal with.

    If the object is not supported, we raise a TypeError.

    Args:
        obj (object): The object to serialize.

    Returns:
        object: The serializable object.

    Raises:
        TypeError: If the object is not a pydantic model, Path, or Media object.
    """
    if isinstance(obj, Engine):
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    from aana.core.models.media import Media

    if isinstance(obj, Media):
        return str(obj)
    raise TypeError


def jsonify(data: Any) -> str:
    """Convert data to JSON string.

    Args:
        data (Any): the data

    Returns:
        str: the JSON string
    """
    return json.dumps(data, default=json_serializer_default)


def orjson_serializer(
    content: Any, option: int | None = orjson.OPT_SERIALIZE_NUMPY
) -> bytes:
    """Serialize content using orjson.

    Args:
        content (Any): The content to serialize.
        option (int | None): The option for orjson.dumps.

    Returns:
        bytes: The serialized content.
    """
    return orjson.dumps(content, option=option, default=json_serializer_default)
