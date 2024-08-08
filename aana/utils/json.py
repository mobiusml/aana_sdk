from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel
from sqlalchemy import Engine

__all__ = ["jsonify", "json_serializer_default"]


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
    if isinstance(obj, type):
        return str(type)

    from aana.core.models.media import Media
    if isinstance(obj, Media):
        return str(obj)

    raise TypeError(type(obj))


def jsonify(data: Any, option: int | None = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS, as_bytes: bool = False) -> str | bytes:
    """Serialize content using orjson.

    Args:
        data (Any): The content to serialize.
        option (int | None): The option for orjson.dumps.
        as_bytes (bool): Return output as bytes instead of string

    Returns:
        bytes | str: The serialized data as desired format.
    """
    output = orjson.dumps(data, option=option, default=json_serializer_default)
    return output if as_bytes else output.decode()
