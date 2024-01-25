from pathlib import Path
from typing import Any

from pydantic import BaseModel


def json_serializer_default(obj: Any) -> Any:
    """Default function for json serializer to handle pydantic models.

    If json serializer does not know how to serialize an object, it calls the default function.

    If we see that the object is a pydantic model,
    we call the dict method to get the dictionary representation of the model
    that json serializer can deal with.

    If the object is not a pydantic model, we raise a TypeError.

    Args:
        obj (Any): The object to serialize.

    Returns:
        Any: The serializable object.

    Raises:
        TypeError: If the object is not a pydantic model.
    """
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError
