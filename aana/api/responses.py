from typing import Any

import orjson
from fastapi.responses import JSONResponse
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
    raise TypeError


class AanaJSONResponse(JSONResponse):
    """Response class that uses orjson to serialize data.

    It has additional support for numpy arrays.
    """

    media_type = "application/json"
    option = None

    def __init__(self, option: int | None = orjson.OPT_SERIALIZE_NUMPY, **kwargs):
        """Initialize the response class with the orjson option."""
        self.option = option
        super().__init__(**kwargs)

    def render(self, content: Any) -> bytes:
        """Override the render method to use orjson.dumps instead of json.dumps."""
        return orjson.dumps(
            content, option=self.option, default=json_serializer_default
        )
