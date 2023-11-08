from typing import Any, Optional
from fastapi.responses import JSONResponse
import orjson
from pydantic import BaseModel


def orjson_default(obj: Any) -> Any:
    """
    Default function for orjson.dumps to handle pydantic models.

    If orjson does not know how to serialize an object, it calls the default function.

    If we see that the object is a pydantic model,
    we call the dict method to get the dictionary representation of the model that orjson can serialize.

    If the object is not a pydantic model, we raise a TypeError.

    Args:
        obj (Any): The object to serialize.

    Returns:
        Any: The serialized object.

    Raises:
        TypeError: If the object is not a pydantic model.
    """

    if isinstance(obj, BaseModel):
        return obj.dict()
    raise TypeError


class AanaJSONResponse(JSONResponse):
    """
    A JSON response class that uses orjson to serialize data.
    It has additional support for numpy arrays.
    """

    media_type = "application/json"
    option = None

    def __init__(self, option: Optional[int] = orjson.OPT_SERIALIZE_NUMPY, **kwargs):
        """
        Initialize the response class with the orjson option.
        """
        self.option = option
        super().__init__(**kwargs)

    def render(self, content: Any) -> bytes:
        """
        Override the render method to use orjson.dumps instead of json.dumps.
        """
        return orjson.dumps(content, option=self.option, default=orjson_default)
