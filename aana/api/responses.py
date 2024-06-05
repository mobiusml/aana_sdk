from typing import Any

import orjson
from fastapi.responses import JSONResponse

from aana.utils.json import orjson_serializer


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
        return orjson_serializer(content, option=self.option)
