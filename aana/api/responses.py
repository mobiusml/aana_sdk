from typing import Any

import orjson
from fastapi.responses import JSONResponse

from aana.utils.json import jsonify


class AanaJSONResponse(JSONResponse):
    """Response class that uses orjson to serialize data.

    It has additional support for numpy arrays.
    """

    media_type = "application/json"
    option = None

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        option: int | None = orjson.OPT_SERIALIZE_NUMPY,
        **kwargs,
    ):
        """Initialize the response class with the orjson option."""
        self.option = option
        super().__init__(content, status_code, **kwargs)

    def render(self, content: Any) -> bytes:
        """Override the render method to use orjson.dumps instead of json.dumps."""
        return jsonify(content, option=self.option, as_bytes=True)


class AanaNDJSONResponse(AanaJSONResponse):
    """Response class that uses orjson to serialize data to newline-delimited JSON (NDJSON)."""

    media_type = "application/x-ndjson"
