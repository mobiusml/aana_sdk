from typing import Any, Optional
from fastapi.responses import JSONResponse
import orjson


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
        return orjson.dumps(content, option=self.option)
