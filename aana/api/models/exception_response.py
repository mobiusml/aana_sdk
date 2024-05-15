from typing import Any

from pydantic import BaseModel, ConfigDict


class ExceptionResponseModel(BaseModel):
    """This class is used to represent an exception response for 400 errors.

    Attributes:
        error (str): The error that occurred.
        message (str): The message of the error.
        data (Optional[Any]): The extra data that helps to debug the error.
        stacktrace (Optional[str]): The stacktrace of the error.
    """

    error: str
    message: str
    data: Any | None = None
    stacktrace: str | None = None
    model_config = ConfigDict(extra="forbid")
