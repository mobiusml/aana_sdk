from typing import Any, Optional
from pydantic import BaseModel, Extra


class ExceptionResponseModel(BaseModel):
    """
    This class is used to represent an exception response for 400 errors.

    Attributes:
        error (str): The error that occurred.
        message (str): The message of the error.
        data (Optional[Any]): The extra data that helps to debug the error.
        stacktrace (Optional[str]): The stacktrace of the error.
    """

    error: str
    message: str
    data: Optional[Any] = None
    stacktrace: Optional[str] = None

    class Config:
        extra = Extra.forbid
