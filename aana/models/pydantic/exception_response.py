from typing import Dict, Optional
from pydantic import BaseModel, Extra


class ExceptionResponseModel(BaseModel):
    """
    This class is used to represent an exception response for 400 errors.

    Attributes:
        error (str): The error that occurred.
        message (str): The message of the error.
        data (Optional[Dict]): The extra data that helps to debug the error.
        stacktrace (Optional[str]): The stacktrace of the error.
    """

    error: str
    message: str
    data: Optional[Dict] = None
    stacktrace: Optional[str] = None

    class Config:
        extra = Extra.forbid
