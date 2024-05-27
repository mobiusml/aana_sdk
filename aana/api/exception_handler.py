import traceback

from fastapi import Request
from pydantic import ValidationError
from ray.exceptions import RayTaskError

from aana.api.responses import AanaJSONResponse
from aana.core.models.exception import ExceptionResponseModel
from aana.exceptions.core import BaseException


async def validation_exception_handler(request: Request, exc: ValidationError):
    """This handler is used to handle pydantic validation errors.

    Args:
        request (Request): The request object
        exc (ValidationError): The validation error

    Returns:
        JSONResponse: JSON response with the error details
    """
    return AanaJSONResponse(
        status_code=422,
        content=ExceptionResponseModel(
            error="ValidationError",
            message="Validation error",
            data=exc.errors(),
        ).model_dump(),
    )


def custom_exception_handler(request: Request | None, exc_raw: Exception):
    """This handler is used to handle custom exceptions raised in the application.

    BaseException is the base exception for all the exceptions
    from the Aana application.
    Sometimes custom exception are wrapped into RayTaskError so we need to handle that as well.

    Args:
        request (Request): The request object
        exc_raw (Exception): The exception raised

    Returns:
        JSONResponse: JSON response with the error details. The response contains the following fields:
            error: The name of the exception class.
            message: The message of the exception.
            data: The additional data returned by the exception that can be used to identify the error (e.g. image path, url, model name etc.)
            stacktrace: The stacktrace of the exception.
    """
    # a BaseException can be wrapped into a RayTaskError
    if isinstance(exc_raw, RayTaskError):
        # str(e) returns whole stack trace
        # if exception is a RayTaskError
        # let's use it to get the stack trace
        stacktrace = str(exc_raw)
        # get the original exception
        exc = exc_raw.cause
    else:
        # if it is not a RayTaskError
        # then we need to get the stack trace
        stacktrace = traceback.format_exc()
        exc = exc_raw
    # get the data from the exception
    # can be used to return additional info
    # like image path, url, model name etc.
    data = exc.get_data() if isinstance(exc, BaseException) else {}
    # get the name of the class of the exception
    # can be used to identify the type of the error
    error = exc.__class__.__name__
    # get the message of the exception
    message = str(exc)
    status_code = getattr(exc, "http_status_code", 400)
    return AanaJSONResponse(
        status_code=status_code,
        content=ExceptionResponseModel(
            error=error, message=message, data=data, stacktrace=stacktrace
        ).model_dump(),
    )


async def aana_exception_handler(request: Request, exc: Exception):
    """This handler is used to handle exceptions raised by the Aana application.

    Args:
        request (Request): The request object
        exc (Exception): The exception raised

    Returns:
        JSONResponse: JSON response with the error details
    """
    return custom_exception_handler(request, exc)
