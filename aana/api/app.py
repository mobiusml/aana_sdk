import traceback

from fastapi import FastAPI, Request
from mobius_pipeline.exceptions import BaseException
from pydantic import ValidationError
from ray.exceptions import RayTaskError

from aana.api.responses import AanaJSONResponse
from aana.models.pydantic.exception_response import ExceptionResponseModel

app = FastAPI()


@app.exception_handler(ValidationError)
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
        ).dict(),
    )


def custom_exception_handler(
    request: Request | None, exc_raw: BaseException | RayTaskError
):
    """This handler is used to handle custom exceptions raised in the application.

    BaseException is the base exception for all the exceptions
    from the Mobius Pipeline and Aana application.
    Sometimes custom exception are wrapped into RayTaskError so we need to handle that as well.

    Args:
        request (Request): The request object
        exc_raw (Union[BaseException, RayTaskError]): The exception raised

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
        exc: BaseException = exc_raw.cause
        if not isinstance(exc, BaseException):
            raise TypeError(exc)
    else:
        # if it is not a RayTaskError
        # then we need to get the stack trace
        stacktrace = traceback.format_exc()
        exc = exc_raw
    # get the data from the exception
    # can be used to return additional info
    # like image path, url, model name etc.
    data = exc.get_data()
    # get the name of the class of the exception
    # can be used to identify the type of the error
    error = exc.__class__.__name__
    # get the message of the exception
    message = str(exc)
    status_code = exc.http_status_code if hasattr(exc, "http_status_code") else 400
    return AanaJSONResponse(
        status_code=status_code,
        content=ExceptionResponseModel(
            error=error, message=message, data=data, stacktrace=stacktrace
        ).dict(),
    )


@app.exception_handler(BaseException)
async def pipeline_exception_handler(request: Request, exc: BaseException):
    """This handler is used to handle exceptions raised by the Mobius Pipeline and Aana application.

    Args:
        request (Request): The request object
        exc (BaseException): The exception raised

    Returns:
        JSONResponse: JSON response with the error details
    """
    return custom_exception_handler(request, exc)


@app.exception_handler(RayTaskError)
async def ray_task_error_handler(request: Request, exc: RayTaskError):
    """This handler is used to handle RayTaskError exceptions.

    Args:
        request (Request): The request object
        exc (RayTaskError): The exception raised

    Returns:
        JSONResponse: JSON response with the error details. The response contains the following fields:
            error: The name of the exception class.
            message: The message of the exception.
            stacktrace: The stacktrace of the exception.
    """
    error = exc.__class__.__name__
    stacktrace = traceback.format_exc()

    return AanaJSONResponse(
        status_code=400,
        content=ExceptionResponseModel(
            error=error, message=str(exc), stacktrace=stacktrace
        ).dict(),
    )
