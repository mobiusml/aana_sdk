import traceback
from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mobius_pipeline.exceptions import PipelineException
from pydantic import ValidationError
from ray.exceptions import RayTaskError

from aana.exceptions.general import AanaException

app = FastAPI()


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    This handler is used to handle pydantic validation errors

    Args:
        request (Request): The request object
        exc (ValidationError): The validation error

    Returns:
        JSONResponse: JSON response with the error details
    """
    # TODO: Structure the error response so that it is consistent with the other error responses
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


def custom_exception_handler(
    request: Request, exc_raw: Union[PipelineException, AanaException, RayTaskError]
):
    """
    This handler is used to handle custom exceptions raised in the application.
    PipelineException is the exception raised by the Mobius Pipeline.
    AanaException is the exception raised by the Aana application.
    Sometimes custom exception are wrapped into RayTaskError so we need to handle that as well.

    Args:
        request (Request): The request object
        exc_raw (Union[PipelineException, AanaException, RayTaskError]): The exception raised

    Returns:
        JSONResponse: JSON response with the error details. The response contains the following fields:
            error: The name of the exception class.
            message: The message of the exception.
            data: The additional data returned by the exception that can be used to identify the error (e.g. image path, url, model name etc.)
            stacktrace: The stacktrace of the exception.
    """
    # a PipelineException or AanaException can be wrapped into a RayTaskError
    if isinstance(exc_raw, RayTaskError):
        # str(e) returns whole stack trace
        # if exception is a RayTaskError
        # let's use it to get the stack trace
        stacktrace = str(exc_raw)
        # get the original exception
        exc: Union[PipelineException, AanaException] = exc_raw.cause
        assert isinstance(exc, (PipelineException, AanaException))
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
    return JSONResponse(
        status_code=400,
        content={
            "error": error,
            "message": message,
            "data": data,
            "stacktrace": stacktrace,
        },
    )


@app.exception_handler(PipelineException)
async def pipeline_exception_handler(request: Request, exc: PipelineException):
    """
    This handler is used to handle exceptions raised by the Mobius Pipeline.

    Args:
        request (Request): The request object
        exc (PipelineException): The exception raised

    Returns:
        JSONResponse: JSON response with the error details
    """
    return custom_exception_handler(request, exc)


@app.exception_handler(AanaException)
async def aana_exception_handler(request: Request, exc: AanaException):
    """
    This handler is used to handle exceptions raised by the Aana application.

    Args:
        request (Request): The request object
        exc (AanaException): The exception raised

    Returns:
        JSONResponse: JSON response with the error details
    """
    return custom_exception_handler(request, exc)


@app.exception_handler(RayTaskError)
async def ray_task_error_handler(request: Request, exc: RayTaskError):
    """
    This handler is used to handle RayTaskError exceptions.

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

    return JSONResponse(
        status_code=400,
        content={"error": error, "message": str(exc), "stacktrace": stacktrace},
    )
