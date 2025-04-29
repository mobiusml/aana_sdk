import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from ray.exceptions import RayTaskError
from starlette.middleware.base import BaseHTTPMiddleware

from aana.api.responses import AanaJSONResponse
from aana.configs.settings import settings as aana_settings
from aana.core.models.exception import ExceptionResponseModel
from aana.exceptions.core import BaseException


def get_app_middleware(
    app: FastAPI, middleware_class: type
) -> BaseHTTPMiddleware | None:
    """Get middleware instance by class from FastAPI app.

    Args:
        app (FastAPI): The FastAPI application
        middleware_class (type): The middleware class to find

    Returns:
        Optional[BaseHTTPMiddleware]: The middleware instance if found, None otherwise
    """
    middleware_index = None
    for index, middleware in enumerate(app.user_middleware):
        if middleware.cls == middleware_class:
            middleware_index = index
            break
    if middleware_index is None:
        return None

    middleware = app.user_middleware[middleware_index]
    return middleware.cls(app, *middleware.args, **middleware.kwargs)


def add_cors_headers(request: Request, response: AanaJSONResponse):
    """Add CORS headers to response based on app CORS middleware configuration.

    Args:
        request (Request): The request object
        response (AanaJSONResponse): The response object to add headers to
    """
    request_origin = request.headers.get("origin")
    if request_origin is None:
        return

    cors_middleware: CORSMiddleware = get_app_middleware(
        app=request.app, middleware_class=CORSMiddleware
    )
    if not cors_middleware:
        return

    response.headers.update(cors_middleware.simple_headers)
    has_cookie = "cookie" in request.headers

    if (cors_middleware.allow_all_origins and has_cookie) or (
        not cors_middleware.allow_all_origins
        and cors_middleware.is_allowed_origin(origin=request_origin)
    ):
        cors_middleware.allow_explicit_origin(response.headers, request_origin)


async def validation_exception_handler(
    request: Request, exc: ValidationError | RequestValidationError
):
    """This handler is used to handle pydantic validation errors.

    Args:
        request (Request): The request object
        exc (ValidationError | RequestValidationError): The exception raised

    Returns:
        JSONResponse: JSON response with the error details
    """
    if isinstance(exc, ValidationError):
        data = exc.errors(include_context=False)
    elif isinstance(exc, RequestValidationError):
        data = exc.errors()
        # Remove ctx from the error messages
        for error in data:
            if "ctx" in error:
                error.pop("ctx")
    response = AanaJSONResponse(
        status_code=422,
        content=ExceptionResponseModel(
            error="ValidationError", message="Validation error", data=data
        ).model_dump(),
    )
    add_cors_headers(request, response)
    return response


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
    # Remove the stacktrace if it is disabled
    if not aana_settings.include_stacktrace:
        stacktrace = None
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

    response = AanaJSONResponse(
        status_code=status_code,
        content=ExceptionResponseModel(
            error=error, message=message, data=data, stacktrace=stacktrace
        ).model_dump(),
    )

    if request:  # Only add CORS headers if we have a request object
        add_cors_headers(request, response)
    return response


async def aana_exception_handler(request: Request, exc: Exception):
    """This handler is used to handle exceptions raised by the Aana application.

    Args:
        request (Request): The request object
        exc (Exception): The exception raised

    Returns:
        JSONResponse: JSON response with the error details
    """
    return custom_exception_handler(request, exc)
