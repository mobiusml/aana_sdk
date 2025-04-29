from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlalchemy import select

from aana.api.exception_handler import (
    aana_exception_handler,
    validation_exception_handler,
)
from aana.configs.settings import settings as aana_settings
from aana.exceptions.api_service import (
    ApiKeyNotFound,
    ApiKeyNotProvided,
    ApiKeyValidationFailed,
)
from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.session import get_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=aana_settings.cors.allow_origins,
    allow_origin_regex=aana_settings.cors.allow_origin_regex,
    allow_credentials=aana_settings.cors.allow_credentials,
    allow_methods=aana_settings.cors.allow_methods,
    allow_headers=aana_settings.cors.allow_headers,
)

app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, aana_exception_handler)


@app.middleware("http")
async def api_key_check(request: Request, call_next):
    """Middleware to check the API key and subscription status."""
    excluded_paths = ["/openapi.json", "/docs", "/redoc"]
    if request.url.path in excluded_paths or request.method == "OPTIONS":
        return await call_next(request)

    if aana_settings.api_service.enabled:
        api_key = request.headers.get("x-api-key")

        if not api_key:
            raise ApiKeyNotProvided()

        async with get_session() as session:
            try:
                result = await session.execute(
                    select(ApiKeyEntity).where(ApiKeyEntity.api_key == api_key)
                )
                api_key_info = result.scalars().first()
            except Exception as e:
                raise ApiKeyValidationFailed() from e

            if not api_key_info:
                raise ApiKeyNotFound(key=api_key)

            request.state.api_key_info = api_key_info.to_model()

    response = await call_next(request)
    return response
