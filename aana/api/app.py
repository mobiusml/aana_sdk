from fastapi import FastAPI, Request
from pydantic import ValidationError

from aana.api.exception_handler import (
    aana_exception_handler,
    validation_exception_handler,
)
from aana.configs.settings import settings as aana_settings
from aana.exceptions.api_service import (
    ApiKeyNotFound,
    ApiKeyNotProvided,
    ApiKeyValidationFailed,
    InactiveSubscription,
)
from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.session import get_session

app = FastAPI()

app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, aana_exception_handler)


@app.middleware("http")
async def api_key_check(request: Request, call_next):
    """Middleware to check the API key and subscription status."""
    excluded_paths = ["/openapi.json", "/docs", "/redoc"]
    if request.url.path in excluded_paths:
        return await call_next(request)

    if aana_settings.api_service.enabled:
        api_key = request.headers.get("x-api-key")

        if not api_key:
            raise ApiKeyNotProvided()

        with get_session() as session:
            try:
                api_key_info = (
                    session.query(ApiKeyEntity).filter_by(api_key=api_key).first()
                )
            except Exception as e:
                raise ApiKeyValidationFailed() from e

            if not api_key_info:
                raise ApiKeyNotFound(key=api_key)

            if not api_key_info.is_subscription_active:
                raise InactiveSubscription(key=api_key)

            request.state.api_key_info = api_key_info.to_dict()

    response = await call_next(request)
    return response


def get_api_key_info(request: Request) -> dict | None:
    """Get the API key info dependency."""
    return getattr(request.state, "api_key_info", None)


def get_user_id(request: Request) -> str | None:
    """Get the user ID dependency."""
    api_key_info = get_api_key_info(request)
    return api_key_info.get("user_id") if api_key_info else None
