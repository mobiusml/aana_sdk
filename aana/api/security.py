import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import select

from aana.configs.settings import settings as aana_settings
from aana.core.models.api_service import ApiKey
from aana.exceptions.api_service import (
    AdminOnlyAccess,
    ApiKeyExpired,
    ApiKeyNotFound,
    ApiKeyNotProvided,
    ApiKeyValidationFailed,
    InactiveSubscription,
)
from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.session import GetDbDependency

logger = logging.getLogger(__name__)


async def extract_api_key_info(request: Request, db: GetDbDependency) -> ApiKey | None:
    """Get the API key info dependency."""
    if aana_settings.api_service.enabled:
        path = request.url.path
        if path.startswith("/__worker/"):
            # For worker (internal) requests, use user_id from headers
            # because API key might change while the the task is queued
            # but the user_id remains the same.
            user_id = request.headers.get("user_id")

            if not user_id:
                raise ApiKeyNotProvided()  # Replace with a more appropriate exception

            try:
                result = await db.execute(
                    select(ApiKeyEntity).where(ApiKeyEntity.user_id == user_id)
                )
                api_key_info = result.scalars().first()
            except Exception as e:
                raise ApiKeyValidationFailed() from e
        else:
            api_key = request.headers.get("x-api-key")

            if not api_key:
                raise ApiKeyNotProvided()

            try:
                result = await db.execute(
                    select(ApiKeyEntity).where(ApiKeyEntity.api_key == api_key)
                )
                api_key_info = result.scalars().first()
            except Exception as e:
                raise ApiKeyValidationFailed() from e

        if not api_key_info:
            raise ApiKeyNotFound(key=api_key)

        if api_key_info.expired_at < datetime.now(timezone.utc):
            raise ApiKeyExpired(key=api_key)

        return api_key_info.to_model()
    return None


ApiKeyInfoDependency = Annotated[ApiKey | None, Depends(extract_api_key_info)]
""" Dependency to get the API key info. """


async def extract_user_id(api_key_info: ApiKeyInfoDependency) -> str | None:
    """Get the user ID dependency."""
    return api_key_info.user_id if api_key_info else None


UserIdDependency = Annotated[str | None, Depends(extract_user_id)]
""" Dependency to get the user ID. """


async def is_admin(api_key_info: ApiKeyInfoDependency) -> bool:
    """Check if the user is an admin.

    Args:
        api_key_info (ApiKeyInfoDependency): Dependency to get the API key info

    Returns:
        bool: True if the user is an admin, False otherwise
    """
    return api_key_info.is_admin if api_key_info else False


IsAdminDependency = Annotated[bool, Depends(is_admin)]
""" Dependency to check if the user is an admin. """


async def require_admin_access(is_admin: IsAdminDependency) -> bool:
    """Check if the user is an admin. If not, raise an exception.

    Args:
        is_admin (IsAdminDependency): Dependency to check if the user is an admin

    Raises:
        AdminOnlyAccess: If the user is not an admin
    """
    if not is_admin:
        raise AdminOnlyAccess()
    return True


AdminAccessDependency = Annotated[bool, Depends(require_admin_access)]
""" Dependency to check if the user is an admin. If not, it will raise an exception. """


async def require_active_subscription(api_key_info: ApiKeyInfoDependency) -> bool:
    """Check if the user has an active subscription. If not, raise an exception.

    Args:
        api_key_info (ApiKeyInfoDependency): Dependency to get the API key info

    Raises:
        InactiveSubscription: If the user does not have an active subscription
    """
    if not api_key_info.is_subscription_active:
        raise InactiveSubscription(key=api_key_info.api_key)
    return True


ActiveSubscriptionRequiredDependency = Annotated[
    bool, Depends(require_active_subscription)
]
""" Dependency to check if the user has an active subscription. If not, it will raise an exception. """
