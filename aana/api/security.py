from typing import Annotated

from fastapi import Depends, Request

from aana.configs.settings import settings as aana_settings
from aana.core.models.api_service import ApiKey
from aana.exceptions.api_service import AdminOnlyAccess, InactiveSubscription


def is_admin(request: Request) -> bool:
    """Check if the user is an admin.

    Args:
        request (Request): The request object

    Returns:
        bool: True if the user is an admin, False otherwise
    """
    if aana_settings.api_service.enabled:
        api_key_info: ApiKey = request.state.api_key_info
        return api_key_info.is_admin if api_key_info else False
    return True


def require_admin_access(request: Request) -> bool:
    """Check if the user is an admin. If not, raise an exception.

    Args:
        request (Request): The request object

    Raises:
        AdminOnlyAccess: If the user is not an admin
    """
    _is_admin = is_admin(request)
    if not _is_admin:
        raise AdminOnlyAccess()
    return True


def extract_api_key_info(request: Request) -> ApiKey | None:
    """Get the API key info dependency."""
    return getattr(request.state, "api_key_info", None)


def extract_user_id(request: Request) -> str | None:
    """Get the user ID dependency."""
    api_key_info = extract_api_key_info(request)
    return api_key_info.user_id if api_key_info else None


def require_active_subscription(request: Request) -> bool:
    """Check if the user has an active subscription. If not, raise an exception.

    Args:
        request (Request): The request object

    Raises:
        InactiveSubscription: If the user does not have an active subscription
    """
    if aana_settings.api_service.enabled:
        api_key_info: ApiKey = request.state.api_key_info
        if not api_key_info.is_subscription_active:
            raise InactiveSubscription(key=api_key_info.api_key)
    return True


AdminAccessDependency = Annotated[bool, Depends(require_admin_access)]
""" Dependency to check if the user is an admin. If not, it will raise an exception. """

IsAdminDependency = Annotated[bool, Depends(is_admin)]
""" Dependency to check if the user is an admin. """

UserIdDependency = Annotated[str | None, Depends(extract_user_id)]
""" Dependency to get the user ID. """

ApiKeyInfoDependency = Annotated[ApiKey | None, Depends(extract_api_key_info)]
""" Dependency to get the API key info. """

ActiveSubscriptionRequiredDependency = Annotated[
    bool, Depends(require_active_subscription)
]
""" Dependency to check if the user has an active subscription. If not, it will raise an exception. """
