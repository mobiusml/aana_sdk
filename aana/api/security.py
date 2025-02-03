from typing import Annotated

from fastapi import Depends, Request

from aana.configs.settings import settings as aana_settings
from aana.exceptions.api_service import AdminOnlyAccess


def check_admin_permissions(request: Request):
    """Check if the user is an admin.

    Args:
        request (Request): The request object

    Raises:
        AdminOnlyAccess: If the user is not an admin
    """
    if aana_settings.api_service.enabled:
        api_key_info = request.state.api_key_info
        is_admin = api_key_info.get("is_admin", False)
        if not is_admin:
            raise AdminOnlyAccess()


class AdminCheck:
    """Dependency to check if the user is an admin."""

    async def __call__(self, request: Request) -> bool:
        """Check if the user is an admin."""
        check_admin_permissions(request)
        return True


AdminRequired = Annotated[bool, Depends(AdminCheck())]
""" Annotation to check if the user is an admin. If not, it will raise an exception. """
