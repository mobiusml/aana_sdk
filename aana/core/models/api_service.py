from typing import Annotated

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class ApiKey(BaseModel):
    """Pydantic model for API key entity."""

    api_key: str
    user_id: str
    subscription_id: str
    is_subscription_active: bool


ApiKeyType = SkipJsonSchema[Annotated[ApiKey, Field(default=None)]]
"""
Type with optional API key information.

Can be None if API service is disabled. Otherwise, it will be an instance of `ApiKey`.

Attributes:
    api_key (str): The API key.
    user_id (str): The user ID.
    subscription_id (str): The subscription ID.
    is_subscription_active (bool): Flag indicating if the subscription is active.
"""
