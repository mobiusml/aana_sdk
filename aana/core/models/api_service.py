
from pydantic import BaseModel


class ApiKey(BaseModel):
    """Pydantic model for API key entity."""

    api_key: str
    user_id: str
    subscription_id: str
    is_subscription_active: bool
