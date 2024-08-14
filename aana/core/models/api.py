from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SDKStatus(str, Enum):
    """The status of the SDK."""

    UNHEALTHY = "UNHEALTHY"
    RUNNING = "RUNNING"
    DEPLOYING = "DEPLOYING"


class SDKStatusResponse(BaseModel):
    """The response for the SDK status endpoint.

    Attributes:
        status (SDKStatus): The status of the SDK.
        message (str): The message for more information like error message.
    """

    status: SDKStatus = Field(description="The status of the SDK.")
    message: str = Field(
        description="The message for more information like error message."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "The response for the SDK status endpoint.",
            "examples": [
                {
                    "status": "RUNNING",
                    "message": "",
                },
                {
                    "status": "DEPLOYING",
                    "message": "",
                },
                {
                    "status": "UNHEALTHY",
                    "message": "Error: Lowercase (lowercase_deployment): A replica's health check failed. "
                    "This deployment will be UNHEALTHY until the replica recovers or a new deploy happens.",
                },
            ],
        }
    )
