from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from ray.serve.schema import ApplicationStatus


class SDKStatus(str, Enum):
    """The status of the SDK."""

    UNHEALTHY = "UNHEALTHY"
    RUNNING = "RUNNING"
    DEPLOYING = "DEPLOYING"


class DeploymentStatus(BaseModel):
    """The status of a deployment."""

    status: ApplicationStatus = Field(description="The status of the deployment.")
    message: str = Field(
        description="The message for more information like error message."
    )


class SDKStatusResponse(BaseModel):
    """The response for the SDK status endpoint.

    Attributes:
        status (SDKStatus): The status of the SDK.
        message (str): The message for more information like error message.
        deployments (dict[str, DeploymentStatus]): The status of each deployment in the Aana app.
    """

    status: SDKStatus = Field(description="The status of the SDK.")
    message: str = Field(
        description="The message for more information like error message."
    )
    deployments: dict[str, DeploymentStatus] = Field(
        description="The status of each deployment in the Aana app."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "The response for the SDK status endpoint.",
            "examples": [
                {
                    "status": "RUNNING",
                    "message": "",
                    "deployments": {
                        "app": {
                            "status": "RUNNING",
                            "message": "",
                        },
                        "lowercase_deployment": {
                            "status": "RUNNING",
                            "message": "",
                        },
                    },
                },
                {
                    "status": "UNHEALTHY",
                    "message": "Error: Lowercase (lowercase_deployment): A replica's health check failed. "
                    "This deployment will be UNHEALTHY until the replica recovers or a new deploy happens.",
                    "deployments": {
                        "app": {
                            "status": "RUNNING",
                            "message": "",
                        },
                        "lowercase_deployment": {
                            "status": "UNHEALTHY",
                            "message": "A replica's health check failed. This deployment will be UNHEALTHY "
                            "until the replica recovers or a new deploy happens.",
                        },
                    },
                },
            ],
        }
    )
