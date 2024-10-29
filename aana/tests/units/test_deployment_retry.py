# ruff: noqa: S101, S113

import pytest
from ray import serve

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment, exception_handler


@serve.deployment(health_check_period_s=1, health_check_timeout_s=30)
class Lowercase(BaseDeployment):
    """Ray deployment that returns the lowercase version of a text."""

    def __init__(self):
        """Initialize the deployment."""
        super().__init__()
        self.num_requests = 0

    @exception_handler
    async def lower(self, text: str) -> dict:
        """Lowercase the text.

        Args:
            text (str): The text to lowercase

        Returns:
            dict: The lowercase text
        """
        # Only every 3rd request should be successful
        self.num_requests += 1
        if self.num_requests % 3 != 0:
            raise Exception("Random exception")  # noqa: TRY002, TRY003

        return {"text": text.lower()}


deployments = [
    {
        "name": "lowercase_deployment",
        "instance": Lowercase,
    }
]


@pytest.mark.asyncio
async def test_deployment_retry(create_app):
    """Test the Ray Serve app."""
    create_app(deployments, [])

    text = "Hello, World!"

    # Get deployment handle without retries
    handle = await AanaDeploymentHandle.create(
        "lowercase_deployment", retry_exceptions=False
    )

    # test the lowercase deployment fails
    with pytest.raises(Exception):  # noqa: B017
        await handle.lower(text=text)

    # Get deployment handle with retries
    handle = await AanaDeploymentHandle.create(
        "lowercase_deployment", retry_exceptions=True
    )

    # test the lowercase deployment works
    response = await handle.lower(text=text)
    assert response == {"text": text.lower()}
