# ruff: noqa: S101, S113
import asyncio

import pytest
from ray import serve

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment, exception_handler
from aana.exceptions.runtime import InferenceException


@serve.deployment(health_check_period_s=1, health_check_timeout_s=30)
class Lowercase(BaseDeployment):
    """Ray deployment that returns the lowercase version of a text."""

    def __init__(self):
        """Initialize the deployment."""
        super().__init__()
        self.active = True

    @exception_handler
    async def lower(self, text: str) -> dict:
        """Lowercase the text.

        Args:
            text (str): The text to lowercase

        Returns:
            dict: The lowercase text
        """
        if text == "inference_exception" or not self.active:
            self.active = False
            raise InferenceException(model_name="lowercase_deployment")

        return {"text": text.lower()}


deployments = [
    {
        "name": "lowercase_deployment",
        "instance": Lowercase,
    }
]


@pytest.mark.asyncio
async def test_deployment_restart(create_app):
    """Test the Ray Serve app."""
    create_app(deployments, [])

    handle = await AanaDeploymentHandle.create("lowercase_deployment")

    text = "Hello, World!"

    # test the lowercase deployment works
    response = await handle.lower(text=text)
    assert response == {"text": text.lower()}

    # Cause an InferenceException in the deployment and make it inactive.
    # After the deployment is inactive, the deployment should always raise an InferenceException.
    with pytest.raises(InferenceException):
        await handle.lower(text="inference_exception")

    # The deployment should restart and work again, wait for around 60 seconds for the deployment to restart.
    for _ in range(60):
        await asyncio.sleep(1)
        try:
            response = await handle.lower(text=text)
            if response == {"text": text.lower()}:
                break
        except:  # noqa: S110
            pass

    assert response == {"text": text.lower()}
