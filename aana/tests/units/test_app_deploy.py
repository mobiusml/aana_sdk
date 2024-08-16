# ruff: noqa: S101, S113
from typing import Any

import pytest
from ray import serve

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.runtime import FailedDeployment, InsufficientResources


@serve.deployment
class DummyFailingDeployment(BaseDeployment):
    """Simple deployment that fails on initialization."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration to the deployment and initialize it."""
        raise Exception("Dummy exception")  # noqa: TRY002, TRY003


@serve.deployment
class Lowercase(BaseDeployment):
    """Simple deployment that lowercases the text."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration to the deployment and initialize it."""
        pass

    async def lower(self, text: str) -> dict:
        """Lowercase the text.

        Args:
            text (str): The text to lowercase

        Returns:
            dict: The lowercase text
        """
        return {"text": [t.lower() for t in text]}


def test_failed_deployment(create_app):
    """Test that a failed deployment raises a FailedDeployment exception."""
    deployments = [
        {
            "name": "deployment",
            "instance": DummyFailingDeployment.options(num_replicas=1, user_config={}),
        }
    ]
    with pytest.raises(FailedDeployment):
        create_app(deployments, [])


def test_insufficient_resources(create_app):
    """Test that deployment fails when there are insufficient resources to deploy."""
    deployments = [
        {
            "name": "deployment",
            "instance": Lowercase.options(
                num_replicas=1,
                ray_actor_options={"num_gpus": 100},  # requires 100 GPUs
                user_config={},
            ),
        }
    ]
    with pytest.raises(InsufficientResources):
        create_app(deployments, [])
