import pytest

from aana.components.deployment_component import DeploymentComponent
from aana.configs.deployments import deployments


@pytest.fixture("deployment_name", ["stablediffusion2_deployment"])
def test_haystack_wrapper(deployment_name):
    """Tests haystack wrapper for deployments."""
    component = DeploymentComponent(deployments[deployment_name])
    component.warm_up()
    # TODO: run functionality
