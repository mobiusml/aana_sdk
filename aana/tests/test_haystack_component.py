import pytest

from aana.components.deployment_component import AanaDeploymentComponent
from aana.configs.deployments import deployments


@pytest.mark.parametrize("deployment_name", ["stablediffusion2_deployment"])
def test_haystack_wrapper(deployment_name):
    """Tests haystack wrapper for deployments."""
    component = AanaDeploymentComponent(deployments[deployment_name])
    component.warm_up()
    # TODO: run functionality
