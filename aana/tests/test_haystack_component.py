import pytest

from aana.components.deployment_component import AanaDeploymentComponent
from aana.configs.deployments import deployments


@pytest.mark.parametrize(
    "deployment_name, method_name", [("stablediffusion2_deployment", "generate")]
)
def test_haystack_wrapper(deployment_name, method_name):
    """Tests haystack wrapper for deployments."""
    _component = AanaDeploymentComponent(deployments[deployment_name], method_name)
    # TODO: run functionality
