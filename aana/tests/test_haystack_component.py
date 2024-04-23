import pytest

from aana.components.deployment_component import AanaDeploymentComponent
from aana.configs.deployments import deployments


@pytest.mark.parametrize(
    "deployment_name, method_name",
    [
        ("stablediffusion2_deployment", "generate"),
        ("hf_blip2_deployment_opt_2_7b", "generate_batch"),
    ],
)
def test_haystack_wrapper(deployment_name, method_name):
    """Tests haystack wrapper for deployments."""
    _component = AanaDeploymentComponent(deployments[deployment_name], method_name)
    # TODO: run functionality


@pytest.mark.parametrize(
    "deployment_name, missing_method_name",
    [("stablediffusion2_deployment", "generate_batch")],
)
def test_haystack_wrapper_fails(deployment_name, missing_method_name):
    """Tests that haystack wrapper raises if method_name is missing."""
    with pytest.raises(ValueError):
        _component = AanaDeploymentComponent(
            deployments[deployment_name], missing_method_name
        )
