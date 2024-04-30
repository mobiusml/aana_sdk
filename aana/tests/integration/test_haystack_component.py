import pytest

from aana.components.deployment_component import AanaDeploymentComponent
from aana.configs.deployments import available_deployments as deployments
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.tests.deployments.test_hf_blip2_deployment import setup_hf_blip2_deployment
from aana.tests.deployments.test_stablediffusion2_deployment import (
    setup_deployment as setup_stablediffusion2_deployment,
)
from aana.tests.utils import is_gpu_available, is_using_deployment_cache
from aana.utils.coroutines import run_sync


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize(
    "setup_deployment, deployment_name, method_name",
    [
        (setup_stablediffusion2_deployment, "stablediffusion2_deployment", "generate"),
        (setup_hf_blip2_deployment, "hf_blip2_deployment_opt_2_7b", "generate_batch"),
    ],
)
def test_haystack_wrapper(setup_deployment, deployment_name, method_name):
    """Tests haystack wrapper for deployments."""
    run_sync(AanaDeploymentHandle.create(deployment_name))
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
