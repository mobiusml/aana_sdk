# ruff: noqa: S101
import pytest

from aana.components.deployment_component import AanaDeploymentComponent
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.tests.deployments.test_stablediffusion2_deployment import (
    setup_deployment as setup_stablediffusion2_deployment,  # noqa: F401
)
from aana.tests.utils import is_gpu_available, is_using_deployment_cache
from aana.utils.coroutines import run_sync


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
def test_haystack_wrapper(setup_stablediffusion2_deployment):  # noqa: F811
    """Tests haystack wrapper for deployments."""
    deployment_name = "sd2_deployment"
    method_name = "generate"
    result_key = "image"
    deployment_handle = run_sync(AanaDeploymentHandle.create(deployment_name))
    component = AanaDeploymentComponent(deployment_handle, method_name)
    result = component.run(prompt="foo")
    assert result_key in result, result


def test_haystack_wrapper_fails(setup_stablediffusion2_deployment):  # noqa: F811
    """Tests that haystack wrapper raises if method_name is missing."""
    deployment_name = "sd2_deployment"
    missing_method_name = "does_not_exist"
    deployment_handle = run_sync(AanaDeploymentHandle.create(deployment_name))
    with pytest.raises(ValueError):
        _component = AanaDeploymentComponent(deployment_handle, missing_method_name)
