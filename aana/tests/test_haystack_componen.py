from unittest.mock import MagicMock

from aana.components.deployment_component import AanaDeploymentComponent


def test_haystack_component():
    """Tests the haystack component."""
    deployment_handle = MagicMock()
    component = AanaDeploymentComponent(deployment_handle, "foo")
    component.warm_up()
    component.run()
    deployment_handle.foo.assert_called_once()
