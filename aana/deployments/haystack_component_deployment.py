from typing import Any

from haystack import component
from pydantic import BaseModel
from ray import serve

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment
from aana.utils.asyncio import run_async
from aana.utils.core import import_from_path


@component
class RemoteHaystackComponent:
    """A component that connects to a remote Haystack component created by HaystackComponentDeployment.

    Attributes:
        deployment_name (str): The name of the deployment to use.
    """

    def __init__(
        self,
        deployment_name: str,
    ):
        """Initialize the component.

        Args:
            deployment_name (str): The name of the deployment to use.
        """
        self.deployment_name = deployment_name

    def warm_up(self):
        """Warm up the component.

        This will properly initialize the component by creating a handle to the deployment
        and setting the input and output types.
        """
        self.handle = run_async(AanaDeploymentHandle.create(self.deployment_name))
        sockets = run_async(self.handle.get_sockets())
        component.set_input_types(
            self, **{socket.name: socket.type for socket in sockets["input"].values()}
        )
        component.set_output_types(
            self, **{socket.name: socket.type for socket in sockets["output"].values()}
        )

    def run(self, **data):
        """Run the component on the input data."""
        return run_async(self.handle.run(**data))


class HaystackComponentDeploymentConfig(BaseModel):
    """Configuration for the HaystackComponentDeployment.

    Attributes:
        component (str): The path to the Haystack component to deploy.
        params (dict): The parameters to pass to the component on initialization (model etc).
    """

    component: str
    params: dict[str, Any]


@serve.deployment
class HaystackComponentDeployment(BaseDeployment):
    """Deployment to deploy a Haystack component."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It creates the Haystack component and warms it up.

        The configuration should conform to the HaystackComponentDeploymentConfig schema.
        """
        config_obj = HaystackComponentDeploymentConfig(**config)

        self.params = config_obj.params
        self.component_path = config_obj.component
        self.component = import_from_path(config_obj.component)(**config_obj.params)

        self.component.warm_up()

    async def run(self, **data: dict[str, Any]) -> dict[str, Any]:
        """Run the model on the input data."""
        return self.component.run(**data)

    async def get_sockets(self):
        """Get the input and output sockets of the component."""
        return {
            "output": self.component.__haystack_output__._sockets_dict,
            "input": self.component.__haystack_input__._sockets_dict,
        }
