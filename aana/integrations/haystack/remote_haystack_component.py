
from haystack import component

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.utils.asyncio import run_async


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
        if hasattr(self, "handle"):
            return
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
