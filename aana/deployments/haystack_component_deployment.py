from typing import Any

from pydantic import BaseModel
from ray import serve

from aana.deployments.base_deployment import BaseDeployment, exception_handler
from aana.utils.core import import_from_path


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

    @exception_handler
    async def run(self, **data: dict[str, Any]) -> dict[str, Any]:
        """Run the model on the input data."""
        return self.component.run(**data)

    async def get_sockets(self):
        """Get the input and output sockets of the component."""
        return {
            "output": self.component.__haystack_output__._sockets_dict,
            "input": self.component.__haystack_input__._sockets_dict,
        }
