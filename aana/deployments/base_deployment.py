from typing import Any


class BaseDeployment:
    """Base class for all deployments.

    We can use this class to define common methods for all deployments.
    For example, we can connect to the database here or download artifacts.
    """

    def __init__(self):
        """Inits to unconfigured state."""
        self.config = None
        self.configured = False

    async def reconfigure(self, config: dict[str, Any]):
        """Reconfigure the deployment.

        The method is called when the deployment is updated.
        """
        self.config = config
        await self.apply_config(config)
        self.configured = True

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        Args:
            config (dict): the configuration
        """
        raise NotImplementedError
