from typing import Any

from aana.configs.settings import settings
from aana.utils.general import is_testing


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
        if is_testing() and settings.use_deployment_cache:
            # If we are in testing mode and we want to use the cache,
            # we don't need to load the model
            self.configured = True
            return
        else:
            await self.apply_config(config)
            self.configured = True

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        Args:
            config (dict): the configuration
        """
        raise NotImplementedError
