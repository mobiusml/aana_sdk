from aana.utils.general import load_options


class BaseDeployment:
    """
    Base class for all deployments.
    We can use this class to define common methods for all deployments.
    For example, we can connect to the database here or download artifacts.
    """

    def __init__(self):
        self.config = None
        self.configured = False

    async def reconfigure(self, config):
        """
        Reconfigure the deployment.
        The method is called when the deployment is updated.
        """
        self.config = config
        # go through the config and try to load options
        for key in self.config:
            self.config[key] = load_options(self.config[key], ignore_errors=True)
        await self.apply_config()
        self.configured = True

    async def apply_config(self):
        """
        Apply the configuration.
        """
        pass
