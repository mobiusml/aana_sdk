import inspect
from typing import Any


class BaseDeployment:
    """Base class for all deployments.

    To create a new deployment, inherit from this class and implement the `apply_config` method
    and your custom methods like `generate`, `predict`, etc.
    """

    def __init__(self):
        """Inits to unconfigured state."""
        self.config = None
        self._configured = False

    async def reconfigure(self, config: dict[str, Any]):
        """Reconfigure the deployment.

        The method is called when the deployment is updated.
        """
        self.config = config
        await self.apply_config(config)
        self._configured = True

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        This method is called when the deployment is created or updated.

        Define the logic to load the model and configure it here.

        Args:
            config (dict): the configuration
        """
        raise NotImplementedError

    async def get_methods(self) -> dict:
        """Returns the methods of the deployment.

        Returns:
            dict: the methods of the deployment with annotations and docstrings
        """
        cls = self.__class__
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        methods_info = {}
        for name, method in methods:
            # Skip private methods
            if name.startswith("_"):
                continue
            # Skip non-asynchronous methods
            if not (
                inspect.iscoroutinefunction(method)
                or inspect.isasyncgenfunction(method)
            ):
                continue

            methods_info[name] = {}
            if method.__annotations__:
                methods_info[name]["annotations"] = method.__annotations__
            if method.__doc__:
                methods_info[name]["doc"] = method.__doc__
        return methods_info
