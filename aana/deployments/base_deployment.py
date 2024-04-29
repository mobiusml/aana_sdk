import inspect
from typing import Any

from aana.configs.settings import settings
from aana.utils.test import check_test_cache_enabled


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
        if (
            settings.test.test_mode
            and settings.test.use_deployment_cache
            and check_test_cache_enabled(self)
        ):
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
