import inspect
import os
from functools import wraps
from typing import Any

import ray
from ray import serve

from aana.configs.settings import settings as aana_settings
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.exceptions.runtime import InferenceException


def exception_handler(func):
    """AanaDeploymentHandle decorator to catch exceptions and store them in the deployment for health check purposes.

    Args:
        func (function): The function to decorate.

    Returns:
        function: The decorated function
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        self.num_requests_since_last_health_check += 1
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            self.raised_exceptions.append(e)
            raise

    @wraps(func)
    async def wrapper_generator(self, *args, **kwargs):
        self.num_requests_since_last_health_check += 1
        try:
            async for item in func(self, *args, **kwargs):
                yield item
        except Exception as e:
            self.raised_exceptions.append(e)
            raise

    if inspect.isasyncgenfunction(func):
        return wrapper_generator
    else:
        return wrapper


class BaseDeployment:
    """Base class for all deployments.

    To create a new deployment, inherit from this class and implement the `apply_config` method
    and your custom methods like `generate`, `predict`, etc.
    """

    def __init__(self):
        """Inits to unconfigured state."""
        self.config = None
        self._configured = False
        self.num_requests_since_last_health_check = 0
        self.raised_exceptions = []
        self.restart_exceptions = [InferenceException]

    async def request_gpu_from_manager(self):
        """Request GPU resources for the deployment from the GPU manager."""
        num_gpus = ray.get_runtime_context().get_assigned_resources().get("GPU", 0)
        # If no GPUs are assigned, skip
        if num_gpus == 0:
            return

        replica_context = serve.get_replica_context()
        self._replica_id = replica_context.replica_id.unique_id
        self._deployment_name = replica_context.replica_id.deployment_id.app_name

        gpu_manager_handle = await AanaDeploymentHandle.create("gpu_manager")
        gpu_ids = await gpu_manager_handle.request_gpu(
            deployment=self._deployment_name,
            replica_id=self._replica_id,
            num_gpus=num_gpus,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    async def reconfigure(self, config: dict[str, Any]):
        """Reconfigure the deployment.

        The method is called when the deployment is updated.
        """
        self.config = config

        if aana_settings.gpu_manager.enabled:
            await self.request_gpu_from_manager()

        await self.apply_config(config)
        self._configured = True

    async def check_health(self):
        """Check the health of the deployment.

        Raises:
            Raises the exception that caused the deployment to be unhealthy.
        """
        raised_restart_exceptions = [
            exception
            for exception in self.raised_exceptions
            if exception.__class__ in self.restart_exceptions
        ]
        # Restart the deployment if more than 50% of the requests raised restart exceptions
        if self.num_requests_since_last_health_check != 0:
            ratio_restart_exceptions = (
                len(raised_restart_exceptions)
                / self.num_requests_since_last_health_check
            )
            if ratio_restart_exceptions > 0.5:
                raise raised_restart_exceptions[0]

        self.raised_exceptions = []
        self.num_requests_since_last_health_check = 0

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        This method is called when the deployment is created or updated.

        Define the logic to load the model and configure it here.

        Args:
            config (dict): the configuration
        """
        raise NotImplementedError(
            "apply_config method must be implemented in the deployment class."
        )

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
