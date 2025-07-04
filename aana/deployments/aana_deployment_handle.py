import logging

import ray
from ray import serve

from aana.utils.core import sleep_exponential_backoff
from aana.utils.typing import is_async_generator

logger = logging.getLogger(__name__)


class AanaDeploymentHandle:
    """A handle to interact with a deployed Aana deployment.

    Use create method to create a deployment handle.

    ```python
    deployment_handle = await AanaDeploymentHandle.create("deployment_name")
    ```

    Attributes:
        handle (ray.serve.handle.DeploymentHandle): Ray Serve deployment handle.
        deployment_name (str): The name of the deployment.
    """

    def __init__(
        self,
        deployment_name: str,
        num_retries: int = 3,
        retry_exceptions: bool | list[Exception] = False,
        retry_delay: float = 0.2,
        retry_max_delay: float = 2.0,
    ):
        """A handle to interact with a deployed Aana deployment.

        Args:
            deployment_name (str): The name of the deployment.
            num_retries (int): The maximum number of retries for the method.
            retry_exceptions (bool | list[Exception]): Whether to retry on application-level errors or a list of exceptions to retry on.
            retry_delay (float): The initial delay between retries.
            retry_max_delay (float): The maximum delay between retries.
        """
        self.handle = serve.get_app_handle(deployment_name)
        self.deployment_name = deployment_name
        self.__methods = None
        self.num_retries = num_retries
        self.retry_exceptions = retry_exceptions
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay

    def _is_exception_retryable(self, e: Exception) -> bool:
        """Check if the exception is retryable based on the configuration."""
        if self.retry_exceptions is True:
            return True
        if isinstance(self.retry_exceptions, list):
            return isinstance(e, tuple(self.retry_exceptions)) or isinstance(
                getattr(e, "cause", None), tuple(self.retry_exceptions)
            )
        return False

    def __create_async_method(self, name: str):  # noqa: C901
        """Create an method to interact with the deployment.

        Args:
            name (str): The name of the method.
        """
        method_info = self.__methods[name]
        annotations = method_info.get("annotations", {})
        return_type = annotations.get("return", None)

        if is_async_generator(return_type):

            async def method(*args, **kwargs):
                retries = 0
                while retries <= self.num_retries:
                    try:
                        async for item in self.handle.options(
                            method_name=name, stream=True
                        ).remote(*args, **kwargs):
                            yield item
                        break
                    except Exception as e:
                        is_retryable = self._is_exception_retryable(e)
                        logger.exception(
                            f"Error in method {name} on attempt {retries + 1}: {e.__class__}"
                        )
                        if not is_retryable or retries >= self.num_retries:
                            raise
                        await sleep_exponential_backoff(
                            initial_delay=self.retry_delay,
                            max_delay=self.retry_max_delay,
                            attempts=retries,
                        )
                        retries += 1

        else:

            async def method(*args, **kwargs):
                retries = 0
                while retries <= self.num_retries:
                    try:
                        return await self.handle.options(method_name=name).remote(
                            *args, **kwargs
                        )
                    except Exception as e:  # noqa: PERF203
                        is_retryable = self._is_exception_retryable(e)
                        logger.exception(
                            f"Error in method {name} on attempt {retries + 1}: {e.__class__}"
                        )
                        if not is_retryable or retries >= self.num_retries:
                            raise
                        await sleep_exponential_backoff(
                            initial_delay=self.retry_delay,
                            max_delay=self.retry_max_delay,
                            attempts=retries,
                        )
                        retries += 1

        if "annotations" in self.__methods[name]:
            method.__annotations__ = self.__methods[name]["annotations"]
        if "doc" in self.__methods[name]:
            method.__doc__ = self.__methods[name]["doc"]
        return method

    async def __load_methods(self):
        """Load the methods available in the deployment."""
        self.__methods = await self.handle.get_methods.remote()
        for name in self.__methods:
            setattr(self, name, self.__create_async_method(name))

    @classmethod
    async def create(
        cls,
        deployment_name: str,
        num_retries: int = 3,
        retry_exceptions: bool | list[Exception] = False,
        retry_delay: float = 0.2,
        retry_max_delay: float = 2.0,
    ):
        """Create a deployment handle.

        Args:
            deployment_name (str): The name of the deployment to interact with.
            num_retries (int): The maximum number of retries for the method.
            retry_exceptions (bool | list[Exception]): Whether to retry on application-level errors or a list of exceptions to retry on.
            retry_delay (float): The initial delay between retries.
            retry_max_delay (float): The maximum delay between retries.
        """
        # Always retry on ActorDiedError to handle actor restarts gracefully.
        if retry_exceptions is False or isinstance(retry_exceptions, list):
            retry_exceptions = (
                retry_exceptions if isinstance(retry_exceptions, list) else []
            )
            retry_exceptions = [*retry_exceptions, ray.exceptions.ActorDiedError]

        handle = cls(
            deployment_name=deployment_name,
            num_retries=num_retries,
            retry_exceptions=retry_exceptions,
            retry_delay=retry_delay,
            retry_max_delay=retry_max_delay,
        )
        await handle.__load_methods()
        return handle
