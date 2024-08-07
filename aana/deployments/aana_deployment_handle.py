from ray import serve

from aana.utils.typing import is_async_generator


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

    def __init__(self, deployment_name: str):
        """A handle to interact with a deployed Aana deployment.

        Args:
            deployment_name (str): The name of the deployment.
        """
        self.handle = serve.get_app_handle(deployment_name)
        self.deployment_name = deployment_name
        self.__methods = None

    def __create_async_method(self, name: str):
        """Create an method to interact with the deployment.

        Args:
            name (str): The name of the method.
        """
        method_info = self.__methods[name]
        annotations = method_info.get("annotations", {})
        return_type = annotations.get("return", None)

        if is_async_generator(return_type):

            async def method(*args, **kwargs):
                async for item in self.handle.options(
                    method_name=name, stream=True
                ).remote(*args, **kwargs):
                    yield item
        else:

            async def method(*args, **kwargs):
                return await self.handle.options(method_name=name).remote(
                    *args, **kwargs
                )

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
    async def create(cls, deployment_name: str):
        """Create a deployment handle.

        Args:
            deployment_name (str): The name of the deployment to interact with.
        """
        handle = cls(deployment_name)
        await handle.__load_methods()
        return handle
