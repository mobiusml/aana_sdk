from collections.abc import Callable
from types import CoroutineType, NoneType
from typing import get_type_hints

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.utils.asyncio import run_async
from aana.utils.typing import is_typed_dict
from haystack import component


def typehints_to_component_types(
    typehints: dict[str, type],
) -> tuple[dict[str, type], dict[str, type]]:
    """Converts a type hint dictionary into something that can be consumed by Haystack.

    Arguments:
        typehints (dict): a dict returned by `typing.get_type_hints()`

    Returns:
        tuple[dict[str,type], dict[str,type]]: Suitable for passing to the haystack component interface.
    """
    output_types = typehints_to_output_types(typehints.pop("return", NoneType))
    input_types = typehints_to_input_types(typehints)
    return input_types, output_types


def typehints_to_output_types(typehint: type) -> dict[str, type]:
    """Converts a return type hint into something that can be consumed by Haystack.

    Arguments:
        typehint: Return type from `typing.get_type_hints()["return"]`

    Returns:
        dict[str, type]: Something that can be passed to `haystack.component.set_output_types()`
    """
    # If no return value, return an empty dictionary
    if typehint is None:
        return {}
    # If annotation is a TypedDict, turn it into a regular dict
    if is_typed_dict(typehint):
        return get_type_hints(typehint)
    # If it's a class with annotations, return the __annotations__ dict
    annotations = getattr(typehint, "__annotations__", None)
    if annotations:
        return annotations
    # Otherwise wrap it into a dictionary
    return {"return": typehint}


def typehints_to_input_types(typehints: dict[str, type]) -> dict[str, type]:
    """Converts a typehint dictionary into something that can be used by Haystack.

    Arguments:
        typehints (dict[str, type]): Type hints from `typing.get_type_hints()` *minus the "return" key*

    Returns:
        dict[str, type]: Something that can be consumed by `haystack.set_input_types()`
    """
    # If typehint is None, or an empty dict, return an empty dict
    if not typehints:
        return {}
    # Otherwise just return the input
    return typehints


@component
class AanaDeploymentComponent:
    """Wrapper for Aana deployments to run as HayStack Components.

    Example:
        ```python
        deployment_handle = await AanaDeploymentHandle.create("my_deployment")
        haystack_component = AanaDeploymentComponent(deployment_handle, "my_method")
        haystack_component.warm_up()  # This is currently a no-op, but subject to change.
        component_result = haystack_component.run(my_input_prompt="This is an input prompt")
        ```
    """

    _deployment_handle: AanaDeploymentHandle
    _run_method: Callable
    _warm: bool

    def __init__(self, deployment_handle: AanaDeploymentHandle, method_name: str):
        """Constructor.

        Arguments:
            deployment_handle (AanaDeploymentHandle): the Aana Ray deployment to be wrapped (must be a class Deployment)
            method_name (str): the name of the method on the deployment to call inside the component's `run()` method.
        """
        self._deployment_handle = deployment_handle

        # Determine input and output types for `run()`
        # Will raise if the function is not defined (e.g. if you pass a function deployment)
        self.run_method = self._get_method(method_name)
        if not self.run_method:
            raise AttributeError(name=method_name, obj=self._deployment_handle)
        hints = get_type_hints(self.run_method)
        input_types, output_types = typehints_to_component_types(hints)
        # The functions `set_input_types()` and `set_output_types()`
        # take an positional instance argument and keyword arguments
        component.set_input_types(self, **input_types)
        component.set_output_types(self, **output_types)

    def warm_up(self):
        """Warms up the deployment to a ready state.

        As we run off an existing deployment handle, this is currently a no-op.
        """
        self._warm = True

    def run(self, *args, **kwargs):
        """Run the component. This is the primary interface for Haystack Components.

        Arguments:
            *args: the arguments to pass to the deployment run function
            **kwargs: the keyword arguments to pass to the deployment run function

        Returns:
            The return value of the deployment's run function
        """
        # Function may (must?) be a coroutine. Resolve it if so.
        return run_async(self._call(*args, **kwargs))

    def _call(self, *args, **kwargs) -> CoroutineType:
        """Calls the deployment's run method. Not public, use the `run()` method."""
        return self.run_method(*args, **kwargs)  # type: ignore

    def _get_method(self, method_name: str) -> Callable | None:
        """Gets a handle to the method specified by the constructor."""
        instance_method_handle: Callable | None = getattr(
            self._deployment_handle, method_name, None
        )
        return instance_method_handle
