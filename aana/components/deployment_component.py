import asyncio
from collections.abc import Callable
from typing import Any, TypeAlias, get_type_hints

import ray
from haystack import component
from pydantic import BaseModel
from ray.serve.deployment import Deployment

from aana.utils.typing import as_dict_of_types, is_typed_dict

DeploymentResult: TypeAlias = Any


def typehints_to_component_types(
    typehints: dict[str, type],
) -> tuple[dict[str, type], dict[str, type]]:
    """Converts a type hint dictionary into something that can be consumed by Haystack.

    Arguments:
        typehints (dict): a dict returned by `typing.get_type_hints()`

    Returns:
        tuple: Something that can be passed to `haystack.component.set_input_types()`
    """
    output_types = typehints_to_output_types(typehints.pop("return"))
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
        return as_dict_of_types(typehint)
    # If it's a single value, wrap into a dict
    # (Not sure if this is correct, Haystack docs are unclear) -EdR
    if not isinstance(typehint, dict):
        return {"return": typehint}
    # Otherwise just return the dictionary
    return typehint


def typehints_to_input_types(typehints: dict[str]) -> dict[str]:
    """Converts a typehint dictionary into something that can be used by Haystack.

    Arguments:
        typehints (dict[str, type]): Type hints from `typing.get_type_hints()` *minus the "return" key*

    Returns:
        dict[str, type]: Something that can be consumed by `haystack.set_input_types()`
    """
    # If typehint is None, or an emtpy dict, return an empty dict
    if not typehints:
        return {}
    # Otherwise just return the input
    # return {key: value for key, value in typehints.items()}
    return typehints


@component
class AanaDeploymentComponent:
    """Wrapper for Aana deployments to run as HayStack Components."""

    # TODO: add usage example to docstring

    _deployment: Deployment
    _config: BaseModel
    _inference_method: Callable
    _warm: bool

    def __init__(self, deployment: Deployment, method_name="generate_batch"):
        """Constructor.

        Arguments:
            deployment (Deployment): the Ray deployment to be wrapped
            config (BaseModel): the config for the deploy to set in its `apply_config()` method.
            method_name (str): the name of the method on the deployment to call inside the component's `run()` method. Defaults to `generate_batch`
        """
        self._deployment = deployment
        self.config = deployment.user_config
        self._warm = False
        self._method_name = method_name
        self._serve_handle = None
        self._run_method = None

        # Determine input and output types for `run()`
        method_handle = self._get_method()
        if not method_handle:
            raise ValueError(self._method_name)
        hints = get_type_hints(method_handle)
        input_types, output_types = typehints_to_component_types(hints)
        # The functions `set_input_types()` and `set_output_types()`
        # take an positionial instance argument and keyword arguments
        component.set_input_types(self, **input_types)
        component.set_output_types(self, **output_types)

    def warm_up(self):
        """Warms up the deployment to a ready state.

        Usually this is to load the model and initialize any preallocated data or parameters.
        """
        if not self._warm:
            self._warm = True

            # Start ray if it isn't already
            # @movchan74 Do we do this here or elsewhere?
            ray.init(ignore_reinit_error=True)
            # Run the deployment
            self._serve_handle = ray.serve.run(self._deployment.bind())
            self._run_method = getattr(self._serve_handle, self._method_name, None)
            if not self._run_method:
                # Should not happen but just in case
                raise ValueError(self._method_name)
            # self._serve_handle.apply_config(self._config)

    def run(self, *args, **kwargs) -> DeploymentResult:
        """Run the component. This is the primary interface for Haystack Components.

        Arguments:
            *args: the arguments to pass to the deployment run function
            **kwargs: the keyword arguments to pass to the deployment run function

        Returns:
            DeploymentResult: The return value of the deployment's run function
        """
        retval = self._call(*args, **kwargs)
        # If the run function returns a coroutine, run it in an event loop
        if asyncio.iscoroutine(retval):
            return asyncio.run(retval)
        return retval

    # Is this supported?
    async def arun(self, *args, **kwargs) -> asyncio.Task[DeploymentResult]:
        """Run the component as async. This is the async version of the primary interface for Haystack Components."""
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs) -> DeploymentResult:
        """Calls the deployment's run method. Not public, use the `run()` method."""
        return self._run_method(*args, **kwargs).remote()  # type: ignore

    def _get_method(self) -> Callable | None:
        """Gets a handle to the method specified by the constructor."""
        # This is the type/constructor of the deployment
        underlying_deployment_t = self._deployment.func_or_class
        class_method_handle = getattr(underlying_deployment_t, self._method_name, None)
        return class_method_handle  # type: ignore
