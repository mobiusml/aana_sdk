import inspect
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from inspect import isasyncgenfunction
from typing import Annotated, Any, get_origin

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, Field, ValidationError, create_model
from pydantic.main import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from aana.api.event_handlers.event_handler import EventHandler
from aana.api.event_handlers.event_manager import EventManager
from aana.api.exception_handler import custom_exception_handler
from aana.api.responses import AanaConcatenatedJSONResponse, AanaJSONResponse
from aana.api.security import require_active_subscription, require_admin_access
from aana.configs.settings import settings as aana_settings
from aana.core.models.api_service import ApiKey
from aana.core.models.exception import ExceptionResponseModel
from aana.routers.task import TaskResponse
from aana.storage.repository.task import TaskRepository
from aana.storage.session import get_session

logger = logging.getLogger(__name__)


def get_default_values(func):
    """Get the default values for the function arguments."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class DeferOption(str, Enum):
    """Enum for defer option.

    Attributes:
        ALWAYS (str): Always defer. Endpoints with this option will always be defer execution to the task queue.
        NEVER (str): Never defer. Endpoints with this option will never be defer execution to the task queue.
        OPTIONAL (str): Optionally defer. Endpoints with this option can be defer execution to the task queue if
            the defer query parameter is set to True.
    """

    ALWAYS = "always"
    NEVER = "never"
    OPTIONAL = "optional"


@dataclass
class Endpoint:
    """Class used to represent an endpoint.

    Attributes:
        name (str): Name of the endpoint.
        path (str): Path of the endpoint (e.g. "/video/transcribe").
        summary (str): Short summary of the endpoint that will used as the title in the API documentation.
        description (str | None): Detailed description of the endpoint that will be shown in the API documentation. Default is None.
        admin_required (bool): Flag indicating if the endpoint requires admin access.
        active_subscription_required (bool): Flag indicating if the endpoint requires an active subscription.
        defer_option (DeferOption): Defer option for the endpoint (always, never, optional).
        tags (list[str]): List of tags for the endpoint. Default is ["app"].
        openapi_extra (dict[str, Any] | None): Extra OpenAPI information for the endpoint.
        event_handlers (list[EventHandler] | None): The list of event handlers to register for the endpoint.
    """

    name: str
    path: str
    summary: str
    description: str | None = None
    admin_required: bool = False
    active_subscription_required: bool = False
    defer_option: DeferOption = DeferOption.OPTIONAL
    initialized: bool = False
    tags: list[str] = field(default_factory=lambda: ["app"])
    openapi_extra: dict[str, Any] | None = None
    event_handlers: list[EventHandler] | None = None

    async def initialize(self):
        """Initialize the endpoint.

        Redefine this method to add initialization logic for the endpoint (e.g. create a handle to the deployment).
        Call super().initialize() to ensure the endpoint is initialized.

        Example:
            ```python
            async def initialize(self):
                await super().initialize()
                self.asr_handle = await AanaDeploymentHandle.create("whisper_deployment")
            ```
        """
        self.initialized = True

    async def run(self, *args, **kwargs):
        """The main method of the endpoint that is called when the endpoint receives a request.

        Redefine this method to implement the logic of the endpoint.

        Example:
            ```python
            async def run(self, video: VideoInput) -> WhisperOutput:
                video_obj: Video = await run_remote(download_video)(video_input=video)
                transcription = await self.asr_handle.transcribe(audio=audio)
                return transcription
            ```
        """
        raise NotImplementedError

    def register(
        self, app: FastAPI, custom_schemas: dict[str, dict], event_manager: EventManager
    ):
        """Register the endpoint in the FastAPI application.

        Args:
            app (FastAPI): FastAPI application.
            custom_schemas (dict[str, dict]): Dictionary containing custom schemas.
            event_manager (EventManager): Event manager for the application.
        """
        RequestModel = self.get_request_model()
        ResponseModel = self.get_response_model()

        if self.event_handlers:
            for handler in self.event_handlers:
                event_manager.register_handler_for_events(handler, [self.path])

        route_func = self.__create_endpoint_func(
            RequestModel=RequestModel,
            event_manager=event_manager,
        )

        if self.defer_option == DeferOption.ALWAYS:
            ResponseModel = TaskResponse

        if self.is_streaming_response():
            response_class = AanaConcatenatedJSONResponse
        else:
            response_class = AanaJSONResponse

        app.post(
            self.path,
            name=self.name,
            summary=self.summary,
            description=self.description,
            operation_id=self.name,
            response_model=ResponseModel,
            response_class=response_class,
            tags=self.tags,
            openapi_extra=self.openapi_extra,
            responses={
                400: {"model": ExceptionResponseModel},
            },
        )(route_func)
        custom_schemas[self.name] = RequestModel.model_json_schema()

    def __generate_model_name(self, suffix: str) -> str:
        """Generate a Pydantic model name based on a given suffix.

        Args:
            suffix (str): Suffix for the model name (e.g. "Request", "Response").

        Returns:
            str: Generated model name.
        """
        return self.__class__.__name__ + suffix

    def __arg_to_field(self, arg_name: str, arg_type: Any) -> tuple[Any, Any]:
        """Convert an argument to a Pydantic field.

        Args:
            arg_name (str): Name of the argument.
            arg_type (Any): Type of the argument.

        Returns:
            Tuple[Any, Any]: Tuple containing the Pydantic model and the Pydantic field.
        """
        # if data model is None or Any, set it to Any
        if arg_type is None or arg_type == Any:
            return (Any, Field(None))

        if get_origin(arg_type) is Annotated:
            default = get_default_values(self.run).get(arg_name)
            if default is not None:
                return (arg_type, default)
            else:
                return (arg_type, ...)

        # try to instantiate the data model
        # to see if any of the fields are required
        try:
            data_model_instance = arg_type()
        except TypeError as e:
            raise ValueError(  # noqa: TRY003
                f"Invalid type for argument {arg_name}: {arg_type}. "
                "Consider using Annotated[..., Field(...)] or Pydantic models."
            ) from e
        except ValidationError:
            # if we can't instantiate the data model
            # it means that it has required fields
            return (arg_type, ...)
        else:
            return (arg_type, data_model_instance)

    def __get_input_fields(self) -> dict[str, tuple[Any, Any]]:
        """Generate fields for the request Pydantic model based on the function annotations.

        Returns:
            dict[str, tuple[Any, Field]]: Dictionary of fields for the request Pydantic model.
        """
        fields = {}
        for arg_name, arg_type in self.run.__annotations__.items():
            if arg_name == "return":
                continue
            fields[arg_name] = self.__arg_to_field(arg_name, arg_type)
        return fields

    def get_request_model(self) -> type[BaseModel]:
        """Generate the request Pydantic model for the endpoint.

        Returns:
            type[BaseModel]: Request Pydantic model.
        """
        model_name = self.__generate_model_name("Request")
        input_fields = self.__get_input_fields()
        return create_model(
            model_name, **input_fields, __config__=ConfigDict(extra="forbid")
        )

    def __get_output_fields(self) -> dict[str, tuple[Any, Any]]:
        """Generate fields for the response Pydantic model based on the function annotations.

        Returns:
            dict[str, tuple[Any, Any]]: Dictionary of fields for the response Pydantic model.
        """
        try:
            if self.is_streaming_response():
                return_type = self.run.__annotations__["return"].__args__[0]
            else:
                return_type = self.run.__annotations__["return"]
        except (AttributeError, KeyError) as e:
            raise ValueError("Endpoint function must have a return annotation.") from e  # noqa: TRY003
        fields = {}
        for arg_name, arg_type in return_type.__annotations__.items():
            fields[arg_name] = (arg_type, Field(None))
        return fields

    def get_response_model(self) -> type[BaseModel]:
        """Generate the response Pydantic model for the endpoint.

        Returns:
            type[BaseModel]: Response Pydantic model.
        """
        model_name = self.__generate_model_name("Response")
        output_fields = self.__get_output_fields()
        return create_model(
            model_name, **output_fields, __config__=ConfigDict(extra="forbid")
        )

    @classmethod
    def is_streaming_response(cls) -> bool:
        """Check if the endpoint returns a streaming response.

        Returns:
            bool: True if the endpoint returns a streaming response, False otherwise.
        """
        return isasyncgenfunction(cls.run)

    def __create_endpoint_func(  # noqa: C901
        self,
        RequestModel: type[BaseModel],
        event_manager: EventManager | None = None,
    ) -> Callable:
        """Create a function for routing an endpoint."""
        # Copy path to a bound variable so we don't retain an external reference
        bound_path = self.path

        async def route_func_body(  # noqa: C901
            request: Request,
            body: str,
            files: dict[str, bytes] | None = None,
            defer=False,
        ):
            if not self.initialized:
                await self.initialize()

            if event_manager:
                event_manager.handle(bound_path, defer=defer)

            # parse form data as a pydantic model and validate it
            data = RequestModel.model_validate_json(body)

            # Add api_key_info to the body if API service is enabled
            api_key_info: ApiKey | None = None
            if aana_settings.api_service.enabled:
                api_key_info = request.state.api_key_info
                api_key_field = next(
                    (
                        field_name
                        for field_name, field in RequestModel.model_fields.items()
                        if field.annotation is ApiKey
                    ),
                    None,
                )
                if api_key_field:
                    setattr(data, api_key_field, api_key_info)

            # if the input requires file upload, add the files to the data
            if files:
                for field_name in data.model_fields:
                    field_value = getattr(data, field_name)
                    if hasattr(field_value, "set_files"):
                        field_value.set_files(files)

            # We have to do this instead of data.dict() because
            # data.dict() will convert all nested models to dicts
            # and we want to keep them as pydantic models
            data_dict = {}
            for field_name in data.model_fields:
                field_value = getattr(data, field_name)
                data_dict[field_name] = field_value

            if defer:
                if not aana_settings.task_queue.enabled:
                    raise RuntimeError("Task queue is not enabled.")  # noqa: TRY003

                async with get_session() as session:
                    task = await TaskRepository(session).save(
                        endpoint=bound_path,
                        data=data_dict,
                        user_id=api_key_info.user_id if api_key_info else None,
                    )
                    return AanaJSONResponse(content={"task_id": str(task.id)})

            if isasyncgenfunction(self.run):

                async def generator_wrapper() -> AsyncGenerator[bytes, None]:
                    """Serializes the output of the generator using ORJSONResponseCustom."""
                    try:
                        async for output in self.run(**data_dict):
                            yield AanaConcatenatedJSONResponse(content=output).body
                    except Exception as e:
                        yield custom_exception_handler(request, e).body

                return StreamingResponse(
                    generator_wrapper(),
                    media_type="application/json-seq",
                    headers={
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                try:
                    output = await self.run(**data_dict)
                except Exception as e:
                    return custom_exception_handler(request, e)
                return AanaJSONResponse(content=output)

        async def route_func(
            request: Request,
            body: str = Form(...),
            defer: bool = Query(
                description="Defer execution of the endpoint to the task queue.",
                default=False,
                include_in_schema=aana_settings.task_queue.enabled
                and self.defer_option == DeferOption.OPTIONAL,
            ),
        ):
            if aana_settings.api_service.enabled and self.admin_required:
                require_admin_access(request)

            if aana_settings.api_service.enabled and self.active_subscription_required:
                require_active_subscription(request)

            if self.defer_option == DeferOption.ALWAYS:
                defer = True
            elif self.defer_option == DeferOption.NEVER:
                defer = False

            form_data = await request.form()

            # Parse files from the form data
            files: dict[str, bytes] = {}
            for field_name, field_value in form_data.items():
                if isinstance(field_value, StarletteUploadFile):
                    files[field_name] = await field_value.read()

            return await route_func_body(
                request=request, body=body, files=files, defer=defer
            )

        return route_func

    def send_usage_event(
        self, api_key: ApiKey, metric: str, properties: dict[str, Any]
    ):
        """Send a usage event to the LAGO API service.

        Args:
            api_key (ApiKey): The API key information.
            metric (str): The metric code.
            properties (dict): The properties of the event (usage data, e.g. {"count": 10}).
        """
        from lago_python_client.client import Client
        from lago_python_client.models import Event
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
        )

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(Exception),
        )
        def send_event_with_retry(client, event):
            return client.events.create(event)

        if not aana_settings.api_service.enabled:
            logger.warning("API service is not enabled. Skipping usage event.")
            return

        if (
            not aana_settings.api_service.lago_url
            or not aana_settings.api_service.lago_api_key
        ):
            logger.warning(
                "LAGO API service URL or API key is not set. Skipping usage event."
            )
            return

        try:
            client = Client(
                api_key=aana_settings.api_service.lago_api_key,
                api_url=aana_settings.api_service.lago_url,
            )

            event = Event(
                transaction_id=str(uuid.uuid4()),
                code=metric,
                external_subscription_id=api_key.subscription_id,
                timestamp=time.time(),
                properties=properties,
            )

            send_event_with_retry(client, event)
        except Exception as e:
            logger.error(
                f"Failed to send usage event after retries: {e}", exc_info=True
            )
