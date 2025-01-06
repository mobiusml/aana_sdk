import asyncio
import inspect
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from inspect import isasyncgenfunction
from typing import Annotated, Any, get_origin

import orjson
from fastapi import Body, FastAPI, Form, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, Field, ValidationError, create_model
from pydantic.main import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from aana.api.event_handlers.event_handler import EventHandler
from aana.api.event_handlers.event_manager import EventManager
from aana.api.exception_handler import custom_exception_handler
from aana.api.responses import AanaJSONResponse
from aana.configs.settings import settings as aana_settings
from aana.core.models.exception import ExceptionResponseModel
from aana.storage.repository.task import TaskRepository
from aana.storage.session import get_session
from aana.utils.json import jsonify


def get_default_values(func):
    """Get the default values for the function arguments."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


@dataclass
class Endpoint:
    """Class used to represent an endpoint.

    Attributes:
        name (str): Name of the endpoint.
        path (str): Path of the endpoint (e.g. "/video/transcribe").
        summary (str): Description of the endpoint that will be shown in the API documentation.
        event_handlers (list[EventHandler] | None): The list of event handlers to register for the endpoint.
    """

    name: str
    path: str
    summary: str
    initialized: bool = False
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

        app.post(
            self.path,
            name=self.name,
            summary=self.summary,
            operation_id=self.name,
            response_model=ResponseModel,
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
        return create_model(model_name, **input_fields)

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
            data: type[BaseModel], files: dict[str, bytes] | None = None, defer=False
        ):
            if not self.initialized:
                await self.initialize()

            if event_manager:
                event_manager.handle(bound_path, defer=defer)

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

                with get_session() as session:
                    task = TaskRepository(session).save(
                        endpoint=bound_path, data=data_dict
                    )
                    return AanaJSONResponse(content={"task_id": str(task.id)})

            if isasyncgenfunction(self.run):

                async def generator_wrapper() -> AsyncGenerator[bytes, None]:
                    """Serializes the output of the generator using ORJSONResponseCustom."""
                    try:
                        async for output in self.run(**data_dict):
                            yield AanaJSONResponse(content=output).body
                    except Exception as e:
                        yield custom_exception_handler(None, e).body

                return StreamingResponse(
                    generator_wrapper(), media_type="application/json"
                )
            else:
                try:
                    output = await self.run(**data_dict)
                except Exception as e:
                    return custom_exception_handler(None, e)
                return AanaJSONResponse(content=output)

        async def route_func(
            request: Request,
            # body: str = Form(...),
            # json_body: dict = Body(...),
            # defer: bool = Query(
            #     description="Defer execution of the endpoint to the task queue.",
            #     default=False,
            #     include_in_schema=aana_settings.task_queue.enabled,
            # ),
        ):
            print("headers", request.headers)
            content_type = request.headers.get("content-type")
            print("content_type", content_type)

            json_body = None
            form_data = None
            if content_type == "application/json":
                json_body = await request.json()
            else:
                form_data = await request.form()

            defer = request.query_params.get("defer", False)

            print("form_data", form_data)
            print("json_body", json_body)

            files: dict[str, bytes] = {}
            if form_data:
                body = form_data.get("body")
                if not body:
                    raise ValueError("body key not found in form data")
                for field_name, field_value in form_data.items():
                    if isinstance(field_value, StarletteUploadFile):
                        files[field_name] = await field_value.read()

                # parse form data as a pydantic model and validate it
                data = RequestModel.model_validate_json(body)
                return await route_func_body(data=data, files=files, defer=defer)
            else:
                # body = json_body
                body = jsonify(json_body)

                if "data" not in json_body:
                    raise ValueError("data key not found in json_data")

                if not isinstance(json_body["data"], list):
                    raise ValueError("data key must be a list")

                data = {}
                futures = {}
                responses = {}
                for row in json_body["data"]:
                    if not isinstance(row, list):
                        raise ValueError("data row must be a list")
                    if len(row) < 2:
                        raise ValueError("data row must have at least 2 elements")
                    if not isinstance(row[0], int):
                        raise ValueError("data row first element must be an integer")
                    row_id = row[0]

                    try:
                        data[row_id] = RequestModel.model_validate(
                            dict(
                                zip(
                                    RequestModel.model_fields.keys(),
                                    row[1:],
                                    strict=False,
                                )
                            )
                        )
                        futures[row_id] = asyncio.create_task(
                            route_func_body(data=data[row_id], files=files, defer=defer)
                        )
                    except Exception as e:
                        # custom_exception_handler(None, e)
                        responses[row_id] = custom_exception_handler(None, e)

                # Wait for all futures to complete
                for row_id, future in futures.items():
                    response = await future
                    responses[row_id] = response

                responses_rows = []
                for row_id, response in responses.items():
                    if not isinstance(response, AanaJSONResponse):
                        raise ValueError("Unsupported response type")
                    response_data = orjson.loads(response.body)
                    responses_rows.append([row_id, response_data])
                return AanaJSONResponse(content={"data": responses_rows})

        return route_func


def add_custom_schemas_to_openapi_schema(
    openapi_schema: dict[str, Any], custom_schemas: dict[str, Any]
) -> dict[str, Any]:
    """Add custom schemas to the openapi schema.

    File upload is that FastAPI doesn't support Pydantic models in multipart requests.
    There is a discussion about it on FastAPI discussion forum.
    See https://github.com/tiangolo/fastapi/discussions/8406
    The topic starter suggests a workaround.
    The workaround is to use Forms instead of Pydantic models in the endpoint definition and
    then convert the Forms to Pydantic models in the endpoint itself
    using parse_raw_as function from Pydantic.
    Since Pydantic model isn't used in the endpoint definition,
    the API documentation will not be generated automatically.
    So the workaround also suggests updating the API documentation manually
    by overriding the openapi method of a FastAPI application.

    Args:
        openapi_schema (dict): The openapi schema.
        custom_schemas (dict): The custom schemas.

    Returns:
        dict: The openapi schema with the custom schemas added.
    """
    if "$defs" not in openapi_schema:
        openapi_schema["$defs"] = {}
    # for schema_name, schema in custom_schemas.items():

    # "requestBody": {
    #     "required": true,
    #     "content": {
    #         "multipart/form-data": {
    #             "schema": {
    #                 "$ref": "#/components/schemas/Body_episodic_summary"
    #             }
    #         }
    #     }
    # },

    #     # if we have a definitions then we need to move them out to the top level of the schema
    #     if "$defs" in schema:
    #         openapi_schema["$defs"].update(schema["$defs"])
    #         del schema["$defs"]
    #     openapi_schema["components"]["schemas"][f"Body_{schema_name}"]["properties"][
    #         "body"
    #     ] = schema
    return openapi_schema
