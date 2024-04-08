import contextlib
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from mobius_pipeline.node.socket import Socket
from mobius_pipeline.pipeline.pipeline import Pipeline
from pydantic import BaseModel, Field, ValidationError, create_model

from aana.api.app import custom_exception_handler
from aana.api.event_handlers.event_handler import EventHandler
from aana.api.event_handlers.event_manager import EventManager
from aana.api.responses import AanaJSONResponse
from aana.exceptions.general import (
    HandlerAlreadyRegisteredException,
    MultipleFileUploadNotAllowed,
)
from aana.models.pydantic.exception_response import ExceptionResponseModel


async def run_pipeline(
    pipeline: Pipeline, data: dict, required_outputs: list[str]
) -> dict[str, Any]:
    """This function is used to run a Mobius Pipeline.

    It creates a container from the data, runs the pipeline and returns the output.

    Args:
        pipeline (Pipeline): The pipeline to run.
        data (dict): The data to create the container from.
        required_outputs (list[str]): The required outputs of the pipeline.

    Returns:
        dict[str, Any]: The output of the pipeline and the execution time of the pipeline.
    """
    # create a container from the data
    container = pipeline.parse_dict(data)

    # run the pipeline
    output, execution_time = await pipeline.run(
        container, required_outputs, return_execution_time=True
    )
    output["execution_time"] = execution_time
    return output


async def run_pipeline_streaming(
    pipeline: Pipeline,
    data: dict,
    requested_partial_outputs: list[str],
    requested_full_outputs: list[str],
) -> AsyncGenerator[dict[str, Any], None]:
    """This function is used to run a Mobius Pipeline as a generator.

    It creates a container from the data, runs the pipeline as a generator and yields the output.

    Args:
        pipeline (Pipeline): The pipeline to run.
        data (dict): The data to create the container from.
        requested_partial_outputs (list[str]): The required partial outputs of the pipeline that should be streamed.
        requested_full_outputs (list[str]): The required full outputs of the pipeline that should be returned at the end.

    Yields:
        dict[str, Any]: The output of the pipeline and the execution time of the pipeline.
    """
    # create a container from the data
    container = pipeline.parse_dict(data)

    # run the pipeline
    async for output in pipeline.run_generator(
        container, requested_partial_outputs, requested_full_outputs
    ):
        yield output


@dataclass
class OutputFilter:
    """Class used to represent an output filter.

    The output filter is a parameter that will be added to the request
    and will allow to choose subset of `outputs` to return.

    Attributes:
        name (str): Name of the output filter.
        description (str): Description of the output filter.
    """

    name: str
    description: str


@dataclass
class FileUploadField:
    """Class used to represent a file upload field.

    Attributes:
        name (str): Name of the field.
        description (str): Description of the field.
    """

    name: str
    description: str


@dataclass
class EndpointOutput:
    """Class used to represent an endpoint output.

    Attributes:
        name (str): Name of the output that should be returned by the endpoint.
        output (str): Output of the pipeline that should be returned by the endpoint.
    """

    name: str
    output: str
    streaming: bool = False


@dataclass
class Endpoint:
    """Class used to represent an endpoint.

    Attributes:
        name (str): Name of the endpoint.
        path (str): Path of the endpoint.
        summary (str): Description of the endpoint that will be shown in the API documentation.
        outputs (List[EndpointOutput]): List of outputs that should be returned by the endpoint.
        output_filter (Optional[OutputFilter]): The parameter will be added to the request and
                                will allow to choose subset of `outputs` to return.
        streaming (bool): Whether the endpoint outputs a stream of data.
        event_handlers (list[EventHandler]): list of event handlers to regist for this endpoint (optional)

    """

    name: str
    path: str
    summary: str
    outputs: list[EndpointOutput]
    output_filter: OutputFilter | None = None
    streaming: bool = False
    event_handlers: list[EventHandler] | None = None

    def __post_init__(self):
        """Post init method.

        Creates dictionaries for fast lookup of outputs.
        """
        self.name_to_output = {output.name: output.output for output in self.outputs}
        self.output_to_name = {output.output: output.name for output in self.outputs}
        self.streaming_outputs = {
            output.output for output in self.outputs if output.streaming
        }

    def generate_model_name(self, suffix: str) -> str:
        """Generate a Pydantic model name based on a given suffix.

        Parameters:
            suffix (str): Suffix for the model name (e.g. "Request", "Response").

        Returns:
            str: Generated model name.
        """
        return "".join([word.capitalize() for word in self.name.split("_")]) + suffix

    def socket_to_field(self, socket: Socket) -> tuple[Any, Any]:
        """Convert a socket to a Pydantic field.

        Parameters:
            socket (Socket): Socket to convert.

        Returns:
            Tuple[Any, Field]: Tuple of the socket's data model and a Pydantic field.
        """
        data_model = socket.data_model

        # if data model is None or Any, set it to Any
        if data_model is None or data_model == Any:
            data_model = Any
            return (data_model, Field(None))

        # try to instantiate the data model
        # to see if any of the fields are required
        try:
            data_model_instance = data_model()
        except ValidationError:
            # if we can't instantiate the data model
            # it means that it has required fields
            return (data_model, ...)
        else:
            return (data_model, data_model_instance)

    def get_input_fields(self, sockets: list[Socket]) -> dict[str, tuple[Any, Any]]:
        """Generate fields for the request Pydantic model based on the provided sockets.

        Parameters:
            sockets (list[Socket]): List of sockets.

        Returns:
            dict[str, tuple[Any, Field]]: Dictionary of fields for the request Pydantic model.
        """
        fields = {}
        for socket in sockets:
            field = self.socket_to_field(socket)
            fields[socket.name] = field
        return fields

    def get_output_fields(self, sockets: list[Socket]) -> dict[str, tuple[Any, Any]]:
        """Generate fields for the response Pydantic model based on the provided sockets.

        Parameters:
            sockets (list[Socket]): List of sockets.

        Returns:
            dict[str, tuple[Any, Field]]: Dictionary of fields for the response Pydantic model.
        """
        fields = {}
        for socket in sockets:
            field = self.socket_to_field(socket)
            name = self.output_to_name[socket.name]
            fields[name] = field
        return fields

    def get_file_upload_field(
        self, input_sockets: list[Socket]
    ) -> FileUploadField | None:
        """Get the file upload field for the endpoint.

        Parameters:
            input_sockets (List[Socket]): List of input sockets.

        Returns:
            Optional[FileUploadField]: File upload field or None if not found.

        Raises:
            MultipleFileUploadNotAllowed: If multiple inputs require file upload.
        """
        file_upload_field = None
        for socket in input_sockets:
            data_model = socket.data_model

            # skip sockets with no data model
            if data_model is None or data_model == Any:
                continue

            # check if pydantic model has file_upload field and it's set to True
            file_upload_enabled = data_model.model_config.get("file_upload", False)
            file_upload_description = data_model.model_config.get(
                "file_upload_description", ""
            )

            if file_upload_enabled and file_upload_field is None:
                file_upload_field = FileUploadField(
                    name=socket.name, description=file_upload_description
                )
            elif file_upload_enabled and file_upload_field is not None:
                # raise an exception if multiple inputs require file upload
                raise MultipleFileUploadNotAllowed(socket.name)
        return file_upload_field

    def get_output_filter_field(self) -> tuple[Any, Field] | None:
        """Get the output filter field for the endpoint.

        Returns:
            Optional[Tuple[Any, Field]]: Output filter field or None if not found.
        """
        if not self.output_filter:
            return None

        description = self.output_filter.description
        outputs_enum_name = self.generate_model_name("Outputs")
        outputs_enum = Enum(  # type: ignore
            outputs_enum_name,
            [(output.name, output.name) for output in self.outputs],
            type=str,
        )
        field = (Optional[list[outputs_enum]], Field(None, description=description))  # noqa: UP007
        return field

    def get_request_model(self, input_sockets: list[Socket]) -> type[BaseModel]:
        """Generate a Pydantic model for the request.

        Parameters:
            input_sockets (List[Socket]): List of input sockets.

        Returns:
            Type[BaseModel]: Pydantic model for the request.
        """
        model_name = self.generate_model_name("Request")
        input_fields = self.get_input_fields(input_sockets)
        output_filter_field = self.get_output_filter_field()
        if output_filter_field and self.output_filter:
            input_fields[self.output_filter.name] = output_filter_field
        RequestModel = create_model(model_name, **input_fields)
        return RequestModel

    def get_response_model(self, output_sockets: list[Socket]) -> type[BaseModel]:
        """Generate a Pydantic model for the response.

        Parameters:
            output_sockets (List[Socket]): List of output sockets.

        Returns:
            Type[BaseModel]: Pydantic model for the response.
        """
        model_name = self.generate_model_name("Response")
        output_fields = self.get_output_fields(output_sockets)
        ResponseModel = create_model(model_name, **output_fields)
        return ResponseModel

    def process_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Process the output of the pipeline.

        Maps the output names of the pipeline to the names defined in the endpoint outputs.

        For example, maps videos_captions_hf_blip2_opt_2_7b to captions.

        Args:
            output (dict): The output of the pipeline.

        Returns:
            dict: The processed output.
        """
        output = {
            self.output_to_name.get(output_name, output_name): output_value
            for output_name, output_value in output.items()
        }
        return output

    def create_endpoint_func(  # noqa: C901
        self,
        pipeline: Pipeline,
        RequestModel: type[BaseModel],
        file_upload_field: FileUploadField | None = None,
        event_manager: EventManager | None = None,
    ) -> Callable:
        """Create a function for routing an endpoint."""
        # Copy path to a bound variable so we don't retain an external reference
        bound_path = self.path

        async def route_func_body(body: str, files: list[UploadFile] | None = None):  # noqa: C901
            if event_manager:
                event_manager.handle(bound_path)

            # parse form data as a pydantic model and validate it
            data = RequestModel.model_validate_json(body)

            # if the input requires file upload, add the files to the data
            if file_upload_field and files:
                files_as_bytes = [await file.read() for file in files]
                getattr(data, file_upload_field.name).set_files(files_as_bytes)

            # We have to do this instead of data.model_dump() because
            # data.model_dump() will convert all nested models to dicts
            # and we want to keep them as pydantic models
            data_dict = {}
            for field_name in data.model_fields:
                field_value = getattr(data, field_name)
                data_dict[field_name] = field_value

            if self.output_filter:
                requested_outputs = data_dict.get(self.output_filter.name, None)
            else:
                requested_outputs = None

            # if user requested specific outputs, use them
            if requested_outputs:
                # get values for requested outputs because it's a list of enums
                requested_outputs = [output.value for output in requested_outputs]
                # map the requested outputs to the actual outputs
                # for example, videos_captions_hf_blip2_opt_2_7b to captions
                outputs = [self.name_to_output[output] for output in requested_outputs]
            # otherwise use the required outputs from the config (all outputs endpoint provides)
            else:
                outputs = [output.output for output in self.outputs]

            # remove the output filter parameter from the data
            if self.output_filter and self.output_filter.name in data_dict:
                del data_dict[self.output_filter.name]

            # run the pipeline
            if self.streaming:
                requested_partial_outputs = []
                requested_full_outputs = []
                for output in outputs:
                    if output in self.streaming_outputs:
                        requested_partial_outputs.append(output)
                    else:
                        requested_full_outputs.append(output)

                async def generator_wrapper() -> AsyncGenerator[bytes, None]:
                    """Serializes the output of the generator using ORJSONResponseCustom."""
                    try:
                        async for output in run_pipeline_streaming(
                            pipeline,
                            data_dict,
                            requested_partial_outputs,
                            requested_full_outputs,
                        ):
                            output = self.process_output(output)
                            yield AanaJSONResponse(content=output).body
                    except Exception as e:
                        yield custom_exception_handler(None, e).body

                return StreamingResponse(
                    generator_wrapper(), media_type="application/json"
                )
            else:
                output = await run_pipeline(pipeline, data_dict, outputs)
                output = self.process_output(output)
                return AanaJSONResponse(content=output)

        if file_upload_field:
            files = File(None, description=file_upload_field.description)
        else:
            files = None

        async def route_func(body: str = Form(...), files=files):
            return await route_func_body(body=body, files=files)

        return route_func

    def register(
        self,
        app: FastAPI,
        pipeline: Pipeline,
        custom_schemas: dict[str, dict],
        event_manager: EventManager,
    ):
        """Register an endpoint to the FastAPI app and add schemas to the custom schemas dictionary.

        Parameters:
            app (FastAPI): FastAPI app to register the endpoint to.
            pipeline (Pipeline): Pipeline to register the endpoint to.
            custom_schemas (Dict[str, Dict]): Dictionary of custom schemas.
            event_manager (EventManager): The event manager for the associated app
        """
        input_sockets, output_sockets = pipeline.get_sockets(
            [output.output for output in self.outputs]
        )
        RequestModel = self.get_request_model(input_sockets)
        ResponseModel = self.get_response_model(output_sockets)
        file_upload_field = self.get_file_upload_field(input_sockets)
        if self.event_handlers:
            for handler in self.event_handlers:
                with contextlib.suppress(HandlerAlreadyRegisteredException):
                    event_manager.register_handler(handler)
        route_func = self.create_endpoint_func(
            pipeline=pipeline,
            RequestModel=RequestModel,
            file_upload_field=file_upload_field,
            event_manager=event_manager,
        )
        app.post(
            self.path,
            summary=self.summary,
            name=self.name,
            operation_id=self.name,
            response_model=ResponseModel,
            responses={
                400: {"model": ExceptionResponseModel},
            },
        )(route_func)
        custom_schemas[self.name] = RequestModel.schema()


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
    if "definitions" not in openapi_schema:
        openapi_schema["definitions"] = {}
    for schema_name, schema in custom_schemas.items():
        # if we have a definitions then we need to move them out to the top level of the schema
        if "definitions" in schema:
            openapi_schema["definitions"].update(schema["definitions"])
            del schema["definitions"]
        openapi_schema["components"]["schemas"][f"Body_{schema_name}"]["properties"][
            "body"
        ] = schema
    return openapi_schema
