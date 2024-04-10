from enum import Enum
from inspect import isasyncgenfunction
from typing import Any, AsyncGenerator, Optional

from fastapi.responses import StreamingResponse
from pydantic import Field, ValidationError, create_model
from pydantic.main import BaseModel

from aana.api.app import custom_exception_handler


def generate_model_name(func, suffix: str) -> str:
    """Generate a Pydantic model name based on a given suffix.

    Parameters:
        suffix (str): Suffix for the model name (e.g. "Request", "Response").

    Returns:
        str: Generated model name.
    """
    return "".join([word.capitalize() for word in func.__name__.split("_")]) + suffix


def type_to_field(arg_type) -> tuple[Any, Any]:
    """Convert a socket to a Pydantic field.

    Parameters:
        socket (Socket): Socket to convert.

    Returns:
        Tuple[Any, Field]: Tuple of the socket's data model and a Pydantic field.
    """
    data_model = arg_type

    # if data model is None or Any, set it to Any
    if data_model is None or data_model == Any:
        return (Any, Field(None))

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


def get_input_fields(func) -> dict[str, tuple[Any, Any]]:
    """Generate fields for the request Pydantic model based on the provided sockets.

    Parameters:
        func (Callable): Function to generate the request model for.

    Returns:
        dict[str, tuple[Any, Field]]: Dictionary of fields for the request Pydantic model.
    """
    fields = {}
    # for socket in sockets:
    #     field = func.socket_to_field(socket)
    #     fields[socket.name] = field
    for arg_name, arg_type in func.__annotations__.items():
        if arg_name == "return":
            continue
        # fields[arg_name] = (arg_type, Field(None))
        fields[arg_name] = type_to_field(arg_type)
    return fields


def get_request_model(func) -> type[BaseModel]:
    """Generate the request Pydantic model for the provided function.

    Parameters:
        func (Callable): Function to generate the request model for.

    Returns:
        type[BaseModel]: Request Pydantic model.
    """
    model_name = generate_model_name(func, "Request")
    input_fields = get_input_fields(func)
    return create_model(model_name, **input_fields)


def get_output_fields(func):
    if isasyncgenfunction(func):
        return_type = func.__annotations__["return"].__args__[0]
    else:
        return_type = func.__annotations__["return"]
    fields = {}
    for arg_name, arg_type in return_type.__annotations__.items():
        fields[arg_name] = type_to_field(arg_type)
    return fields


def get_response_model(func) -> type[BaseModel]:
    model_name = generate_model_name(func, "Response")
    output_fields = get_output_fields(func)
    return create_model(model_name, **output_fields)


from typing import Callable

from fastapi import File, Form, UploadFile

from aana.api.api_generation import FileUploadField
from aana.api.responses import AanaJSONResponse


def create_endpoint_func(
    func,
    RequestModel: type[BaseModel],
    file_upload_field: FileUploadField | None = None,
) -> Callable:
    """Create a function for routing an endpoint."""

    async def route_func_body(body: str, files: list[UploadFile] | None = None):  # noqa: C901
        # parse form data as a pydantic model and validate it
        data = RequestModel.model_validate_json(body)

        # if the input requires file upload, add the files to the data
        if file_upload_field and files:
            files_as_bytes = [await file.read() for file in files]
            getattr(data, file_upload_field.name).set_files(files_as_bytes)

        # We have to do this instead of data.dict() because
        # data.dict() will convert all nested models to dicts
        # and we want to keep them as pydantic models
        data_dict = {}
        for field_name in data.model_fields:
            field_value = getattr(data, field_name)
            data_dict[field_name] = field_value

        # # run the pipeline
        # if self.streaming:
        #     requested_partial_outputs = []
        #     requested_full_outputs = []
        #     for output in outputs:
        #         if output in self.streaming_outputs:
        #             requested_partial_outputs.append(output)
        #         else:
        #             requested_full_outputs.append(output)

        #     async def generator_wrapper() -> AsyncGenerator[bytes, None]:
        #         """Serializes the output of the generator using ORJSONResponseCustom."""
        #         try:
        #             async for output in run_pipeline_streaming(
        #                 pipeline,
        #                 data_dict,
        #                 requested_partial_outputs,
        #                 requested_full_outputs,
        #             ):
        #                 output = self.process_output(output)
        #                 yield AanaJSONResponse(content=output).body
        #         except Exception as e:
        #             yield custom_exception_handler(None, e).body

        #     return StreamingResponse(
        #         generator_wrapper(), media_type="application/json"
        #     )
        # else:
        # output = await run_pipeline(pipeline, data_dict, outputs)
        # output = self.process_output(output)
        # return AanaJSONResponse(content=output)

        if isasyncgenfunction(func):

            async def generator_wrapper() -> AsyncGenerator[bytes, None]:
                """Serializes the output of the generator using ORJSONResponseCustom."""
                try:
                    async for output in func(**data_dict):
                        yield AanaJSONResponse(content=output).body
                except Exception as e:
                    yield custom_exception_handler(None, e).body

            return StreamingResponse(generator_wrapper(), media_type="application/json")
        else:
            output = await func(**data_dict)
            return AanaJSONResponse(content=output)

    if file_upload_field:
        files = File(None, description=file_upload_field.description)
    else:
        files = None

    async def route_func(body: str = Form(...), files=files):
        return await route_func_body(body=body, files=files)

    return route_func


from dataclasses import dataclass
from typing import Callable

from aana.api.new_api_generation import (
    create_endpoint_func,
    get_request_model,
    get_response_model,
)
from aana.models.pydantic.exception_response import ExceptionResponseModel


@dataclass
class Endpoint:
    """Class used to represent an endpoint.

    Attributes:
        name (str): Name of the endpoint.
        path (str): Path of the endpoint.
        summary (str): Description of the endpoint that will be shown in the API documentation.
    """

    name: str
    path: str
    summary: str
    func: Callable
    # endpoint_class: Callable

    # def __post_init__(self):
    #     self.func = self.endpoint_class.__call__

    def register(self, app, custom_schemas):
        """Register the endpoint in the FastAPI application.

        Args:
            app (FastAPI): FastAPI application.
            custom_schemas (dict): Dictionary containing custom schemas.
        """
        RequestModel = get_request_model(self.func)
        ResponseModel = get_response_model(self.func)

        route_func = create_endpoint_func(self.func, RequestModel)

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
