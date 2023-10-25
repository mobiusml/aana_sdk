from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Type, Any, List, Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.openapi.utils import get_openapi
from mobius_pipeline.pipeline.pipeline import Pipeline
from mobius_pipeline.node.socket import Socket
from pydantic import Field, create_model, BaseModel, parse_raw_as

# from aana.api.app import app

from aana.exceptions.general import MultipleFileUploadNotAllowed
from aana.models.pydantic.exception_response import ExceptionResponseModel


@dataclass
class OutputFilter:
    name: str
    description: str


@dataclass
class Endpoint:
    name: str
    path: str
    summary: str
    outputs: List[str]
    output_filter: Optional[OutputFilter] = None
    streaming: bool = False


@dataclass
class FileUploadField:
    name: str
    description: str


def generate_model_name(endpoint_name: str, suffix: str) -> str:
    """
    Generate a Pydantic model name based on the endpoint name and a given suffix.

    Parameters:
        endpoint_name (str): Name of the endpoint.
        suffix (str): Suffix for the model name (e.g. "Request", "Response").

    Returns:
        str: Generated model name.
    """
    return "".join([word.capitalize() for word in endpoint_name.split("_")]) + suffix


def socket_to_field(socket: Socket) -> Tuple[Any, Field]:
    """
    Convert a socket to a Pydantic field.

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

    # check if any of the fields are required
    if any(field.required for field in data_model.__fields__.values()):
        return (data_model, ...)

    return (data_model, data_model())


def get_fields(sockets: List[Socket]) -> Dict[str, Tuple[Any, Field]]:
    """
    Generate fields for the Pydantic model based on the provided sockets.

    Parameters:
        sockets (List[Socket]): List of sockets.

    Returns:
        Dict[str, Tuple[Any, Field]]: Dictionary of fields for the Pydantic model.
    """
    fields = {}
    for socket in sockets:
        field = socket_to_field(socket)
        fields[socket.name] = field
    return fields


def get_file_upload_field(
    endpoint: Endpoint, pipeline: Pipeline
) -> Optional[FileUploadField]:
    """
    Get the file upload field from the provided sockets.

    Parameters:
        endpoint (Endpoint): Endpoint to get the file upload field from.
        pipeline (Pipeline): Pipeline to get the sockets from.

    Returns:
        Optional[FileUploadField]: File upload field or None if not found.

    Raises:
        MultipleFileUploadNotAllowed: If multiple inputs require file upload.
    """
    input_sockets, _ = pipeline.get_sockets(endpoint.outputs)

    file_upload_field = None
    for socket in input_sockets:
        data_model = socket.data_model

        # skip sockets with no data model
        if data_model is None or data_model == Any:
            continue

        # check if pydantic model has file_upload field and it's set to True
        file_upload_enabled = getattr(data_model.Config, "file_upload", False)
        file_upload_description = getattr(
            data_model.Config, "file_upload_description", ""
        )

        if file_upload_enabled and file_upload_field is None:
            file_upload_field = FileUploadField(
                name=socket.name, description=file_upload_description
            )
        elif file_upload_enabled and file_upload_field is not None:
            # raise an exception if multiple inputs require file upload
            raise MultipleFileUploadNotAllowed(socket.name)
    return file_upload_field


def get_output_filter_field(endpoint: Endpoint) -> Optional[Tuple[Any, Field]]:
    """
    Get the output filter field from the provided endpoint.

    Parameters:
        endpoint (Endpoint): Endpoint to get the output filter field from.

    Returns:
        Optional[Tuple[Any, Field]]: Output filter field or None if not found.
    """
    if not endpoint.output_filter:
        return None

    name = endpoint.output_filter.name
    description = endpoint.output_filter.description
    outputs_enum_name = generate_model_name(endpoint.name, "Outputs")
    outputs_enum = Enum(
        outputs_enum_name, [(output, output) for output in endpoint.outputs], type=str
    )
    field = (Optional[List[outputs_enum]], Field(None, description=description))
    return field


def get_request_model(endpoint: Endpoint, pipeline: Pipeline) -> Type[BaseModel]:
    """
    Generate a Pydantic model for the request based on the provided endpoint.

    Parameters:
        endpoint (Endpoint): Endpoint to generate the model for.
        pipeline (Pipeline): Pipeline to get the sockets from.

    Returns:
        Type[BaseModel]: Pydantic model for the request.
    """
    model_name = generate_model_name(endpoint.name, "Request")
    input_sockets, _ = pipeline.get_sockets(endpoint.outputs)
    input_fields = get_fields(input_sockets)
    output_filter_field = get_output_filter_field(endpoint)
    if output_filter_field:
        input_fields[endpoint.output_filter.name] = output_filter_field
    RequestModel = create_model(model_name, **input_fields)
    return RequestModel


def get_response_model(endpoint: Endpoint, pipeline: Pipeline) -> Type[BaseModel]:
    """
    Generate a Pydantic model for the response based on the provided endpoint.

    Parameters:
        endpoint (Endpoint): Endpoint to generate the model for.
        pipeline (Pipeline): Pipeline to get the sockets from.

    Returns:
        Type[BaseModel]: Pydantic model for the response.
    """
    model_name = generate_model_name(endpoint.name, "Response")
    _, output_sockets = pipeline.get_sockets(endpoint.outputs)
    output_fields = get_fields(output_sockets)
    ResponseModel = create_model(model_name, **output_fields)
    return ResponseModel


async def run_pipeline(
    pipeline: Pipeline, data: Dict, required_outputs: List[str]
) -> Dict[str, float]:
    """
    This function is used to run a Mobius Pipeline.
    It creates a container from the data, runs the pipeline and returns the output.

    Args:
        pipeline (Pipeline): The pipeline to run.
        data (dict): The data to create the container from.
        required_outputs (List[str]): The required outputs of the pipeline.

    Returns:
        dict[str, float]: The output of the pipeline and the execution time of the pipeline.
    """

    # create a container from the data
    container = pipeline.parse_dict(data)

    # run the pipeline
    output, execution_time = await pipeline.run(
        container, required_outputs, return_execution_time=True
    )
    output["execution_time"] = execution_time
    return output


def create_endpoint_func(
    pipeline: Pipeline,
    endpoint: Endpoint,
    RequestModel: Type[BaseModel],
    ResponseModel: Type[BaseModel],
    file_upload_field: FileUploadField = None,
):
    async def route_func_body(body: str, files: List[UploadFile] = None):
        # parse form data as a pydantic model and validate it
        data = parse_raw_as(RequestModel, body)

        # if the input requires file upload, add the files to the data
        if file_upload_field and files:
            files = [await file.read() for file in files]
            getattr(data, file_upload_field).set_files(files)

        # We have to do this instead of data.dict() because
        # data.dict() will convert all nested models to dicts
        # and we want to keep them as pydantic models
        data_dict = {}
        for field_name in data.__fields__:
            field_value = getattr(data, field_name)
            # check if it has a method convert_to_entities
            # if it does, call it to convert the model to an entity
            if hasattr(field_value, "convert_to_entity"):
                field_value = field_value.convert_to_entity()
            data_dict[field_name] = field_value

        if endpoint.output_filter:
            requested_outputs = data_dict.get(endpoint.output_filter.name, None)
        else:
            requested_outputs = None

        # if user requested specific outputs, use them
        if requested_outputs:
            # get values for requested outputs because it's a list of enums
            requested_outputs = [output.value for output in requested_outputs]
            outputs = requested_outputs
        # otherwise use the required outputs from the config (all outputs endpoint provides)
        else:
            outputs = endpoint.outputs

        # remove the output filter parameter from the data
        if endpoint.output_filter and endpoint.output_filter.name in data_dict:
            del data_dict[endpoint.output_filter.name]

        # run the pipeline
        return await run_pipeline(pipeline, data_dict, outputs)

    if file_upload_field:

        async def route_func_files(
            body: str = Form(...),
            files: List[UploadFile] = File(
                None, description=file_upload_field.description
            ),
        ):
            return await route_func_body(body=body, files=files)

        return route_func_files
    else:

        async def route_func(body: str = Form(...)):
            return await route_func_body(body=body)

        return route_func


def add_custom_schemas_to_openapi_schema(
    openapi_schema: Dict[str, Any], custom_schemas: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add custom schemas to the openapi schema.

    File upload is that FastAPI doesn't support Pydantic models in multipart requests.
    There is a discussion about it on FastAPI discussion forum.
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

    for schema_name, schema in custom_schemas.items():
        # if we have a definitions then we need to move them out to the top level of the schema
        if "definitions" in schema:
            if "definitions" not in openapi_schema:
                openapi_schema["definitions"] = {}
            openapi_schema["definitions"].update(schema["definitions"])
            del schema["definitions"]
        openapi_schema["components"]["schemas"][f"Body_{schema_name}"]["properties"][
            "body"
        ] = schema
    return openapi_schema


def register_endpoint(app: FastAPI, pipeline: Pipeline, endpoint: Endpoint):
    """
    Register an endpoint to the FastAPI app.

    Parameters:
        pipeline (Pipeline): Pipeline to register the endpoint to.
        endpoint (Endpoint): Endpoint to register.
    """
    RequestModel = get_request_model(endpoint, pipeline)
    ResponseModel = get_response_model(endpoint, pipeline)
    file_upload_field = get_file_upload_field(endpoint, pipeline)
    route_func = create_endpoint_func(
        pipeline, endpoint, RequestModel, ResponseModel, file_upload_field
    )
    app.post(
        endpoint.path,
        summary=endpoint.summary,
        name=endpoint.name,
        operation_id=endpoint.name,
        response_model=ResponseModel,
        responses={
            400: {"model": ExceptionResponseModel},
        },
    )(route_func)


def get_request_schema(pipeline: Pipeline, endpoint: Endpoint):
    RequestModel = get_request_model(endpoint, pipeline)
    return RequestModel.schema()
