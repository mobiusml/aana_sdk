from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Type, Any, List, Optional
from fastapi import FastAPI, File, Form, UploadFile
from mobius_pipeline.pipeline.pipeline import Pipeline
from mobius_pipeline.node.socket import Socket
from pydantic import Field, create_model, BaseModel, parse_raw_as
from aana.api.responses import AanaJSONResponse

from aana.exceptions.general import MultipleFileUploadNotAllowed
from aana.models.pydantic.exception_response import ExceptionResponseModel


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


@dataclass
class OutputFilter:
    """
    Class used to represent an output filter.

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
    """
    Class used to represent a file upload field.

    Attributes:
        name (str): Name of the field.
        description (str): Description of the field.
    """

    name: str
    description: str


@dataclass
class Endpoint:
    """
    Class used to represent an endpoint.

    Attributes:
        name (str): Name of the endpoint.
        path (str): Path of the endpoint.
        summary (str): Description of the endpoint that will be shown in the API documentation.
        outputs (List[str]): List of required outputs from the pipeline that should be returned
                                by the endpoint.
        output_filter (Optional[OutputFilter]): The parameter will be added to the request and
                                will allow to choose subset of `outputs` to return.
        streaming (bool): Whether the endpoint outputs a stream of data.
    """

    name: str
    path: str
    summary: str
    outputs: List[str]
    output_filter: Optional[OutputFilter] = None
    streaming: bool = False

    def generate_model_name(self, suffix: str) -> str:
        """
        Generate a Pydantic model name based oon a given suffix.

        Parameters:
            suffix (str): Suffix for the model name (e.g. "Request", "Response").

        Returns:
            str: Generated model name.
        """
        return "".join([word.capitalize() for word in self.name.split("_")]) + suffix

    def socket_to_field(self, socket: Socket) -> Tuple[Any, Any]:
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

    def get_fields(self, sockets: List[Socket]) -> Dict[str, Tuple[Any, Any]]:
        """
        Generate fields for the Pydantic model based on the provided sockets.

        Parameters:
            sockets (List[Socket]): List of sockets.

        Returns:
            Dict[str, Tuple[Any, Field]]: Dictionary of fields for the Pydantic model.
        """
        fields = {}
        for socket in sockets:
            field = self.socket_to_field(socket)
            fields[socket.name] = field
        return fields

    def get_file_upload_field(
        self, input_sockets: List[Socket]
    ) -> Optional[FileUploadField]:
        """
        Get the file upload field for the endpoint.

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

    def get_output_filter_field(self) -> Optional[Tuple[Any, Any]]:
        """
        Get the output filter field for the endpoint.

        Returns:
            Optional[Tuple[Any, Field]]: Output filter field or None if not found.
        """
        if not self.output_filter:
            return None

        description = self.output_filter.description
        outputs_enum_name = self.generate_model_name("Outputs")
        outputs_enum = Enum(  # type: ignore
            outputs_enum_name, [(output, output) for output in self.outputs], type=str
        )
        field = (Optional[List[outputs_enum]], Field(None, description=description))
        return field

    def get_request_model(self, input_sockets: List[Socket]) -> Type[BaseModel]:
        """
        Generate a Pydantic model for the request.

        Parameters:
            input_sockets (List[Socket]): List of input sockets.

        Returns:
            Type[BaseModel]: Pydantic model for the request.
        """
        model_name = self.generate_model_name("Request")
        input_fields = self.get_fields(input_sockets)
        output_filter_field = self.get_output_filter_field()
        if output_filter_field and self.output_filter:
            input_fields[self.output_filter.name] = output_filter_field
        RequestModel = create_model(model_name, **input_fields)
        return RequestModel

    def get_response_model(self, output_sockets: List[Socket]) -> Type[BaseModel]:
        """
        Generate a Pydantic model for the response.

        Parameters:
            output_sockets (List[Socket]): List of output sockets.

        Returns:
            Type[BaseModel]: Pydantic model for the response.
        """
        model_name = self.generate_model_name("Response")
        output_fields = self.get_fields(output_sockets)
        ResponseModel = create_model(model_name, **output_fields)
        return ResponseModel

    def create_endpoint_func(
        self,
        pipeline: Pipeline,
        RequestModel: Type[BaseModel],
        ResponseModel: Type[BaseModel],
        file_upload_field: Optional[FileUploadField] = None,
    ):
        async def route_func_body(body: str, files: Optional[List[UploadFile]] = None):
            # parse form data as a pydantic model and validate it
            data = parse_raw_as(RequestModel, body)

            # if the input requires file upload, add the files to the data
            if file_upload_field and files:
                files_as_bytes = [await file.read() for file in files]
                getattr(data, file_upload_field.name).set_files(files_as_bytes)

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

            if self.output_filter:
                requested_outputs = data_dict.get(self.output_filter.name, None)
            else:
                requested_outputs = None

            # if user requested specific outputs, use them
            if requested_outputs:
                # get values for requested outputs because it's a list of enums
                requested_outputs = [output.value for output in requested_outputs]
                outputs = requested_outputs
            # otherwise use the required outputs from the config (all outputs endpoint provides)
            else:
                outputs = self.outputs

            # remove the output filter parameter from the data
            if self.output_filter and self.output_filter.name in data_dict:
                del data_dict[self.output_filter.name]

            # run the pipeline
            output = await run_pipeline(pipeline, data_dict, outputs)
            return AanaJSONResponse(content=output)

        if file_upload_field:
            files = File(None, description=file_upload_field.description)
        else:
            files = None

        async def route_func(body: str = Form(...), files=files):
            return await route_func_body(body=body, files=files)

        return route_func

    def register(self, app: FastAPI, pipeline: Pipeline):
        """
        Register an endpoint to the FastAPI app.

        Parameters:
            app (FastAPI): FastAPI app to register the endpoint to.
            pipeline (Pipeline): Pipeline to register the endpoint to.
        """
        input_sockets, output_sockets = pipeline.get_sockets(self.outputs)
        RequestModel = self.get_request_model(input_sockets)
        ResponseModel = self.get_response_model(output_sockets)
        file_upload_field = self.get_file_upload_field(input_sockets)
        route_func = self.create_endpoint_func(
            pipeline=pipeline,
            RequestModel=RequestModel,
            ResponseModel=ResponseModel,
            file_upload_field=file_upload_field,
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

    def get_request_schema(self, pipeline: Pipeline):
        input_sockets, _ = pipeline.get_sockets(self.outputs)
        RequestModel = self.get_request_model(input_sockets)
        return RequestModel.schema()


def add_custom_schemas_to_openapi_schema(
    openapi_schema: Dict[str, Any], custom_schemas: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add custom schemas to the openapi schema.

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