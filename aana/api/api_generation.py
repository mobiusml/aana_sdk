import inspect
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from inspect import isasyncgenfunction
from typing import Annotated, Any, get_origin

from fastapi import File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import Field, ValidationError, create_model
from pydantic.main import BaseModel

from aana.api.app import custom_exception_handler
from aana.api.responses import AanaJSONResponse
from aana.exceptions.general import MultipleFileUploadNotAllowed
from aana.models.pydantic.exception_response import ExceptionResponseModel


def get_default_values(func):
    """Get the default values for the function arguments."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


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
    initialized: bool = False

    async def initialize(self):
        """Initialize the endpoint."""
        pass

    async def run(self, *args, **kwargs):
        """Run the endpoint."""
        raise NotImplementedError

    def register(self, app, custom_schemas):
        """Register the endpoint in the FastAPI application.

        Args:
            app (FastAPI): FastAPI application.
            custom_schemas (dict): Dictionary containing custom schemas.
        """
        RequestModel = self.get_request_model()
        ResponseModel = self.get_response_model()

        route_func = self.create_endpoint_func(RequestModel)

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

    def generate_model_name(self, suffix: str) -> str:
        """Generate a Pydantic model name based on a given suffix.

        Args:
            suffix (str): Suffix for the model name (e.g. "Request", "Response").

        Returns:
            str: Generated model name.
        """
        return self.__class__.__name__ + suffix

    def arg_to_field(self, arg_name: str, arg_type: Any) -> tuple[Any, Any]:
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
        except ValidationError:
            # if we can't instantiate the data model
            # it means that it has required fields
            return (arg_type, ...)
        else:
            return (arg_type, data_model_instance)

    def get_input_fields(self) -> dict[str, tuple[Any, Any]]:
        """Generate fields for the request Pydantic model based on the function annotations.

        Returns:
            dict[str, tuple[Any, Field]]: Dictionary of fields for the request Pydantic model.
        """
        fields = {}
        for arg_name, arg_type in self.run.__annotations__.items():
            if arg_name == "return":
                continue
            fields[arg_name] = self.arg_to_field(arg_name, arg_type)
        return fields

    def get_request_model(self) -> type[BaseModel]:
        """Generate the request Pydantic model for the endpoint.

        Returns:
            type[BaseModel]: Request Pydantic model.
        """
        model_name = self.generate_model_name("Request")
        input_fields = self.get_input_fields()
        return create_model(model_name, **input_fields)

    def get_output_fields(self) -> dict[str, tuple[Any, Any]]:
        """Generate fields for the response Pydantic model based on the function annotations.

        Returns:
            dict[str, tuple[Any, Any]]: Dictionary of fields for the response Pydantic model.
        """
        if self.is_streaming_response():
            return_type = self.run.__annotations__["return"].__args__[0]
        else:
            return_type = self.run.__annotations__["return"]
        fields = {}
        for arg_name, arg_type in return_type.__annotations__.items():
            fields[arg_name] = self.arg_to_field(arg_name, arg_type)
        return fields

    def get_response_model(self) -> type[BaseModel]:
        """Generate the response Pydantic model for the endpoint.

        Returns:
            type[BaseModel]: Response Pydantic model.
        """
        model_name = self.generate_model_name("Response")
        output_fields = self.get_output_fields()
        return create_model(model_name, **output_fields)

    def get_file_upload_field(self) -> FileUploadField | None:
        """Get the file upload field for the endpoint.

        Returns:
            Optional[FileUploadField]: File upload field or None if not found.

        Raises:
            MultipleFileUploadNotAllowed: If multiple inputs require file upload.
        """
        file_upload_field = None
        for arg_name, arg_type in self.run.__annotations__.items():
            if arg_name == "return":
                continue

            # check if pydantic model has file_upload field and it's set to True
            if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
                file_upload_enabled = arg_type.model_config.get("file_upload", False)
                file_upload_description = arg_type.model_config.get(
                    "file_upload_description", ""
                )
            else:
                file_upload_enabled = False
                file_upload_description = ""

            if file_upload_enabled and file_upload_field is None:
                file_upload_field = FileUploadField(
                    name=arg_name, description=file_upload_description
                )
            elif file_upload_enabled and file_upload_field is not None:
                # raise an exception if multiple inputs require file upload
                raise MultipleFileUploadNotAllowed(arg_name)
        return file_upload_field

    @classmethod
    def is_streaming_response(cls) -> bool:
        """Check if the endpoint returns a streaming response.

        Returns:
            bool: True if the endpoint returns a streaming response, False otherwise.
        """
        return isasyncgenfunction(cls.run)

    def create_endpoint_func(  # noqa: C901
        self,
        RequestModel: type[BaseModel],
        file_upload_field: FileUploadField | None = None,
    ) -> Callable:
        """Create a function for routing an endpoint."""

        async def route_func_body(body: str, files: list[UploadFile] | None = None):
            if not self.initialized:
                await self.initialize()

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

        if file_upload_field:
            files = File(None, description=file_upload_field.description)
        else:
            files = None

        async def route_func(body: str = Form(...), files=files):
            return await route_func_body(body=body, files=files)

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
    for schema_name, schema in custom_schemas.items():
        # if we have a definitions then we need to move them out to the top level of the schema
        if "$defs" in schema:
            openapi_schema["$defs"].update(schema["$defs"])
            del schema["$defs"]
        openapi_schema["components"]["schemas"][f"Body_{schema_name}"]["properties"][
            "body"
        ] = schema
    return openapi_schema
