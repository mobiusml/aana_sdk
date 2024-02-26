# ruff: noqa: S101, A003
from typing import Any

import pytest
from mobius_pipeline.node.socket import Socket
from pydantic import BaseModel, ConfigDict, Field

from aana.api.api_generation import Endpoint, EndpointOutput
from aana.exceptions.general import MultipleFileUploadNotAllowed


class InputModel(BaseModel):
    """Model for a text input."""

    input: str = Field(..., description="Input text")
    model_config = ConfigDict(extra="forbid")


class FileUploadModel(BaseModel):
    """Model for a file upload input."""

    content: bytes | None = Field(
        None,
        description="The content in bytes. Set this field to 'file' to upload files to the endpoint.",
    )

    def set_files(self, files):
        """Set files."""
        if files:
            if isinstance(files, list):
                files = files[0]
            self.content = files
    model_config = ConfigDict(extra="forbid", file_upload=True, file_upload_description="Upload image files.")


class OutputModel(BaseModel):
    """Model for outputs."""

    output: str = Field(..., description="Output text")
    model_config = ConfigDict(extra="forbid")


def test_get_request_model():
    """Test the get_request_model function."""
    endpoint = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=[EndpointOutput(name="output", output="output")],
    )

    input_sockets = [
        Socket(name="input", path="input", key="input", data_model=InputModel),
        Socket(
            name="input_without_datamodel",
            path="input_without_datamodel",
            key="input_without_datamodel",
        ),
    ]

    RequestModel = endpoint.get_request_model(input_sockets)

    # Check that the request model named correctly
    assert RequestModel.__name__ == "TestEndpointRequest"

    # Check that the request model has the correct fields
    assert RequestModel.model_fields.keys() == {"input", "input_without_datamodel"}

    # Check that the request fields have the correct types
    assert RequestModel.model_fields["input"].annotation == InputModel
    assert RequestModel.model_fields["input_without_datamodel"].annotation == Any


def test_get_response_model():
    """Test the get_response_model function."""
    endpoint = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=[
            EndpointOutput(name="output", output="output"),
            EndpointOutput(
                name="output_without_datamodel", output="output_without_datamodel"
            ),
        ],
    )

    output_sockets = [
        Socket(name="output", path="output", key="output", data_model=OutputModel),
        Socket(
            name="output_without_datamodel",
            path="output_without_datamodel",
            key="output_without_datamodel",
        ),
    ]

    ResponseModel = endpoint.get_response_model(output_sockets)

    # Check that the response model named correctly
    assert ResponseModel.__name__ == "TestEndpointResponse"

    # Check that the response model has the correct fields
    assert ResponseModel.model_fields.keys() == {"output", "output_without_datamodel"}

    # Check that the response fields have the correct types
    assert ResponseModel.model_fields["output"].annotation == OutputModel
    assert ResponseModel.model_fields["output_without_datamodel"].annotation == Any

    endpoint_with_one_output = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=[EndpointOutput(name="output", output="output")],
    )

    output_sockets = [
        Socket(name="output", path="output", key="output", data_model=OutputModel),
    ]

    ResponseModel = endpoint_with_one_output.get_response_model(output_sockets)

    # Check that the response model named correctly
    assert ResponseModel.__name__ == "TestEndpointResponse"

    # Check that the response model has the correct fields
    assert ResponseModel.model_fields.keys() == {"output"}

    # Check that the response fields have the correct types
    assert ResponseModel.model_fields["output"].annotation == OutputModel


def test_get_file_upload_field():
    """Test the get_file_upload_field function."""
    endpoint = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=[EndpointOutput(name="output", output="output")],
    )

    input_sockets = [
        Socket(
            name="input",
            path="input",
            key="input",
            data_model=FileUploadModel,
        ),
    ]

    file_upload_field = endpoint.get_file_upload_field(input_sockets)

    # Check that the file upload field named correctly
    assert file_upload_field.name == "input"

    # Check that the file upload field has the correct description
    assert file_upload_field.description == "Upload image files."


def test_get_file_upload_field_multiple_file_uploads():
    """Test the get_file_upload_field function with multiple file uploads."""
    endpoint = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=[EndpointOutput(name="output", output="output")],
    )

    input_sockets = [
        Socket(
            name="input",
            path="input",
            key="input",
            data_model=FileUploadModel,
        ),
        Socket(
            name="input2",
            path="input2",
            key="input2",
            data_model=FileUploadModel,
        ),
    ]

    with pytest.raises(MultipleFileUploadNotAllowed):
        endpoint.get_file_upload_field(input_sockets)
