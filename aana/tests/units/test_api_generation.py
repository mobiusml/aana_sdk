# ruff: noqa: S101
from collections.abc import AsyncGenerator
from typing import Annotated, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field

from aana.api.api_generation import Endpoint


class InputModel(BaseModel):
    """Model for a text input."""

    input: str = Field(..., description="Input text")
    model_config = ConfigDict(extra="forbid")


class TestEndpointOutput(TypedDict):
    """The output of the test endpoint."""

    output: Annotated[str, Field(..., description="Output text")]


class TestEndpoint(Endpoint):
    """Test endpoint for get_request_model."""

    async def run(self, input_data: InputModel) -> TestEndpointOutput:
        """Run the endpoint."""
        return {"output": input_data.input}


class TestEndpointMissingReturn(Endpoint):
    """Test endpoint for get_response_model with missing return type."""

    async def run(self, input_data: InputModel):
        """Run the endpoint."""
        return {"output": input_data.input}


class TestEndpointMissingReturnStreaming(Endpoint):
    """Test endpoint for get_response_model with missing return type in streaming endpoint."""

    async def run(self, input_data: InputModel) -> AsyncGenerator:
        """Run the endpoint."""
        yield {"output": input_data.input}


def test_get_request_model():
    """Test the get_request_model function."""
    endpoint = TestEndpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
    )

    RequestModel = endpoint.get_request_model()

    # Check that the request model named correctly
    assert RequestModel.__name__ == "TestEndpointRequest"

    # Check that the request model has the correct fields
    assert RequestModel.model_fields.keys() == {"input_data"}

    # Check that the request fields have the correct types
    assert RequestModel.model_fields["input_data"].annotation == InputModel


def test_get_response_model():
    """Test the get_response_model function."""
    endpoint = TestEndpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
    )
    ResponseModel = endpoint.get_response_model()

    # Check that the response model named correctly
    assert ResponseModel.__name__ == "TestEndpointResponse"

    # Check that the response model has the correct fields
    assert ResponseModel.model_fields.keys() == {"output"}

    # Check that the response fields have the correct types
    assert ResponseModel.model_fields["output"].annotation == str


def test_get_response_model_missing_return():
    """Test the get_response_model function with missing return type."""
    endpoint = TestEndpointMissingReturn(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
    )

    with pytest.raises(ValueError):
        endpoint.get_response_model()


def test_get_response_model_missing_return_streaming():
    """Test the get_response_model function with missing return type in streaming endpoint."""
    endpoint = TestEndpointMissingReturnStreaming(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
    )

    with pytest.raises(ValueError):
        endpoint.get_response_model()
