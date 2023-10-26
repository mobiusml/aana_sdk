from typing import Any
from unittest.mock import Mock
from mobius_pipeline.node.socket import Socket
from mobius_pipeline.pipeline.pipeline import Pipeline
import pytest

from pydantic import BaseModel, Field, Extra

from aana.api.api_generation import Endpoint


class InputModel(BaseModel):
    input: str = Field(..., description="Input text")

    class Config:
        extra = Extra.forbid


class OutputModel(BaseModel):
    output: str = Field(..., description="Output text")

    class Config:
        extra = Extra.forbid


@pytest.fixture
def mock_pipeline():
    mock = Mock(spec=Pipeline)

    def mock_get_sockets(outputs):
        input_sockets = [
            Socket(name="input", path="input", key="input", data_model=InputModel),
            Socket(
                name="input_without_datamodel",
                path="input_without_datamodel",
                key="input_without_datamodel",
            ),
        ]
        possible_outputs = {
            "output": Socket(
                name="output", path="output", key="output", data_model=OutputModel
            ),
            "output_without_datamodel": Socket(
                name="output_without_datamodel",
                path="output_without_datamodel",
                key="output_without_datamodel",
            ),
        }
        output_sockets = [possible_outputs[output] for output in outputs]
        return input_sockets, output_sockets

    mock.get_sockets.side_effect = mock_get_sockets
    return mock


def test_get_request_model(mock_pipeline):
    """Test the get_request_model function."""

    endpoint = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=["output"],
    )

    RequestModel = endpoint.get_request_model(mock_pipeline)

    # Check that the request model named correctly
    assert RequestModel.__name__ == "TestEndpointRequest"

    # Check that the request model has the correct fields
    assert RequestModel.__fields__.keys() == {"input", "input_without_datamodel"}

    # Check that the request fields have the correct types
    assert RequestModel.__fields__["input"].type_ == InputModel
    assert RequestModel.__fields__["input_without_datamodel"].type_ == Any


def test_get_response_model(mock_pipeline):
    """Test the get_response_model function."""

    endpoint = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=["output", "output_without_datamodel"],
    )

    ResponseModel = endpoint.get_response_model(mock_pipeline)

    # Check that the response model named correctly
    assert ResponseModel.__name__ == "TestEndpointResponse"

    # Check that the response model has the correct fields
    assert ResponseModel.__fields__.keys() == {"output", "output_without_datamodel"}

    # Check that the response fields have the correct types
    assert ResponseModel.__fields__["output"].type_ == OutputModel
    assert ResponseModel.__fields__["output_without_datamodel"].type_ == Any

    endpoint_with_one_output = Endpoint(
        name="test_endpoint",
        summary="Test endpoint",
        path="/test_endpoint",
        outputs=["output"],
    )

    ResponseModel = endpoint_with_one_output.get_response_model(mock_pipeline)

    # Check that the response model named correctly
    assert ResponseModel.__name__ == "TestEndpointResponse"

    # Check that the response model has the correct fields
    assert ResponseModel.__fields__.keys() == {"output"}

    # Check that the response fields have the correct types
    assert ResponseModel.__fields__["output"].type_ == OutputModel
