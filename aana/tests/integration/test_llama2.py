# ruff: noqa: S101
# Test llama2 endpoints.

import hashlib
import json

import pytest

from aana.tests.utils import call_endpoint, check_output, is_gpu_available
from aana.utils.json import json_serializer_default

TARGET = "llama2"


def generate(
    target: str,
    port: int,
    route_prefix: str,
    prompt: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Generate text for a given prompt."""
    endpoint_path = "/llm/generate"
    data = {"prompt": prompt}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target,
        endpoint_path,
        data_hash,
        output,
        ignore_expected_output,
        expected_error,
    )
    return output


def generate_stream(
    target: str,
    port: int,
    route_prefix: str,
    prompt: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Generate text for a given prompt (streaming)."""
    endpoint_path = "/llm/generate_stream"
    data = {"prompt": prompt}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target,
        endpoint_path,
        data_hash,
        output,
        ignore_expected_output,
        expected_error,
    )
    return output


def chat(
    target: str,
    port: int,
    route_prefix: str,
    dialog: dict,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Chat with LLaMa2."""
    endpoint_path = "/llm/chat"
    data = {"dialog": dialog}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, data_hash, output, ignore_expected_output, expected_error
    )
    return output


def chat_stream(
    target: str,
    port: int,
    route_prefix: str,
    dialog: dict,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Chat with LLaMa2 (streaming)."""
    endpoint_path = "/llm/chat_stream"
    data = {"dialog": dialog}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, data_hash, output, ignore_expected_output, expected_error
    )
    return output


@pytest.fixture(scope="module")
def app(app_setup):
    """Setup app for a specific target."""
    return app_setup(TARGET)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.parametrize(
    "prompt, error",
    [
        ("[INST] Who is Elon Musk? [/INST]", None),
        ("[INST] Where is the Eiffel Tower? [/INST]", None),
        ("[INST] Who is Elon Musk? [/INST]" * 1000, "PromptTooLongException"),
    ],
)
def test_llama_generate(app, prompt, error):
    """Test llama generate endpoints."""
    target = TARGET
    handle, port, route_prefix = app

    generate(target, port, route_prefix, prompt, expected_error=error)

    generate_stream(target, port, route_prefix, prompt, expected_error=error)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.parametrize(
    "dialog, error",
    [
        ({"messages": [{"role": "user", "content": "Who is Elon Musk?"}]}, None),
        (
            {"messages": [{"role": "user", "content": "Where is the Eiffel Tower?"}]},
            None,
        ),
        (
            {"messages": [{"role": "user", "content": "Who is Elon Musk?" * 1000}]},
            "PromptTooLongException",
        ),
    ],
)
def test_llama_chat(app, dialog, error):
    """Test llama chat endpoint."""
    target = TARGET
    handle, port, route_prefix = app

    chat(target, port, route_prefix, dialog, expected_error=error)

    chat_stream(target, port, route_prefix, dialog, expected_error=error)
