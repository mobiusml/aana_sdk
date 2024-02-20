# ruff: noqa: S101
# Test llama2 endpoints.


import pytest

from aana.tests.utils import is_gpu_available, is_using_deployment_cache

TARGET = "llama2"

LLM_GENERATE = "/llm/generate"
LLM_GENERATE_STREAM = "/llm/generate_stream"
LLM_CHAT = "/llm/chat"
LLM_CHAT_STREAM = "/llm/chat_stream"


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "prompt, error",
    [
        ("[INST] Who is Elon Musk? [/INST]", None),
        ("[INST] Where is the Eiffel Tower? [/INST]", None),
        ("[INST] Who is Elon Musk? [/INST]" * 1000, "PromptTooLongException"),
    ],
)
def test_llama_generate(call_endpoint, prompt, error):
    """Test llama generate endpoints."""
    call_endpoint(
        LLM_GENERATE,
        {"prompt": prompt},
        expected_error=error,
    )

    call_endpoint(
        LLM_GENERATE_STREAM,
        {"prompt": prompt},
        expected_error=error,
    )


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "dialog, error",
    [
        ({"messages": [{"role": "user", "content": "Who is Elon Musk?"}]}, None),
        (
            {"messages": [{"role": "user", "content": "Where is the Eiffel Tower?"}]},
            None,
        ),
        (
            {"messages": [{"role": "user", "content": "Who is Elon Musk?" * 100}]},
            None,
        ),
        (
            {"messages": [{"role": "user", "content": "Who is Elon Musk?" * 1000}]},
            "PromptTooLongException",
        ),
    ],
)
def test_llama_chat(call_endpoint, dialog, error):
    """Test llama chat endpoint."""
    call_endpoint(
        LLM_CHAT,
        {"dialog": dialog},
        expected_error=error,
    )

    call_endpoint(
        LLM_CHAT_STREAM,
        {"dialog": dialog},
        expected_error=error,
    )
