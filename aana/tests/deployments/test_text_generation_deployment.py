# ruff: noqa: S101
import pytest
from ray import serve

from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.sampling import SamplingParams
from aana.exceptions.runtime import PromptTooLongException
from aana.tests.utils import (
    compare_texts,
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


def get_expected_output(name):
    """Gets expected output for a given text_generation model."""
    if name == "vllm_llama2_7b_chat_deployment":
        return (
            "  Elon Musk is a South African-born entrepreneur, inventor, "
            "and business magnate who is best known for his innovative companies in"
        )
    elif name == "meta_llama3_8b_instruct_deployment":
        return (
            " Elon Musk is a South African-born entrepreneur, inventor,"
            "and business magnate. He is the CEO and CTO of SpaceX, "
            "CEO and product architect of Tesla"
        )
    elif name == "hf_phi3_mini_4k_instruct_text_gen_deployment":
        return (
            "Elon Musk is a prominent entrepreneur and business magnate known for "
            "his significant contributions to the technology and automotive industries. He was born"
        )
    else:
        raise ValueError(f"Unknown deployment name: {name}")  # noqa: TRY003


def get_expected_chat_output(name):
    """Gets expected output for a given text_generation model."""
    if name == "vllm_llama2_7b_chat_deployment":
        return (
            "  Elon Musk is a South African-born entrepreneur, inventor, "
            "and business magnate who is best known for his innovative companies in"
        )
    elif name == "meta_llama3_8b_instruct_deployment":
        return "Elon Musk is a South African-born entrepreneur, inventor, and business magnate. He is best known for his ambitious goals to revolutionize the transportation, energy"
    elif name == "hf_phi3_mini_4k_instruct_text_gen_deployment":
        return (
            "Elon Musk is a prominent entrepreneur and business magnate known for "
            "his significant contributions to the technology and automotive industries. He was born"
        )
    else:
        raise ValueError(f"Unknown deployment name: {name}")  # noqa: TRY003


def get_prompt(name):
    """Gets the prompt for a given text_generation model."""
    if name == "vllm_llama2_7b_chat_deployment":
        return "[INST] Who is Elon Musk? [/INST]"
    elif name == "hf_phi3_mini_4k_instruct_text_gen_deployment":
        return "<|user|>\ Who is Elon Musk? <|end|>\n<|assistant|>"
    elif name == "meta_llama3_8b_instruct_deployment":
        return "[INST] Who is Elon Musk? [/INST]"
    else:
        raise ValueError(f"Unknown deployment name: {name}")  # noqa: TRY003


@pytest.fixture(
    scope="function",
    params=get_deployments_by_type("VLLMDeployment")
    + get_deployments_by_type("HfTextGenerationDeployment"),
)
def setup_text_generation_deployment(app_setup, request):
    """Setup text_generation deployment."""
    name, deployment = request.param
    deployments = [
        {
            "name": "text_generation_deployment",
            "instance": deployment,
        }
    ]
    endpoints = []

    return name, deployment, app_setup(deployments, endpoints)


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
async def test_text_generation_deployments(setup_text_generation_deployment):
    """Test text_generation deployments."""
    name, deployment, app = setup_text_generation_deployment

    handle = serve.get_app_handle("text_generation_deployment")

    expected_text = get_expected_output(name)
    prompt = get_prompt(name)

    # test generate method
    output = await handle.generate.remote(
        prompt=prompt,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    )
    text = output["text"]

    compare_texts(expected_text, text)

    # test generate_stream method
    stream = handle.options(stream=True).generate_stream.remote(
        prompt=prompt,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    )
    text = ""
    async for chunk in stream:
        text += chunk["text"]

    compare_texts(expected_text, text)

    # test generate_batch method
    output = await handle.generate_batch.remote(
        prompts=[prompt, prompt],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    )
    texts = output["texts"]

    for text in texts:
        compare_texts(expected_text, text)

    # test chat method
    expected_text = get_expected_chat_output(name)
    dialog = ChatDialog(
        messages=[
            ChatMessage(
                role="user",
                content="Who is Elon Musk?",
            )
        ]
    )
    output = await handle.chat.remote(
        dialog=dialog,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    )

    response_message = output["message"]
    assert response_message.role == "assistant"
    text = response_message.content

    compare_texts(expected_text, text)

    # test chat_stream method
    stream = handle.options(stream=True).chat_stream.remote(
        dialog=dialog,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
    )

    text = ""
    async for chunk in stream:
        text += chunk["text"]

    compare_texts(expected_text, text)

    # test generate method with too long prompt
    with pytest.raises(PromptTooLongException):
        output = await handle.generate.remote(
            prompt=prompt * 1000,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
        )
