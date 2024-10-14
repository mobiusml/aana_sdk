# ruff: noqa: S101
import json
import re

import pytest
from pydantic import BaseModel, ValidationError

from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

deployments = [
    (
        "phi3_mini_4k_instruct_vllm_deployment",
        VLLMDeployment.options(
            num_replicas=1,
            max_ongoing_requests=1000,
            ray_actor_options={"num_gpus": 0.5},
            user_config=VLLMConfig(
                model_id="microsoft/Phi-3-mini-4k-instruct",
                dtype=Dtype.FLOAT16,
                gpu_memory_reserved=10000,
                enforce_eager=True,
                default_sampling_params=SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=1024,
                ),
                engine_args={
                    "trust_remote_code": True,
                },
            ).model_dump(mode="json"),
        ),
    ),
]


class CityDescription(BaseModel):
    """A test model for city descriptions."""

    city: str
    country: str
    description: str


@pytest.mark.parametrize(
    "setup_deployment",
    deployments,
    indirect=["setup_deployment"],
)
class TestStructuredGeneration:
    """Test schema and regex constrained generation with vLLM deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", ["Tell me about Vienna.", "Describe Berlin."])
    async def test_chat_with_json_schema(self, setup_deployment, query):
        """Test chat method with JSON schema."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        dialog = ChatDialog(
            messages=[
                ChatMessage(role="user", content=query),
            ]
        )

        # Define a JSON schema for CityDescription
        schema = json.dumps(CityDescription.model_json_schema(), indent=2)

        # Test chat method with JSON schema validation
        output = await handle.chat(
            dialog=dialog,
            sampling_params=SamplingParams(
                json_schema=schema, temperature=0.0, max_tokens=512
            ),
        )

        response_message = output["message"]
        assert response_message.role == "assistant"
        text = response_message.content
        print(text)

        # Validate JSON response against the schema
        try:
            CityDescription.model_validate_json(text)
        except ValidationError as e:
            pytest.fail(f"Response does not match schema: {e}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            "Tell me about Paris and New York. Return a list of dictionaries.",
            "Describe Tokyo and London. Return a list of dictionaries.",
        ],
    )
    async def test_chat_with_list_schema(self, setup_deployment, query):
        """Test chat method with list of city descriptions schema."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="user",
                    content=query,
                ),
            ]
        )

        # Define a JSON schema for a list of CityDescription
        from pydantic import RootModel

        CityDescriptionList = RootModel[list[CityDescription]]
        schema = json.dumps(CityDescriptionList.model_json_schema(), indent=2)

        # Test chat method with JSON schema validation
        output = await handle.chat(
            dialog=dialog,
            sampling_params=SamplingParams(
                json_schema=schema, temperature=0.0, max_tokens=512
            ),
        )

        response_message = output["message"]
        assert response_message.role == "assistant"
        text = response_message.content
        print(text)

        # Validate JSON response against the schema
        try:
            CityDescriptionList.model_validate_json(text)
        except ValidationError as e:
            pytest.fail(f"Response does not match schema: {e}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query, regex_pattern",
        [
            (
                "What is Pi? Give me the first 15 digits. Only return the number.",
                "(-)?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-][0-9]+)?",
            )
        ],
    )
    async def test_chat_with_regex(self, setup_deployment, query, regex_pattern):
        """Test chat method with regex."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="user",
                    content=query,
                ),
            ]
        )

        # Test chat method with regex validation
        output = await handle.chat(
            dialog=dialog,
            sampling_params=SamplingParams(
                regex_string=regex_pattern, temperature=0.0, max_tokens=32
            ),
        )

        response_message = output["message"]
        assert response_message.role == "assistant"
        text = response_message.content
        print(text)

        # Validate response against the regex pattern
        if not re.fullmatch(regex_pattern, text):
            pytest.fail(f"Response does not match regex pattern: {text}")
