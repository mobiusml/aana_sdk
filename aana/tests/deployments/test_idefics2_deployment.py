# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.chat import ChatMessage
from aana.core.models.image import Image
from aana.core.models.image_chat import ImageChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.idefics_2_deployment import Idefics2Config, Idefics2Deployment
from aana.tests.utils import verify_deployment_results
from aana.utils.core import get_object_hash

deployments = [
    (
        "idefics_2_8b_deployment",
        Idefics2Deployment.options(
            num_replicas=1,
            ray_actor_options={"num_gpus": 0.85},
            user_config=Idefics2Config(
                model="HuggingFaceM4/idefics2-8b",
                dtype=Dtype.FLOAT16,
                default_sampling_params=SamplingParams(
                    temperature=1.0, max_tokens=256, kwargs={"diversity_penalty": 0.0}
                ),
            ).model_dump(mode="json"),
        ),
    )
]


@pytest.mark.parametrize("setup_deployment", deployments, indirect=True)
class TestIdefics2Deployment:
    """Test Idefics 2 deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt, image_name",
        [("Who is the painter of the image?", "Starry_Night.jpeg")],
    )
    async def test_idefics2_deployment_chat(self, setup_deployment, prompt, image_name):
        """Test Idefics 2 deployments."""
        deployment_name, handle_name, app = setup_deployment
        handle = await AanaDeploymentHandle.create(handle_name)

        prompt_hash = get_object_hash(prompt)
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "idefics"
            / f"{deployment_name}_{image_name}_{prompt_hash}.json"
        )

        image = Image(path=resources.files("aana.tests.files.images") / image_name)
        dialog = ImageChatDialog.from_prompt(prompt=prompt, images=[image])

        # test chat method
        output = await handle.chat(dialog=dialog)
        output_message = output["message"]

        assert isinstance(output_message, ChatMessage)
        assert output_message.role == "assistant"
        verify_deployment_results(expected_output_path, output_message.content)

        # test chat_batch method
        batch_size = 2
        dialogs = [dialog] * batch_size
        outputs = await handle.chat_batch(dialogs=dialogs)
        assert len(outputs) == batch_size
        for output in outputs:
            output_message = output["message"]

            assert isinstance(output_message, ChatMessage)
            assert output_message.role == "assistant"
            verify_deployment_results(expected_output_path, output_message.content)

        # test chat_stream method
        stream = handle.chat_stream(dialog=dialog)
        text = ""
        async for chunk in stream:
            text += chunk["text"]

        verify_deployment_results(expected_output_path, text)
