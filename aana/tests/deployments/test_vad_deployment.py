# ruff: noqa: S101
from importlib import resources
from pathlib import Path

import pytest

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_to_dict
from aana.core.models.vad import VadParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.vad_deployment import VadConfig, VadDeployment
from aana.tests.utils import verify_deployment_results

deployments = [
    (
        "vad_deployment",
        VadDeployment.options(
            num_replicas=1,
            max_ongoing_requests=1000,
            ray_actor_options={"num_gpus": 0},
            user_config=VadConfig(
                model=(
                    "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/"
                    "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
                ),
                onset=0.5,
                sample_rate=16000,
            ).model_dump(mode="json"),
        ),
    )
]


@pytest.mark.parametrize("setup_deployment", deployments, indirect=True)
class TestVadDeployment:
    """Test VAD deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("audio_file", ["physicsworks.wav", "squirrel.wav"])
    async def test_vad(self, setup_deployment, audio_file):
        """Test VAD."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        audio_file_name = Path(audio_file).stem
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "vad"
            / f"{audio_file_name}.json"
        )

        path = resources.files("aana.tests.files.audios") / audio_file
        assert path.exists(), f"Audio not found: {path}"

        audio = Audio(path=path)

        output = await handle.asr_preprocess_vad(audio=audio, params=VadParams())
        output = pydantic_to_dict(output)
        verify_deployment_results(expected_output_path, output)
