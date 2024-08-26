# ruff: noqa: S101
from importlib import resources
from pathlib import Path

import pytest

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_to_dict
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.speaker_diarization_deployment import (
    SDConfig,
    SpeakerDiarizationDeployment,
)
from aana.tests.utils import verify_deployment_results

deployments = [
    (
        "sd_deployment",
        SpeakerDiarizationDeployment.options(
            num_replicas=1,
            max_ongoing_requests=1000,
            ray_actor_options={"num_gpus": 0.05},
            user_config=SDConfig(
                model_name=("pyannote/speaker-diarization-3.1"),
                sample_rate=16000,
            ).model_dump(mode="json"),
        ),
    )
]


@pytest.mark.parametrize("setup_deployment", deployments, indirect=True)
class TestSpeakerDiarizationDeployment:
    """Test SD deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("audio_file", ["sd_sample.wav"])  # physicsworks.wav
    async def test_sd(self, setup_deployment, audio_file):
        """Test SD."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        audio_file_name = Path(audio_file).stem
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "sd"
            / f"{audio_file_name}.json"
        )

        path = resources.files("aana.tests.files.audios") / audio_file
        assert path.exists(), f"Audio not found: {path}"

        audio = Audio(path=path)

        output = await handle.diarize_segments(audio=audio)
        output = pydantic_to_dict(output)

        verify_deployment_results(expected_output_path, output)
