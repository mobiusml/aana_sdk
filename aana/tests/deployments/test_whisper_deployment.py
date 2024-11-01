# ruff: noqa: S101
import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

import pytest

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_to_dict
from aana.core.models.vad import VadSegment
from aana.core.models.whisper import BatchedWhisperParams, WhisperParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.whisper_deployment import (
    WhisperComputeType,
    WhisperConfig,
    WhisperDeployment,
    WhisperModelSize,
)
from aana.tests.utils import verify_deployment_results

deployments = [
    (
        "whisper_tiny",
        WhisperDeployment.options(
            num_replicas=1,
            user_config=WhisperConfig(
                model_size=WhisperModelSize.TINY,
                compute_type=WhisperComputeType.FLOAT32,
            ).model_dump(mode="json"),
        ),
    ),
    (
        "whisper_medium",
        WhisperDeployment.options(
            num_replicas=1,
            ray_actor_options={"num_gpus": 0.25},
            user_config=WhisperConfig(
                model_size=WhisperModelSize.MEDIUM,
                compute_type=WhisperComputeType.FLOAT16,
            ).model_dump(mode="json"),
        ),
    ),
    (
        "whisper_turbo",
        WhisperDeployment.options(
            num_replicas=1,
            ray_actor_options={"num_gpus": 0.25},
            user_config=WhisperConfig(
                model_size=WhisperModelSize.TURBO,
                compute_type=WhisperComputeType.FLOAT16,
            ).model_dump(mode="json"),
        ),
    ),
]


@pytest.mark.parametrize("setup_deployment", deployments, indirect=True)
class TestWhisperDeployment:
    """Test Whisper deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
    async def test_transcribe(self, setup_deployment, audio_file):
        """Test transcribe methods."""
        deployment_name, handle_name, _ = setup_deployment

        handle = await AanaDeploymentHandle.create(handle_name)

        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "whisper"
            / f"{deployment_name}_{audio_file}.json"
        )

        # Test transcribe method
        path = resources.files("aana.tests.files.audios") / audio_file
        assert path.exists(), f"Audio not found: {path}"
        audio = Audio(path=path, media_id=audio_file)

        output = await handle.transcribe(
            audio=audio, params=WhisperParams(word_timestamps=True, temperature=0.0)
        )

        verify_deployment_results(expected_output_path, output)

        # Test transcribe_stream method
        stream = handle.transcribe_stream(
            audio=audio, params=WhisperParams(word_timestamps=True, temperature=0.0)
        )

        # Combine individual segments and compare with the final dict
        grouped_dict = defaultdict(list)
        transcript = ""
        async for chunk in stream:
            output = pydantic_to_dict(chunk)
            transcript += output["transcription"]["text"]
            grouped_dict["segments"].extend(output.get("segments", []))

        grouped_dict["transcription"] = {"text": transcript}
        grouped_dict["transcription_info"] = output.get("transcription_info")

        verify_deployment_results(expected_output_path, grouped_dict)

        # Test transcribe_in_chunks method: Note that the expected asr output is different
        expected_output_path = (
            resources.files("aana.tests.files.expected")
            / "whisper"
            / f"{deployment_name}_{audio_file}_batched.json"
        )
        audio_file_name = audio_file.removesuffix(".wav")
        # Get expected vad segments
        vad_path = (
            resources.files("aana.tests.files.expected.vad") / f"{audio_file_name}.json"
        )

        assert vad_path.exists(), f"vad expected predictions not found: {vad_path}"
        with Path(vad_path) as path, path.open() as f:
            expected_output_vad = json.load(f)
        vad_input = [
            VadSegment(time_interval=seg["time_interval"], segments=seg["segments"])
            for seg in expected_output_vad["segments"]
        ]

        batched_stream = handle.transcribe_in_chunks(
            audio=audio,
            vad_segments=vad_input,
            batch_size=16,
            params=BatchedWhisperParams(temperature=0.0),
        )
        # Combine individual segments and compare with the final dict
        grouped_dict = defaultdict(list)
        transcript = ""
        async for chunk in batched_stream:
            output = pydantic_to_dict(chunk)
            transcript += output["transcription"]["text"]
            grouped_dict["segments"].extend(output.get("segments", []))
        grouped_dict["transcription"] = {"text": transcript}
        grouped_dict["transcription_info"] = output.get("transcription_info")
        verify_deployment_results(expected_output_path, grouped_dict)

        # Test with whisper internal VAD
        batched_stream = handle.transcribe_in_chunks(
            audio=audio,
            batch_size=16,
            params=BatchedWhisperParams(temperature=0.0),
        )
        # Combine individual segments and compare with the final dict
        grouped_dict = defaultdict(list)
        transcript = ""
        async for chunk in batched_stream:
            output = pydantic_to_dict(chunk)
            transcript += output["transcription"]["text"]
            grouped_dict["segments"].extend(output.get("segments", []))
        grouped_dict["transcription"] = {"text": transcript}
        grouped_dict["transcription_info"] = output.get("transcription_info")
        verify_deployment_results(expected_output_path, grouped_dict)
