# ruff: noqa: S101
from collections import defaultdict
from importlib import resources

import pytest

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_to_dict
from aana.core.models.whisper import WhisperParams
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

    # Test transcribe_batch method

    # Test transcribe_in_chunks method: Note that the expected asr output is different
    # TODO: Update once batched whisper PR is merged
    # expected_batched_output_path = resources.path(
    #     f"aana.tests.files.expected.whisper.{model_size}",
    #     f"{audio_file_name}_batched.json",
    # )
    # assert (
    #     expected_batched_output_path.exists()
    # ), f"Expected output not found: {expected_batched_output_path}"
    # with Path(expected_batched_output_path) as path, path.open() as f:
    #     expected_output_batched = json.load(f)

    # # Get expected vad segments
    # vad_path = resources.path(
    #     "aana.tests.files.expected.vad", f"{audio_file_name}_vad.json"
    # )
    # assert vad_path.exists(), f"vad expected predictions not found: {vad_path}"

    # with Path(vad_path) as path, path.open() as f:
    #     expected_output_vad = json.load(f)

    # final_input = [
    #     VadSegment(time_interval=seg["time_interval"], segments=seg["segments"])
    #     for seg in expected_output_vad["segments"]
    # ]

    # batched_stream = handle.options(stream=True).transcribe_in_chunks.remote(
    #     audio=audio,
    #     segments=final_input,
    #     batch_size=16,
    #     params=BatchedWhisperParams(temperature=0.0),
    # )

    # # Combine individual segments and compare with the final dict
    # transcript = ""
    # grouped_dict = defaultdict(list)
    # async for chunk in batched_stream:
    #     output = pydantic_to_dict(chunk)
    #     transcript += output["transcription"]["text"]
    #     grouped_dict["segments"].extend(output.get("segments", []))

    # grouped_dict["transcription"] = {"text": transcript}
    # grouped_dict["transcription_info"] = output.get("transcription_info")

    # compare_transcriptions(
    #     expected_output_batched,
    #     dict(grouped_dict),
    # )
