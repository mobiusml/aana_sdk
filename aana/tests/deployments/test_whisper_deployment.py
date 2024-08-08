from importlib import resources

import pytest

from aana.core.models.audio import Audio
from aana.core.models.whisper import WhisperParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.whisper_deployment import (
    WhisperComputeType,
    WhisperConfig,
    WhisperDeployment,
    WhisperModelSize,
)
from aana.tests.utils import verify_deployment_results

# Define the deployments to test as a list of tuples.
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
            max_ongoing_requests=1000,
            ray_actor_options={"num_gpus": 0.25},
            user_config=WhisperConfig(
                model_size=WhisperModelSize.MEDIUM,
                compute_type=WhisperComputeType.FLOAT16,
            ).model_dump(mode="json"),
        ),
    ),
]


# Parametrize the test with the deployments.
@pytest.mark.parametrize("setup_deployment", deployments, indirect=True)
class TestWhisperDeployment:
    """Test Whisper deployment."""

    # The test is asynchronous because it interacts with the deployment.
    @pytest.mark.asyncio
    # Parametrize the test with the audio files (this can be anything else like prompts etc.).
    @pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
    # Define the test function, add `setup_deployment` fixture, and parameterized arguments to the function.
    async def test_transcribe(self, setup_deployment, audio_file):
        """Test transcribe methods."""
        # Get deployment name, handle name, and app instance from the setup_deployment fixture.
        deployment_name, handle_name, app = setup_deployment

        # Create the deployment handle, use the handle name from the setup_deployment fixture.
        handle = await AanaDeploymentHandle.create(handle_name)

        # Define the path to the expected output file.
        # There are 3 parts:
        # - The path to the expected output directory (aana/tests/files/expected), should not be changed.
        # - The name of the subdirectory for the deployment (whisper), should be changed for each deployment type.
        # - File name with based on the parameters (deployment_name, audio_file, etc.).
        expected_output_path = (
            resources.path("aana.tests.files.expected", "")
            / "whisper"
            / f"{deployment_name}_{audio_file}.json"
        )

        # Run the deployment method.
        path = resources.path("aana.tests.files.audios", audio_file)
        assert path.exists(), f"Audio not found: {path}"

        audio = Audio(path=path, media_id=audio_file)

        output = await handle.transcribe(
            audio=audio, params=WhisperParams(word_timestamps=True, temperature=0.0)
        )

        # Verify the results with the expected output.
        verify_deployment_results(expected_output_path, output)
