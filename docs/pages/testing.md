# Testing

The project uses pytest for testing. To run the tests, use the following command:

```bash
poetry run pytest
```

Most deployment tests require GPU to run. They are skipped if the GPU is not available. Right now we don't GPU runner for our CI/CD pipeline, so if you change anything related to deployments, make sure to run the tests locally with GPU and mention it in the PR description.

If you are using VS Code, you can run the tests using the Test Explorer that is installed with the [Python extension](https://code.visualstudio.com/docs/python/testing).

## Testing Deployments

This guide explains how to test the deployments. Aana SDK provides a set of fixtures and utilities that help you to write tests for the deployments. 

### Goals

The goal is to verify that the deployment, wrapper around the model, works as expected. The goal is NOT to test the model itself. That's why we only test 1-2 deployment configurations, just to make sure that the deployment works as expected.

### Setup Deployment Fixture

The `setup_deployment` fixture is used to start Aana SDK application with the given deployment configuration. The fixture takes two arguments: 
- `deployment_name`: The name of the deployment. This is used to identify the deployment in the test.
- `deployment`: The deployment configuration. 

We use the `@pytest.mark.parametrize` decorator to run the test with multiple deployment configurations.

```python
deployments = [("your_deployment_name", your_deployment_config), ...]

@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_name, deployment", deployments)
async def test_your_deployment(setup_deployment, deployment_name, deployment):
    app = setup_deployment(deployment_name, deployment)
    ...
```

You don't need to import the `setup_deployment` fixture because it is automatically imported from conftest.py.

### Deployment Handle

The `AanaDeploymentHandle` class is used to interact with the deployment. The class allows you to call the methods on the deployment remotely. 

To create an instance of the `AanaDeploymentHandle` class, use the class method `create`. The method takes the deployment name as an argument.

```python
handle = await AanaDeploymentHandle.create(deployment_name)
```

### Verify Results Utility

The `aana.tests.utils.verify_deployment_results` utility is used to compare the expected output with the actual output. The utility takes two arguments:
- `expected_output_path (Path)`: The path to the expected output file.
- `output (Dict)`: The actual output of the deployment.

The expected outputs are stored in the `aana/tests/files/expected` directory. There is a convention to store the expected output files for each deployment in a separate directory.

For example,

```python
expected_output_path = (
    resources.path("aana.tests.files.expected", "")
    / "whisper"
    / f"{deployment_name}_{audio_file}.json"
)
```

### SAVE_EXPECTED_OUTPUT Environment Variable

`verify_deployment_results` has a built-in mechanism to save the actual output as the expected output. If you set the environment variable `SAVE_EXPECTED_OUTPUT` to `True` and `verify_deployment_results` does not find the expected output file, it will save the actual output as the expected output. It is useful when you are writing tests for a new deployment because you don't need to create the expected output files manually but only need to verify the output. But remember to set the environment variable back to `False` after you have created the expected output files and check created files manually to make sure they are correct.


### GPU Availability

If the deployment requires a GPU, make sure to add `@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")` decorator to the test function. It will skip the test if the GPU is not available which is common in CI/CD environments.


### Example

Here is an example of the test for the Whisper deployment. You can find the full test in the [`aana/tests/deployments/test_whisper_deployment.py`](https://github.com/mobiusml/aana_sdk/blob/main/aana/tests/deployments/test_whisper_deployment.py).

```python
import pytest
from importlib import resources
from aana.core.models.audio import Audio
from aana.core.models.whisper import WhisperParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.whisper_deployment import WhisperComputeType, WhisperConfig, WhisperDeployment, WhisperModelSize
from aana.tests.utils import is_gpu_available, verify_deployment_results

# Define the deployments to test as a list of tuples.
deployments = [
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
    )
]

# Skip the test if GPU is not available.
@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
# The test is asynchronous because it interacts with the deployment.
@pytest.mark.asyncio
# Parametrize the test with the deployments.
@pytest.mark.parametrize("deployment_name, deployment", deployments)
# Parametrize the test with the audio files (this can be anything else like prompts etc.).
@pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
# Define the test function, add `setup_deployment` fixture, and parameterized arguments to the function.
async def test_whisper_deployment(
    setup_deployment, deployment_name, deployment, audio_file
):
    """Test whisper deployment."""
    # Start the deployment.
    setup_deployment(deployment_name, deployment)

    # Create the deployment handle.
    handle = await AanaDeploymentHandle.create(deployment_name)

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
```
    