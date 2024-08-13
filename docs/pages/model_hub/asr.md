# Automatic Speech Recognition (ASR) Models

[WhisperDeployment](./../../reference/deployments.md#aana.deployments.WhisperDeployment) allows you to transcribe or translate audio with Whisper models. The deployment is based on the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library.

[WhisperConfig](./../../reference/deployments.md#aana.deployments.WhisperConfig) is used to configure the Whisper deployment.

::: aana.deployments.WhisperConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the Whisper deployment for the [Whisper Medium model](https://huggingface.co/Systran/faster-whisper-medium).


!!! example "Whisper Medium"
    
    ```python
    WhisperDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=WhisperConfig(
            model_size=WhisperModelSize.MEDIUM,
            compute_type=WhisperComputeType.FLOAT16,
        ).model_dump(mode="json"),
    )
    ```

Model size is the one of the Whisper model sizes available in the `faster-whisper` library. `compute_type` is the data type to be used for the model.

Here are some other possible configurations for the Whisper deployment:

??? example "Whisper Tiny on CPU"
    
    ```python
    # for CPU do not specify num_gpus and use FLOAT32 compute type
    WhisperDeployment.options(
        num_replicas=1,
        user_config=WhisperConfig(
            model_size=WhisperModelSize.TINY,
            compute_type=WhisperComputeType.FLOAT32,
        ).model_dump(mode="json"),
    )
    ```