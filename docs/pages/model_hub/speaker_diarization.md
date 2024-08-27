# Speaker Diarization (SD) Models

[SpeakerDiarizationDeployment](./../../reference/deployments.md#aana.deployments.SpeakerDiarizationDeployment) allows you to diarize the audio for speakers audio with pyannote models. The deployment is based on the [pyannote.audio](https://github.com/pyannote/pyannote-audio) library.

[SpeakerDiarizationConfig](./../../reference/deployments.md#aana.deployments.SpeakerDiarizationConfig) is used to configure the Speaker Diarization deployment.

::: aana.deployments.SpeakerDiarizationConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the Speaker Diarization deployment for the [Speaker Diarization-3.1 model](https://huggingface.co/pyannote/speaker-diarization-3.1).

Note that pyannote speaker diarization models are gated. Get access to the model from (https://huggingface.co/pyannote/speaker-diarization-3.1) and use HuggingFace access token via setting environment variable in the SDK (`HF_TOKEN` variable in `.env`) before using the model.

!!! example "Speaker diarization-3.1"
    
    ```python
    from aana.deployments.speaker_diarization_deployment import SpeakerDiarizationDeployment, SpeakerDiarizationConfig

    SpeakerDiarizationDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.05},
        user_config=SpeakerDiarizationConfig(
            model_name=("pyannote/speaker-diarization-3.1"),
            sample_rate=16000,
        ).model_dump(mode="json"),
    )
    ```