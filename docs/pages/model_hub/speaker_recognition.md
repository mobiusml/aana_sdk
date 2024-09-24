# Speaker Recognition 

## Speaker Diarization (SD) Models

[PyannoteSpeakerDiarizationDeployment](./../../reference/deployments.md#aana.deployments.PyannoteSpeakerDiarizationDeployment) allows you to diarize the audio for speakers audio with pyannote models. The deployment is based on the [pyannote.audio](https://github.com/pyannote/pyannote-audio) library.

[PyannoteSpeakerDiarizationConfig](./../../reference/deployments.md#aana.deployments.SpeakerDiarizationConfig) is used to configure the Speaker Diarization deployment.

::: aana.deployments.PyannoteSpeakerDiarizationConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list


## Accessing Gated Models

The PyAnnote speaker diarization models are gated, requiring special access. To use these models:

1. **Request Access**:  
    Visit the [PyAnnote Speaker Diarization 3.1 model page](https://huggingface.co/pyannote/speaker-diarization-3.1) on Hugging Face. Log in, fil out the form, and request access.

2. **Approval**:  
    - If automatic, access is granted immediately.
    - If manual, wait for the model authors to approve your request.

3. **Set Up the SDK**:  
    After approval, add your Hugging Face access token to your `.env` file by setting the `HF_TOKEN` variable:

    ```plaintext
    HF_TOKEN=your_huggingface_access_token
    ```

    To get your Hugging Face access token, visit the [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens).


## Example Configurations

As an example, let's see how to configure the Pyannote Speaker Diarization deployment for the [Speaker Diarization-3.1 model](https://huggingface.co/pyannote/speaker-diarization-3.1).

!!! example "Speaker diarization-3.1"
    
    ```python
    from aana.deployments.pyannote_speaker_diarization_deployment import PyannoteSpeakerDiarizationDeployment, PyannoteSpeakerDiarizationConfig

    PyannoteSpeakerDiarizationDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.05},
        user_config=PyannoteSpeakerDiarizationConfig(
            model_name=("pyannote/speaker-diarization-3.1"),
            sample_rate=16000,
        ).model_dump(mode="json"),
    )
    ```

## Diarized ASR

Speaker Diarization output can be combined with ASR to generate transcription with speaker information. Further details and code snippet are available in [ASR model hub](./asr.md/#diarized-asr).

