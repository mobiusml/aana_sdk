# Deployments

## AanaDeploymentHandle

AanaDeploymentHandle is a class that allows you to interact with Aana deployments.

::: aana.deployments.AanaDeploymentHandle

## Base classes for deployments

At the moment there are two base classes that you can use to create your own deployments:
- BaseDeployment: This is the base class for all Aana deployments.
- BaseTextGenerationDeployment: This is the base class for all text generation deployments (LLM deployments).

::: aana.deployments.base_text_generation_deployment.BaseDeployment
::: aana.deployments.base_text_generation_deployment.BaseTextGenerationDeployment

The BaseTextGenerationDeployment class defines a few classes for the output of the deployment:

::: aana.deployments.base_text_generation_deployment.LLMOutput
::: aana.deployments.base_text_generation_deployment.LLMBatchOutput
::: aana.deployments.base_text_generation_deployment.ChatOutput


## Text generation deployments

### Hugging Face Text Generation Deployment

Hugging Face Text Generation Deployment allows you to use Hugging Face transformers library to serve LLMs.

::: aana.deployments.hf_text_generation_deployment.HfTextGenerationConfig
::: aana.deployments.hf_text_generation_deployment.BaseHfTextGenerationDeployment
::: aana.deployments.hf_text_generation_deployment.HfTextGenerationDeployment

### vLLM Deployment

vLLM Deployment allows you to use vLLM library to serve LLMs.

::: aana.deployments.vllm_deployment.VLLMConfig
::: aana.deployments.vllm_deployment.VLLMDeployment

vLLM Deployment supports Gemlite for quantization. Use GemliteMode set to `GemliteMode.PREQUANTIZED` or `GemliteMode.ONTHEFLY` to enable it. GemliteQuantizationConfig is used to configure the quantization settings for "on-the-fly" mode.

::: aana.deployments.vllm_deployment.GemliteQuantizationConfig
::: aana.deployments.vllm_deployment.GemliteMode

## Whisper Deployment

Whisper Deployment allows you to do audio transcription and translation using Whisper model.

::: aana.deployments.whisper_deployment.WhisperConfig
::: aana.deployments.whisper_deployment.WhisperDeployment

The Whisper deployment defines a few classes for model configuration and output:

::: aana.deployments.whisper_deployment.WhisperComputeType
::: aana.deployments.whisper_deployment.WhisperModelSize

::: aana.deployments.whisper_deployment.WhisperOutput
::: aana.deployments.whisper_deployment.WhisperBatchOutput

## Haystack Deployment

Haystack Deployment allows to deploy Haystack Components as Aana deployments.

::: aana.deployments.haystack_component_deployment.HaystackComponentDeploymentConfig
::: aana.deployments.haystack_component_deployment.HaystackComponentDeployment

## Hugging Face Pipeline Deployment

Hugging Face Pipeline Deployment allows to deploy almost any Hugging Face pipeline as Aana deployment.

::: aana.deployments.hf_pipeline_deployment.HfPipelineConfig
::: aana.deployments.hf_pipeline_deployment.HfPipelineDeployment

## HF BLIP2 Deployment

HF BLIP2 Deployment allows to deploy BLIP2 model as Aana deployment.

::: aana.deployments.hf_blip2_deployment.HFBlip2Config
::: aana.deployments.hf_blip2_deployment.HFBlip2Deployment

The HF BLIP2 deployment defines a few classes for outputs:

::: aana.deployments.hf_blip2_deployment.CaptioningOutput
::: aana.deployments.hf_blip2_deployment.CaptioningBatchOutput

## HF Idefics2 Deployment

HF Idefics2 Deployment allows to deploy Idefics2 model as Aana deployment.

::: aana.deployments.idefics_2_deployment.Idefics2Config
::: aana.deployments.idefics_2_deployment.Idefics2Deployment

## Pyannote Speaker Diarization Deployment

Pyannote Speaker Diarization Deployment allows to deploy Pyannote speaker diarization model as Aana deployment.

::: aana.deployments.pyannote_speaker_diarization_deployment.PyannoteSpeakerDiarizationConfig
::: aana.deployments.pyannote_speaker_diarization_deployment.PyannoteSpeakerDiarizationDeployment

::: aana.deployments.pyannote_speaker_diarization_deployment.SpeakerDiarizationOutput

## VAD Deployment

VAD Deployment allows to deploy VAD model as Aana deployment.

::: aana.deployments.vad_deployment.VadConfig   
::: aana.deployments.vad_deployment.VadDeployment

::: aana.deployments.vad_deployment.SegmentX
::: aana.deployments.vad_deployment.VadOutput