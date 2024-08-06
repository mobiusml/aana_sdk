# Deployments

Aana SDK comes with a set of predefined deployments that you can use out of the box to deploy models.

## Whisper

Whisper deployment allows you to transcribe audio with an automatic Speech Recognition (ASR) model based on the [faster-whisper](https://github.com/SYSTRAN/faster-whisper). 

See [WhisperDeployment](./../reference/deployments.md#aana.deployments.WhisperDeployment) to learn more about the deployment capabilities.

```python
from aana.deployments.whisper_deployment import WhisperDeployment, WhisperConfig, WhisperModelSize, WhisperComputeType

WhisperDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=WhisperConfig(model_size=WhisperModelSize.MEDIUM, compute_type=WhisperComputeType.FLOAT16).model_dump(mode="json"),
)
```

## vLLM

vLLM deployment allows you to efficiently serve Large Language Model (LLM) with the [vLLM](https://github.com/vllm-project/vllm/) library.

See [VLLMDeployment](./../reference/deployments.md#aana.deployments.VLLMDeployment) to learn more about the deployment capabilities.

```python
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

VLLMDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    user_config=VLLMConfig(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        dtype=Dtype.AUTO,
        gpu_memory_reserved=30000,
        enforce_eager=True,
        default_sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
        ),
    ).model_dump(mode="json"),
)
```

## Idefics2

Idefics 2 deployment allows you to serve the [Idefics 2 models](https://huggingface.co/docs/transformers/main/en/model_doc/idefics2) using the [Hugging Face Transformers](https://huggingface.co/transformers/) library. Idefics 2 is a vision-language model (VLM) that can answer questions about images, describe visual content, create stories grounded on multiple images, or simply behave as a pure language model without visual inputs.

Idefics 2 deployment also supports using `Flash Attention 2` to boost the efficiency of the transformer model. You can set the value to `True` or leave it to `None`, so the deployment will check the availability of the `Flash Attention 2` on the server node, automatically.

See [Idefics2Deployment](./../reference/deployments.md#aana.deployments.Idefics2Deployment) to learn more about the deployment capabilities.

```python
from aana.deployments.idefics_2_deployment import Idefics2Config, Idefics2Deployment

Idefics2Deployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.85},
    user_config=Idefics2Config(
        model="HuggingFaceM4/idefics2-8b",
        dtype=Dtype.FLOAT16,
        enable_flash_attention_2=True,
    ).model_dump(mode="json"),
)
```

## Hugging Face Transformers

Hugging Face Pipeline deployment allows you to serve *almost* any model from the [Hugging Face Hub](https://huggingface.co/models). It is a wrapper for [Hugging Face Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) so you can deploy and scale *almost* any model from the Hugging Face Hub with a few lines of code.

See [HfPipelineDeployment](./../reference/deployments.md#aana.deployments.HfPipelineDeployment) to learn more about the deployment capabilities.

```python
from transformers import BitsAndBytesConfig
from aana.deployments.hf_pipeline_deployment import HfPipelineConfig, HfPipelineDeployment

HfPipelineDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    user_config=HfPipelineConfig(
        model_id="Salesforce/blip2-opt-2.7b",
        task="image-to-text",
        model_kwargs={
            "quantization_config": BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True),
        },
    ).model_dump(mode="json"),
)
```

There are a few notebooks that demonstrate how to use the Hugging Face Transformers deployments:

- [HF Pipeline deployment notebook](https://github.com/mobiusml/aana_sdk/tree/main/notebooks/hf_pipeline_deployment.ipynb)
- [HF Text Generation deployment notebook](https://github.com/mobiusml/aana_sdk/tree/main/notebooks/hf_text_gen_deployment.ipynb)

## Haystack

Haystack integration allows you to build Retrieval-Augmented Generation (RAG) systems with the [Deepset Haystack](https://github.com/deepset-ai/haystack). 

See [Haystack integration notebook](https://github.com/mobiusml/aana_sdk/tree/main/notebooks/haystack_integration.ipynb) for a detailed example.

## OpenAI-compatible Chat Completions API

The OpenAI-compatible Chat Completions API allows you to access the Aana applications with any OpenAI-compatible client. See [OpenAI-compatible API docs](openai_api.md) for more details.
