# Text Generation Models (LLMs)

Aana SDK has three deployments to serve text generation models (LLMs):

- [VLLMDeployment](./../../reference/deployments.md#aana.deployments.vllm_deployment.VLLMDeployment): allows you to efficiently serve Large Language Models (LLM) and Vision Language Models (VLM) with the [vLLM](https://github.com/vllm-project/vllm/) library.

- [HfTextGenerationDeployment](./../../reference/deployments.md#aana.deployments.HfTextGenerationDeployment): uses the [Hugging Face Transformers](https://huggingface.co/transformers/) library to deploy text generation models.

- [HqqTextGenerationDeployment](./../../reference/deployments.md#aana.deployments.HqqTextGenerationDeployment): uses [Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq) to quantize and deploy text generation models.

All deployments have the same interface and provide similar capabilities. 

## vLLM Deployment

vLLM deployment allows you to efficiently serve Large Language Models (LLM) and Vision Language Models (VLM) with the [vLLM](https://github.com/vllm-project/vllm/) library.

[VLLMConfig](./../../reference/deployments.md#aana.deployments.vllm_deployment.VLLMConfig) is used to configure the vLLM deployment.

::: aana.deployments.vllm_deployment.VLLMConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the vLLM deployment for the [Meta Llama 3 8B Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). 

!!! example "Meta Llama 3 8B Instruct"

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.core.models.types import Dtype
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.45},
        user_config=VLLMConfig(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            dtype=Dtype.AUTO,
            gpu_memory_reserved=30000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
        ).model_dump(mode="json"),
    )
    ```

Model name is the Hugging Face model ID. We use `Dtype.AUTO` to let the deployment choose the best data type for the model. We reserve 30GB of GPU memory for the model. We set `enforce_eager=True` to helps to reduce memory usage but may harm performance. We also set the default sampling parameters for the model.

VLLM deployment also supports Vision Language Models (VLM). Here is an example configuration for the [Phi 3.5 Vision Instruct model](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).

!!! example "Phi 3.5 Vision Instruct"

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 1.0},
        user_config=VLLMConfig(
            model_id="microsoft/Phi-3.5-vision-instruct",
            gpu_memory_reserved=12000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            max_model_len=2048,
            engine_args=dict(
                trust_remote_code=True,
                max_num_seqs=32,
                limit_mm_per_prompt={"image": 3},
            ),
        ).model_dump(mode="json"),
    )
    ```

Here are some other example configurations for the VLLM deployment. Keep in mind that the list is not exhaustive. You can deploy any model that is [supported by the vLLM library](https://docs.vllm.ai/en/latest/models/supported_models.html).


??? example "Llama 2 7B Cha t with AWQ quantization"

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.core.models.types import Dtype
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        user_config=VLLMConfig(
            model_id="TheBloke/Llama-2-7b-Chat-AWQ",
            dtype=Dtype.AUTO,
            quantization="awq",
            gpu_memory_reserved=13000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            chat_template="llama2",
        ).model_dump(mode="json"),
    )
    ```

??? example "InternLM 2.5 7B Chat"

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.core.models.types import Dtype
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.45},
        user_config=VLLMConfig(
            model_id="internlm/internlm2_5-7b-chat",
            dtype=Dtype.AUTO,
            gpu_memory_reserved=30000,
            max_model_len=50000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            engine_args={"trust_remote_code": True},
        ).model_dump(mode="json"),
    )
    ```

??? example "Phi 3 Mini 4K Instruct"

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.core.models.types import Dtype
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=VLLMConfig(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            dtype=Dtype.AUTO,
            gpu_memory_reserved=10000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            engine_args={
                "trust_remote_code": True,
            },
        ).model_dump(mode="json"),
    )
    ```

??? example "Qwen2-VL 7B Instruct"

    For LLaVA-NeXT-Video and Qwen2-VL, the latest release of huggingface/transformers doesn’t work yet (as of 18 September 2024), so we need to use a developer version (21fac7abba2a37fae86106f87fcf9974fd1e3830) for now. This can be installed by running the following command:

    ```bash
    pip install git+https://github.com/huggingface/transformers.git@21fac7abba2a37fae86106f87fcf9974fd1e3830
    ```

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 1.0},
        user_config=VLLMConfig(
            model_id="Qwen/Qwen2-VL-7B-Instruct",
            gpu_memory_reserved=40000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            max_model_len=4096,
            engine_args=dict(
                limit_mm_per_prompt={"image": 3},
            ),
        ).model_dump(mode="json"),
    )
    ```

??? example "Pixtral 12B 2409"

    The model is gated so you need to the [model page](https://huggingface.co/mistralai/Pixtral-12B-2409), request access to the model and set `HF_TOKEN` environment variable to your Hugging Face API token.

    ```python
    from aana.core.models.sampling import SamplingParams
    from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment

    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 1.0},
        user_config=VLLMConfig(
            model_id="mistralai/Pixtral-12B-2409",
            gpu_memory_reserved=40000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            max_model_len=4096,
            engine_args=dict(
                tokenizer_mode="mistral",
                limit_mm_per_prompt={"image": 3},
            ),
        ).model_dump(mode="json"),
    )
    ```

### Structured Generation

Structured generation is a feature that allows you to generate structured data using the vLLM deployment forcing LLM to adhere to a specific JSON schema or regular expression pattern.

Structured generation is supported only for the vLLM deployment at the moment. 

To enable structured generation, you need to pass JSON schema or regular expression pattern to `SamplingParams` object.

```python
# For JSON schema set json_schema parameter to the JSON schema string
sampling_params = SamplingParams(json_schema=schema, temperature=0.0, max_tokens=512)

# For regular expression set regex_string parameter to the regular expression pattern
sampling_params = SamplingParams(regex_string=regex_pattern, temperature=0.0, max_tokens=512)

# Pass the sampling_params to one of the vLLM deployment methods like chat or chat_stream
# Here handle is an AanaDeploymentHandle for the vLLM deployment.
response = await handle.chat(dialog, sampling_params=sampling_params)
```

You can use Pydantic models to generate JSON schema.

```python
import json
from pydantic import BaseModel

class CityDescription(BaseModel):
    city: str
    country: str
    description: str

schema = json.dumps(CityDescription.model_json_schema())
# {"properties": {"city": {"title": "City", "type": "string"}, "country": {"title": "Country", "type": "string"}, "description": {"title": "Description", "type": "string"}}, "required": ["city", "country", "description"], "title": "CityDescription", "type": "object"}
```

You can find detailed tutorials on how to use structured generation in the [Structured Generation notebook](https://github.com/mobiusml/aana_sdk/blob/main/notebooks/structured_generation.ipynb).

## Hugging Face Text Generation Deployment

[HfTextGenerationConfig](./../../reference/deployments.md#aana.deployments.hf_text_generation_deployment.HfTextGenerationConfig) is used to configure the vLLM deployment. 

::: aana.deployments.hf_text_generation_deployment.HfTextGenerationConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the Hugging Face Text Generation deployment for the [Phi 3 Mini 4K Instruct model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

!!! example "Phi 3 Mini 4K Instruct"

    ```python
    from aana.deployments.hf_text_generation_deployment import HfTextGenerationConfig, HfTextGenerationDeployment

    HfTextGenerationDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        user_config=HfTextGenerationConfig(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            model_kwargs={
                "trust_remote_code": True,
            },
        ).model_dump(mode="json"),
    )
    ```

Model ID is the Hugging Face model ID. `trust_remote_code=True` is required to load the model from the Hugging Face model hub. You can define other model arguments in the `model_kwargs` dictionary.

Here are other example configurations for the Hugging Face Text Generation deployment. Keep in mind that the list is not exhaustive. You can deploy other text generation models that are [supported by the Hugging Face Transformers library](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending).


??? example "Phi 3 Mini 4K Instruct with 4-bit quantization"

    ```python
    from transformers import BitsAndBytesConfig
    from aana.deployments.hf_text_generation_deployment import HfTextGenerationConfig, HfTextGenerationDeployment

    HfTextGenerationDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        user_config=HfTextGenerationConfig(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            model_kwargs={
                "trust_remote_code": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=False, load_in_4bit=True
                ),
            },
        ).model_dump(mode="json"),
    )
    ```


## Half-Quadratic Quantization (HQQ) Text Generation Deployment

[HqqTexGenerationDeployment](./../../reference/deployments.md#aana.deployments.hqq_text_generation_deployment.HqqTextGenerationDeployment) uses [Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq) to quantize and deploy text generation models from the [Hugging Face Hub](https://huggingface.co/models).

It supports already quantized models as well as quantizing models on the fly. The quantization is blazing fast and can be done on the fly with minimal overhead. Check out the [the collections of already quantized models](https://huggingface.co/collections/mobiuslabsgmbh/llama3-hqq-6604257a96fc8b9c4e13e0fe) from Mobius Labs.

[HqqTexGenerationConfig](./../../reference/deployments.md#aana.deployments.hqq_text_generation_deployment.HqqTexGenerationConfig) is used to configure the HQQ Text Generation deployment.

::: aana.deployments.hqq_text_generation_deployment.HqqTexGenerationConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### HQQ Backends

The HQQ Text Generation framework supports two backends, each optimized for specific scenarios:

1. **HqqBackend.BITBLAS (Default)**
    - **Library Installation**: Install via:
     ```bash
     pip install bitblas
     ```
     More details can be found on the [BitBLAS GitHub page](https://github.com/microsoft/BitBLAS).
    - **Compatibility**: Works on a broader range of GPUs, including older models.
    - **Precision Support**: Supports both 4-bit and 2-bit quantization, allowing for more compact models and efficient inference.
    - **Strengths**: BitBLAS excels in handling large batch sizes, especially when properly configured. But HQQ is optimized for decoding with a batch size of 1 leading to slower inference times compared to the `TORCHAO_INT4` backend.
     - **Limitations**: Slower initialization due to the need for per-shape and per-GPU compilation.

2. **HqqBackend.TORCHAO_INT4**
    - **Library Installation**: No additional installation required.
    - **Compatibility**: Only works on Ampere and newer GPUs, limiting its usage to more recent hardware.
    - **Precision Support**: Supports only 4-bit quantization.
    - **Strengths**: Much faster to initialize compared to BitBLAS, making it a good choice for situations where quick startup times are crucial. Faster inference times compared to the `BITBLAS` backend.
    - **Limitations**: It doesn't support 2-bit quantization.


### Example Configurations

#### On-the-fly Quantization

As an example, let's see how to configure HQQ Text Generation deployment to quantize and deploy the [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) model.

!!! example "Meta-Llama-3.1-8B-Instruct"
    
    ```python
    from hqq.core.quantize import BaseQuantizeConfig
    from aana.deployments.hqq_text_generation_deployment import (
        HqqBackend,
        HqqTexGenerationConfig,
        HqqTextGenerationDeployment,
    )

    HqqTextGenerationDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.5},
        user_config=HqqTexGenerationConfig(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            backend=HqqBackend.BITBLAS,
            quantize_on_fly=True,
            quantization_config=BaseQuantizeConfig(nbits=4, group_size=64, axis=1),
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512
            ),
            model_kwargs={
                "attn_implementation": "sdpa"
            },
        ).model_dump(mode="json"),
    )
    ```

Model ID is the Hugging Face model ID. We set `quantize_on_fly=True` to quantize the model on the fly since the model is not pre-quantized. We deploy the model with 4-bit quantization by setting `quantization_config` in the `HqqConfig`. We use `HqqBackend.BITBLAS` as the backend for quantization, it is optional as BitBLAS is the default backend. You can pass extra arguments to the model in the `model_kwargs` dictionary.

#### Pre-quantized Models

You can also deploy already quantized models with HQQ Text Generation deployment. Here is an example of deploying the 


!!! example "Quantized Meta-Llama-3.1-8B-Instruct"
    
    ```python
    from hqq.core.quantize import BaseQuantizeConfig
    from aana.deployments.hqq_text_generation_deployment import (
        HqqBackend,
        HqqTexGenerationConfig,
        HqqTextGenerationDeployment,
    )

    HqqTextGenerationDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.5},
        user_config=HqqTexGenerationConfig(
            model_id="mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib",
            backend=HqqBackend.BITBLAS,
            quantization_config=BaseQuantizeConfig(
                nbits=4,
                group_size=64,
                quant_scale=False,
                quant_zero=False,
                axis=1,
            ),
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512
            ),
        ).model_dump(mode="json"),
    )
    ```

Model ID is the Hugging Face model ID of a pre-quantized model. We use `HqqBackend.BITBLAS` as the backend for quantization, it is optional as BitBLAS is the default backend. We set the quantization configuration according to the model page.