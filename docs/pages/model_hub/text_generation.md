# Text Generation Models (LLMs)

Aana SDK has three deployments to serve text generation models (LLMs):

- [VLLMDeployment](./../../reference/deployments.md#aana.deployments.VLLMDeployment): allows you to efficiently serve Large Language Model (LLM) with the [vLLM](https://github.com/vllm-project/vllm/) library.

- [HfTextGenerationDeployment](./../../reference/deployments.md#aana.deployments.HfTextGenerationDeployment): uses the [Hugging Face Transformers](https://huggingface.co/transformers/) library to deploy text generation models.

- [HqqTextGenerationDeployment](./../../reference/deployments.md#aana.deployments.HqqTextGenerationDeployment): uses [Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq) to quantize and deploy text generation models.

All deployments have the same interface and provide similar capabilities. 

## vLLM Deployment

[VLLMConfig](./../../reference/deployments.md#aana.deployments.VLLMConfig) is used to configure the vLLM deployment.

::: aana.deployments.VLLMConfig
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

## Hugging Face Text Generation Deployment

[HfTextGenerationConfig](./../../reference/deployments.md#aana.deployments.HfTextGenerationConfig) is used to configure the vLLM deployment. 

::: aana.deployments.HfTextGenerationConfig
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

[HqqTexGenerationDeployment](./../../reference/deployments.md#aana.deployments.HqqTextGenerationDeployment) uses [Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq) to quantize and deploy text generation models from the [Hugging Face Hub](https://huggingface.co/models).

It supports already quantized models as well as quantizing models on the fly. The quantization is blazing fast and can be done on the fly with minimal overhead. Check out the [the collections of already quantized models](https://huggingface.co/collections/mobiuslabsgmbh/llama3-hqq-6604257a96fc8b9c4e13e0fe) from Mobius Labs.

[HqqTexGenerationConfig](./../../reference/deployments.md#aana.deployments.HqqTexGenerationConfig) is used to configure the HQQ Text Generation deployment.

::: aana.deployments.HqqTexGenerationConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### HQQ Backends

The HQQ Text Generation framework supports two primary backends, each optimized for specific scenarios:

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
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            model_kwargs={
                "attn_implementation": "sdpa"
            },
        ).model_dump(mode="json"),
    )
    ```

Model ID is the Hugging Face model ID. We set `quantize_on_fly=True` to quantize the model on the fly since the model is not pre-quantized. We deploy the model with 4-bit quantization by setting `quantization_config` in the `HQQConfig`. We use `HQQBackend.BITBLAS` as the backend for quantization, it is optional as BitBLAS is the default backend. You can pass extra arguments to the model in the `model_kwargs` dictionary.

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
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
        ).model_dump(mode="json"),
    )
    ```

Model ID is the Hugging Face model ID of a pre-quantized model. We use `HQQBackend.BITBLAS` as the backend for quantization, it is optional as BitBLAS is the default backend. We set the quantization configuration according to the model page.