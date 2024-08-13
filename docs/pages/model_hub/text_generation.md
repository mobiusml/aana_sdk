# Text Generation Models (LLMs)

Aana SDK has two deployments to serve text generation models (LLMs):

- [VLLMDeployment](./../../reference/deployments.md#aana.deployments.VLLMDeployment): allows you to efficiently serve Large Language Model (LLM) with the [vLLM](https://github.com/vllm-project/vllm/) library.

- [HfTextGenerationDeployment](./../../reference/deployments.md#aana.deployments.HfTextGenerationDeployment): uses the [Hugging Face Transformers](https://huggingface.co/transformers/) library to deploy text generation models.

Both deployments have the same interface and provide similar capabilities. 

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
    VLLMDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.45},
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

Model name is the Hugging Face model ID. We use `Dtype.AUTO` to let the deployment choose the best data type for the model. We reserve 30GB of GPU memory for the model. We set `enforce_eager=True` to helps to reduce memory usage but may harm performance. We also set the default sampling parameters for the model.


Here are some other example configurations for the VLLM deployment. Keep in mind that the list is not exhaustive. You can deploy any model that is [supported by the vLLM library](https://docs.vllm.ai/en/latest/models/supported_models.html).


??? example "Llama 2 7B Cha t with AWQ quantization"

    ```python
    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        user_config=VLLMConfig(
            model="TheBloke/Llama-2-7b-Chat-AWQ",
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
    VLLMDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.45},
        user_config=VLLMConfig(
            model="internlm/internlm2_5-7b-chat",
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
    VLLMDeployment.options(
        num_replicas=1,
        max_ongoing_requests=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=VLLMConfig(
            model="microsoft/Phi-3-mini-4k-instruct",
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