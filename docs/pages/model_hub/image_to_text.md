# Image-to-Text Models  

Aana SDK has two deployments to serve image-to-text models:

- [Idefics2Deployment](./../../reference/deployments.md#aana.deployments.idefics_2_deployment.Idefics2Deployment): used to deploy the [Idefics2](https://huggingface.co/docs/transformers/main/en/model_doc/idefics2) models. Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text outputs.

- [HFBlip2Deployment](./../../reference/deployments.md#aana.deployments.idefics_2_deployment.HFBlip2Deployment): used to deploy the [BLIP-2](https://huggingface.co/docs/transformers/en/model_doc/blip-2) models. `HFBlip2Deployment` only supports image captioning capabilities of the BLIP-2 model.

!!! Tip
    To use Idefics2 or HF BLIP2 deployments, install required libraries with `pip install transformers` or include extra dependencies using `pip install aana[transformers]`.

## Idefics2 Deployment

[Idefics2Config](./../../reference/deployments.md#aana.deployments.idefics_2_deployment.Idefics2Config) is used to configure the Idefics2 deployment.

::: aana.deployments.idefics_2_deployment.Idefics2Config
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the Idefics2 deployment for the [Hugging Face Idefics2 8B model](https://huggingface.co/HuggingFaceM4/idefics2-8b).

!!! example "Hugging Face Idefics2 8B"

    ```python
    from aana.core.models.types import Dtype
    from aana.deployments.idefics_2_deployment import Idefics2Config, Idefics2Deployment

    Idefics2Deployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.85},
        user_config=Idefics2Config(
            model_id="HuggingFaceM4/idefics2-8b",
            dtype=Dtype.FLOAT16,
        ).model_dump(mode="json"),
    )
    ```

Model is the Hugging Face model ID. `dtype=Dtype.FLOAT16` is used to specify the data type to be used for the model. Idefics2 supports `Dtype.BFLOAT16` and it is generally faster but not supported by all GPUs. You can define other model arguments in the `model_kwargs` dictionary.

## BLIP-2 Deployment

[HFBlip2Config](./../../reference/deployments.md#aana.deployments.hf_blip2_deployment.HFBlip2Config) is used to configure the BLIP-2 deployment.    

::: aana.deployments.hf_blip2_deployment.HFBlip2Config
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the BLIP-2 deployment for the [Salesforce BLIP-2 OPT-2.7b model](https://huggingface.co/Salesforce/blip2-opt-2.7b).

!!! example "BLIP-2 OPT-2.7b"

    ```python
    from aana.core.models.types import Dtype
    from aana.deployments.hf_blip2_deployment import HFBlip2Config, HFBlip2Deployment

    HFBlip2Deployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        user_config=HFBlip2Config(
            model_id="Salesforce/blip2-opt-2.7b",
            dtype=Dtype.FLOAT16,
            batch_size=2,
            num_processing_threads=2,
        ).model_dump(mode="json"),
    )
    ```

Model is the Hugging Face model ID. We use `dtype=Dtype.FLOAT16` to load the model in half-precision for faster inference and lower memory usage. `batch_size` and `num_processing_threads` are used to configure the batch size (the bigger the batch size, the more memory is required) and the number of processing threads respectively.
