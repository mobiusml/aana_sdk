# Hugging Face Pipeline Models

[Hugging Face Pipeline deployment](./../../reference/deployments.md#aana.deployments.hf_pipeline_deployment.HfPipelineDeployment) allows you to serve *almost* any model from the [Hugging Face Hub](https://huggingface.co/models). It is a wrapper for [Hugging Face Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) so you can deploy and scale *almost* any model from the Hugging Face Hub with a few lines of code.

!!! Tip
    To use HF Pipeline deployment, install required libraries with `pip install transformers` or include extra dependencies using `pip install aana[transformers]`.


[HfPipelineConfig](./../../reference/deployments.md#aana.deployments.hf_pipeline_deployment.HfPipelineConfig) is used to configure the Hugging Face Pipeline deployment.

::: aana.deployments.hf_pipeline_deployment.HfPipelineConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the Hugging Face Pipeline deployment to serve [Salesforce BLIP-2 OPT-2.7b model](https://huggingface.co/Salesforce/blip2-opt-2.7b).

!!! example "BLIP-2 OPT-2.7b"
    
    ```python
    from transformers import BitsAndBytesConfig
    from aana.deployments.hf_pipeline_deployment import HfPipelineConfig, HfPipelineDeployment

    HfPipelineDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.25},
        user_config=HfPipelineConfig(
            model_id="Salesforce/blip2-opt-2.7b",
            task="image-to-text",
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True),
            },
        ).model_dump(mode="json"),
    )
    ```

Model ID is the Hugging Face model ID. `task` is one of the [Hugging Face Pipelines tasks](https://huggingface.co/transformers/main_classes/pipelines.html) that the model can perform. We deploy the model with 4-bit quantization by setting `quantization_config` in the `model_kwargs` dictionary. You can pass extra arguments to the model in the `model_kwargs` dictionary.
