# Half-Quadratic Quantization Models

[Half-Quadratic Quantization deployment](./../../reference/deployments.md#aana.deployments.HQQDeployment) allows you to serve *almost* any model from the [Hugging Face Hub](https://huggingface.co/models) and quantize it. It is a wrapper for [HQQ](https://github.com/mobiusml/hqq) so you can quantize and deploy *almost* any model from the Hugging Face Hub with a few lines of code.

[HQQConfig](./../../reference/deployments.md#aana.deployments.HQQConfig) is used to configure the HQQ deployment.

::: aana.deployments.HQQConfig
    options:
        show_bases: false
        heading_level: 4
        show_docstring_description: false
        docstring_section_style: list

### Example Configurations

As an example, let's see how to configure the Hugging Face pipeline deployment to quantize [https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/Salesforce/blip2-opt-2.7b).

!!! example "Meta-Llama-3.1-8B-Instruct"
    
    ```python
    from aana.deployments.hqq_deployment import HQQConfig, HQQDeployment

    HQQDeployment.options(
        num_replicas=1,
        ray_actor_options={"num_gpus": 0.5},
        user_config=HQQConfig(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            backend=HQQBackend.BITBLAS,
            quantize_on_fly=True,
            dtype=Dtype.FLOAT16,
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

Model ID is the Hugging Face model ID. `quantize_on_fly` is whether the model needs to be qunatized or model is pre-quantized and we just need to load it. We deploy the model with 4-bit quantization by setting `quantization_config` in the `HQQConfig`. You can pass extra arguments to the model in the `model_kwargs` dictionary.
