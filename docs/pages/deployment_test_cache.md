# Deployment Test Cache

The deployment test cache is a feature of integration tests that allows you to simulate running the model endpoints without having to go to the effort of downloading the model, loading it, and running on a GPU. This is useful to save time as well as to be able to run the integration tests without needing a GPU (for example if you are on your laptop without internet access).

To mark a function as cacheable so its output can be stored in the deployment cache, annotate it with `@test_cache` imported from [aana.deployments.base_deployment](https://github.com/mobiusml/aana_sdk/tree/main/aana/deployments/base_deployment.py). Here's our StableDiffusion 2 deployment from above with the generate method annotated:

```python
class StableDiffusion2Deployment(BaseDeployment):
    """Stable Diffusion 2 deployment."""

    async def apply_config(self, _config: dict[str, Any]):
        """Apply configuration.

        The method is called when the deployment is created or updated.

        Normally we'd have a Config object, a TypedDict, to represent configurable parameters. In this case, hardcoded values are used and we load the model and scheduler from HuggingFace. You could also use the HuggingFace pipeline deployment class in `aana.deployments.hf_pipeline_deployment.py`.
        """

        # Load the model and processor from HuggingFace
        model_id = "stabilityai/stable-diffusion-2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=Dtype.FLOAT16.to_torch(),
            scheduler=EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            ),
            device_map="auto",
        )

        # Move the model to the GPU.
        self.model.to(device)

    @test_cache
    async def generate_single(self, prompt: str) -> StableDiffusion2Output:
        """Runs the model on a given prompt and returns the first output.

        Arguments:
            prompt (str): the prompt to the model.

        Returns:
            StableDiffusion2Output: a dictionary with one key containing the result
        """
        image = self.model(prompt).images[0]
        return {"image": image}

```

It just needs to be added to your inference methods. In this case, we have only one, `generate_single()`. 

There are a few environment variables that can be set to control the behavior of the tests:

- `USE_DEPLOYMENT_CACHE`: If set to `true`, the tests will use the deployment cache to avoid downloading the models and running the deployments. This is useful for running integration tests faster and in the environment where GPU is not available.
- `SAVE_DEPLOYMENT_CACHE`: If set to `true`, the tests will save the deployment cache after running the deployments. This is useful for updating the deployment cache if new deployments or tests are added.

### How to use the deployment cache environment variables

Here are some examples of how to use the deployment cache environment variables.

#### Do you want to run the tests normally using GPU?
    
```bash
USE_DEPLOYMENT_CACHE=false
SAVE_DEPLOYMENT_CACHE=false
```

This is the default behavior. The tests will run normally using GPU and the deployment cache will be completely ignored.

#### Do you want to run the tests faster without GPU?

```bash
USE_DEPLOYMENT_CACHE=true
SAVE_DEPLOYMENT_CACHE=false
```

This will run the tests using the deployment cache to avoid downloading the models and running the deployments. The deployment cache will not be updated after running the deployments. Only use it if you are sure that the deployment cache is up to date.

#### Do you want to update the deployment cache?

```bash
USE_DEPLOYMENT_CACHE=false
SAVE_DEPLOYMENT_CACHE=true
```

This will run the tests normally using GPU and save the deployment cache after running the deployments. Use it if you have added new deployments or tests and want to update the deployment cache.
