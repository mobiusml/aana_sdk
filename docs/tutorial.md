# Aana SDK Tutorial - Adding and Running a New Project

Aana SDK is an SDK for deploying and serving multimodal machine-learning models of Mobius Labs.

The SDK is designed according to the following principles:

Reliability: Aana is designed to be reliable and robust. It is built to be fault-tolerant and to handle failures gracefully.
Scalability: Aana is designed to be scalable. It is built on top of Ray, a distributed computing framework, and can be easily scaled to multiple servers.
Efficiency: Aana is designed to be efficient. It is built to be fast and parallel and to use resources efficiently.
Easy to Use: Aana is designed to be easy to use by developers. It is built to be modular, with a lot of automation and abstraction.
These design principles are reflected in the architecture of the SDK. The SDK is built on top of Ray, a distributed computing framework.

Although we are trying to follow these principles, we are aware that there is always room for improvement. The SDK is still in development, and not all features are fully implemented. We are constantly working on improving the SDK, and we welcome any feedback or suggestions.

## Code overview

    aana/ - top level source code directory for the project

        alembic/ - directory for database automigration

            versions/ - individual migrations

        api/ - code for generating an API from deployment and endpoint configurations

        config/ - various configuration objects, including settings, but preconfigured deployments

            db.py - config for the database

            deployments.py - preconfigured for deployments

            settings.py - app settings

        deployments/ - classes for individual ray deployments to wrap models or other functionality for ray

        exceptions/ - custom exception classes

        models/ - model classes for data

            core/ - core data models for the SDK

            db/ - database models (SQLAlchemy)

            pydantic/ - pydantic models (web request inputs + outputs, other data)

        projects/ - prebuilt projects with endpoints

            chat_with_video/ - extract audio and visual information and query it with a chat interface

            llama2/ - large language model with streaming responses

            lowercase/ - a simple example project with no AI that just converts text inputs to lowercase

            stablediffusion2/ - the basis for this tutorial

            whisper/ - automatic speech recognition and transcription API

        repository/ - repository classes for storage

            datastore/ - repositories for structured databases (SQL)

        tests/ - automated tests for the SDK

            db/ - tests for database functions

            deployments/ - tests for model deployments

            files/ - files used for tests

            unit/ - unit tests

        utils/ - various utility functionality

            chat_templates/ - templates for chat models

        cli.py - command-line interface to build and deploy the SDK as an app

        sdk.py - base class for defining an app/project using the SDK

## Creating a New Project

A runnable application using the SDK is a _project_. Default projects are located in `aana/sdk/projects`, and consist, minimally, of a Python module defining the application. Typically we organize a project into its own directory, putting the application definition in `app.py`, optionally with endpoints or other functionality in other Python modules, for example endpoint definitions in `endpoints.py`. This application can make use of existing deployments, or can add new ones as needed. 

For this example, we'll create a project that runs the Stable Diffusion 2 image generation model, for which we'll also create a new deployment. 

We'll start by creating a directory `./stablediffusion2` (don't forget to add an empty `__init__.py` if your development environment doesn't add this automatically). Then we add `app.py`. We need to create an instance of the `AanaSDK` class, 

```python
from aana.sdk import AanaSDK

from .deployments import stablediffusion2_deployment
from .endpoints import IMAGEGEN_DEPLOYMENT_NAME, ImageGenerationEndpoint

aana_app = AanaSDK(name="stablediffusion2")

aana_app.register_deployment(
    name=IMAGEGEN_DEPLOYMENT_NAME,
    instance=stablediffusion2_deployment,
)

aana_app.register_endpoint(
    name="generate_image",
    path="/generate_image",
    summary="Generates an image from a text prompt",
    endpoint_cls=ImageGenerationEndpoint,
)
```

Of course, this won't run yet, because we need to define our deployment and our endpoint, so let's do that!


## Adding a New Deployment

A ray deployment is a standardized interface for any kind of functionality that needs to manage state (loading a model onto a GPU counts as state). New deployments should inherit from `aana.deployments.base_deployment.BaseDeployment`. The deployment should have an async apply_config() method to initialize the deployment, but you can structure the operational methods of the deployment however you like. Additionally, a deployment config for the deployment will have to be created.

Let's make a file `./deployments.py`:

```python
from typing import Any, TypedDict

import PIL
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from ray import serve

from aana.core.models.types import Dtype
from aana.deployments.base_deployment import BaseDeployment


class StableDiffusion2Output(TypedDict):
    """Output class for StableDiffusion2Deployment."""

    image: PIL.Image.Image


@serve.deployment
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

Now that we have a Deployment class, we need to create an instance of that class. The usual way to do that with Ray is a little, er, *un*usual:

```python
stablediffusion2_deployment = StableDiffusion2Deployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    ray_actor_options={"num_gpus": 1},
    user_config={}, # This is what gets passed to apply_config()
)
```

Now we have a something we can run inference on! The only thing missing now is an endpoint definition. Let's add `./endpoints.py`:

```python
from typing import Annotated, TypedDict

import numpy as np
from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle

IMAGEGEN_DEPLOYMENT_NAME = "image_generation_deployment"

class ImageGenerationEndpointOutput(TypedDict):
    """Output model for the image generation endpoint."""

    image: Annotated[
        list, Field(description="The generated image as a array of pixel values.")
    ]


class ImageGenerationEndpoint(Endpoint):
    """Endpoint for image generation."""

    async def initialize(self):
        """Initialize the endpoint.
        
        Here we load a handle to the remote Ray deployment for image generation. The handle allows us to seamlessly make (`async`) calls to functions on the Deployment class istance, even if it's running in another process, or on another machine altogether.
        """
        self.image_generation_handle = await AanaDeploymentHandle.create(
            IMAGEGEN_DEPLOYMENT_NAME
        )

    async def run(self, prompt: str) -> ImageGenerationEndpointOutput:
        """Run the image generation endpoint.
        
        This calls our remote endpoint and formats the results.
        
        Because most of our production code works focuses on handling structured response data, we haven't added support for returning binary files. Maybe that could be your first contribution?
        """
        image_generation_output = await self.image_generation_handle.generate_single(
            prompt=prompt,
        )
        image = np.array(image_generation_output["image"]).tolist()
        return ImageGenerationEndpointOutput(image=image)
```

Now we're ready to run the app and call endpoints!

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0 poetry run aana deploy .app:aana_app
```

Lots of log messages from Ray and the SDK itself will scroll through. By default this will open an interface on port 8000. At the end, you should see

`Deployed Successfully.`  

In another tab/window, we can make a request to the endpoint using cURL:

```bash
$ curl -X POST 0.0.0.0:8000/generate_image -F body='{"prompt": "dogs playing poker but neo-cubist"}'

{"image": [[[45,25,102],[49,26,102], ...[29,62,141]]]}
```

Well, that's just an array of pixels expressed as bytes. Not very useful unless we want to do something else on top. But we can add another deployment for the BLIP2 image captioning model, and turn the image back into text! This time we'll just use the BLIP2 deployment already defined in `aana.deployments.` Let's go back to `./endpoints.py` and add the following:

```python
from aana.deployments.hf_blip2_deployment import HFBlip2Deployment, HFBlip2DeploymentOptions
from aana.models.core.image import Image


IMAGE_CAPTION_DEPLOYMENT_NAME = "image_caption_deployment"

class ImageGenerationCaptionEndpointOutput(TypedDict):
    """Output model for generating and then captioning an image."""

    caption: Annotated(str, Field(description="The caption of the image generated from the prompt"))

class ImageGenerationCaptionEndpoint(Endpoint):
        """Endpoint for generating and captioning an image."""

    async def initialize(self):
        """Initialize the endpoint.
        
        Once again we are loading handles, this time to two different deployments. Even though we now have two endpoints requesting handles to the image generation deployment (in this endpoint and in the old one), it will still only be deployed once and shared between endpoints."""
        self.image_generation_handle = await AanaDeploymentHandle.create(
            IMAGEGEN_DEPLOYMENT_NAME
        )
        self.image_caption_handle = await AanaDeploymentHandle.create(IMAGE_CAPTION_DEPLOYMENT_NAME)

    async def run(self, prompt: Prompt) -> ImageGenerationEndpointOutput:
        """Run the image generation and captioning endpoint.
        
        This calls our remote endpoint and formats the results.
        """
        image_generation_output = await self.image_generation_handle.generate_single(
            prompt=prompt,
        )
        image = Image(numpy=np.array(image_generation_output["image"]))
        caption_result = await self.image_caption_handle.generate(image)
        
        return ImageGenerationCaptionEndpointOutput(caption=caption_result["caption"])

```

Now we register the BLIP2 deployment and the new endpoint in `./app.py`:

```python
from aana.configs.deployments import hf_blip2_opt_2_7b_deployment

from endpoints import IMAGE_CAPTION_DEPLOYMENT_NAME, ImageGenerationCaptionEndpoint

aana_app.register_deployment(
    name=IMAGE_CAPTION_DEPLOYMENT_NAME, deployment=hf_blip2_opt_2_7b_deployment
)

aana_app.register_endpoint(
    name="generate_and_caption_image",
    path="/generate_and_caption_image",
    summary="Generates an image from a text prompt and a caption from the image",
    endpoint_cls=ImageGenerationCaptionEndpoint,
)


```

We can run this like before:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0 poetry run aana deploy .app:aana_app
```

Then make a request:

```bash
curl -X POST 0.0.0.0:8000/generate_image -F body='{"prompt": "dogs playing poker but neo-cubist"}'

```

That's at least a human readable result. 

## A note about testing
We have unit tests for freestanding functions and tasks. Additionally, we have tests for deployments and integration tests. Since testing deep learning models poses some problems - both in terms of reproducibility and in terms of CI/CD tests on common source control infrastructure - so we developed the deployment test cache.

You can run the existing tests with `poetry run pytest aana`.

### Deployment test cache
The deployment test cache is a feature of integration tests that allows you to simulate running the model endpoints without having to go to the effort of downloading the model, loading it, and running on a GPU. This is useful to save time as well as to be able to run the integration tests without needing a GPU (for example if you are on your laptop without internet access).

To mark a function as cacheable so its output can be stored in the deployment cache, annotate it with @test_cache imported from `[aana.utils.test](../aana/utils/test.py)`. Here's our StableDiffusion 2 deployment from above with the generate method annotated:

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


To enable the deployment test cache for existing tests, set the environment variable `USE_DEPLOYMENT_CACHE=true` when running `pytest`.

If you add new tests and new deployments, you must first run `pytest` with the environment variables `USE_DEPLOYMENT_CACHE=false` and `SAVE_DEPLOYMENT_CACHE=true`.


This will generate and save some JSON files with the SDK code base which represent the
deployment cache; be sure to commit them when you commit the rest of your changes. 

Once those files are generated you can set `USE_DEPLOYMENT_CACHE=true` and it will skip loading
models and running inference on methods or functions where `@test_cache` is set.
