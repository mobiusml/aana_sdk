# Model Hub

<!-- Aana SDK provides a collection of pre-trained models that can be used for various tasks. -->

Model deployment is a crucial part of the machine learning workflow. Aana SDK uses concept of deployments to serve models.

The deployments are "recipes" that can be used to deploy models. With the same deployment, you can deploy multiple different models by providing specific configurations.

Aana SDK comes with a set of predefined deployments, like [VLLMDeployment](./../../reference/deployments.md#aana.deployments.VLLMDeployment) for serving Large Language Models (LLMs) with [vLLM](https://github.com/vllm-project/vllm/) library or [WhisperDeployment](./../../reference/deployments.md#aana.deployments.WhisperDeployment) for automatic Speech Recognition (ASR) based on the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library. 

Each deployment has its own configuration class that specifies which model to deploy and with which parameters. 

The model hub provides a collection of configurations for different models that can be used with the predefined deployments. 

The full list of predefined deployments can be found in the [Deployments](./../integrations.md).

!!! tip

    The Model Hub provides only a subset of the available models. You can deploy a lot more models using predefined deployments. For example, [Hugging Face Pipeline Deployment](./../../reference/deployments.md#aana.deployments.HfPipelineDeployment) is a generic deployment that can be used to deploy any model from the [Hugging Face Model Hub](https://huggingface.co/models) that can be used with [Hugging Face Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html). It would be impossible to list all the models that can be deployed with this deployment.

!!! tip
    
    The SDK is not limited to the predefined deployments. You can create your own deployment.

## How to Use the Model Hub

There are a few ways to use the Model Hub (from the simplest to the most advanced):

- Find the model configuration you are interested in and copy the configuration code to your project.

- Use the provided examples as a starting point to create your own configurations for existing deployments.

- Create a new deployment with your own configuration.

See [Tutorial](./../tutorial.md#deployments) for more information on how to use the deployments.

## Models by Category

- [Text Generation Models (LLMs)](./text_generation.md)
- [Image-to-Text Models](./image_to_text.md)
- [Automatic Speech Recognition (ASR) Models](./asr.md)
- [Hugging Face Pipeline Models](./hf_pipeline.md)
