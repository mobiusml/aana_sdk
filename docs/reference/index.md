# Reference Documentation (Code API)

This section contains the reference documentation for the public API of the project. Quick links to the most important classes and functions are provided below.

## SDK

[`aana.AanaSDK`](./sdk.md#aana.AanaSDK) - The main class for interacting with the Aana SDK. Use it to register endpoints and deployments and to start the server.

## Endpoint

[`aana.api.Endpoint`](./endpoint.md#aana.api.Endpoint) - The base class for defining endpoints in the Aana SDK.

## Deployments

[Deployments](./deployments.md) contains information about how to deploy models with a number of predefined deployments for such models as Whisper, LLMs, Hugging Face models, and more.

## Models

- [Media Models](./models/media.md) - Models for working with media types like audio, video, and images.
- [Automatic Speech Recognition (ASR) Models](./models/asr.md) - Models for working with automatic speech recognition (ASR) models.
- [Caption Models](./models/captions.md) - Models for working with captions.
- [Chat Models](./models/chat.md) - Models for working with chat models.
- [Custom Config](./models/custom_config.md) - Custom Config model can be used to pass arbitrary configuration to the deployment.
- [Sampling Models](./models/sampling.md) - Contains Sampling Parameters model which can be used to pass sampling parameters to the LLM models.
- [Time Models](./models/time.md) - Contains time models like TimeInterval.
- [Types Models](./models/types.md) - Contains types models like Dtype.
- [Video Models](./models/video.md) - Models for working with video files.
- [Whisper Models](./models/whisper.md) - Models for working with whispers.

