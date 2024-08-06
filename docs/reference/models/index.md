# Models

## [Media Models](./media.md)

Aana SDK provides models for such media types as audio, video, and images. These models make it easy to work with media files, download them, convert from other formats, and more.

- [`aana.core.models.Audio`](./media.md#aana.core.models.Audio)
- [`aana.core.models.Video`](./media.md#aana.core.models.Video)
- [`aana.core.models.Image`](./media.md#aana.core.models.Image)
- [`aana.core.models.Media`](./media.md#aana.core.models.Media)


Don't use models above in the Endpoint definition (as request or response body) because they are not serializable to JSON. Instead, use the following models as input models and call `convert_input_to_object()` method to convert them to the appropriate media type.

- [`aana.core.models.VideoInput`](./media.md#aana.core.models.VideoInput)
- [`aana.core.models.ImageInput`](./media.md#aana.core.models.ImageInput)
- [`aana.core.models.VideoInputList`](./media.md#aana.core.models.VideoInputList)
- [`aana.core.models.ImageInputList`](./media.md#aana.core.models.ImageInputList)

For example, in the following code snippet, the `ImageInput` model is used in the endpoint definition, and then it is converted to the `Image` object.

```python
class ImageClassificationEndpoint(Endpoint):
    async def run(self, image: ImageInput) -> ImageClassificationOutput:
        image_obj: Image = image.convert_input_to_object()
        ...
```

## [Automatic Speech Recognition (ASR) Models](./asr.md)

Models for working with automatic speech recognition (ASR) models. These models represent the output of ASR model like whisper and represent the transcription, segments, and words etc.

<!-- - [`aana.core.models.asr.AsrTranscription`](./asr.md#aana.core.models.asr.AsrTranscription)
- [`aana.core.models.asr.AsrTranscriptionInfo`](./asr.md#aana.core.models.asr.AsrTranscriptionInfo)
- [`aana.core.models.asr.AsrSegment`](./asr.md#aana.core.models.asr.AsrSegment)
- [`aana.core.models.asr.AsrWord`](./asr.md#aana.core.models.asr.AsrWord) -->

## [Caption Models](./captions.md)

Models for working with captions. These models represent the output of image captioning models like BLIP 2.

## [Chat Models](./chat.md)

Models for working with chat models. These models represent the input and output of chat models and models for OpenAI-compatible API.

## [Visual Chat Models](./image_chat.md)

Models for working with visual chat models. These models represent the input of chat models containing text and image input for describing visual content of vision-language model (VLM).

## [Custom Config](./custom_config.md)

Custom Config model can be used to pass arbitrary configuration to the deployment.

## [Sampling Models](./sampling.md)

Contains Sampling Parameters model which can be used to pass sampling parameters to the LLM models.

## [Time Models](./time.md)

Contains time models like TimeInterval.

## [Types Models](./types.md)

Contains types models like Dtype.

## [VAD Models](./vad.md)

Contains Voice Activity Detection (VAD) models like VadParams, VadSegment, and VadSegments.

## [Video Models](./video.md)

Contains video models like VideoMetadata, VideoStatus, and VideoParams.

## [Whisper Models](./whisper.md)

Contains models for working with whisper models like WhisperParams.
