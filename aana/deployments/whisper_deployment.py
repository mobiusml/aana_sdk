from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, TypedDict, cast

import torch
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field
from ray import serve

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import InferenceException
from aana.models.core.video import Video
from aana.models.pydantic.asr_output import (
    AsrSegment,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.models.pydantic.whisper_params import WhisperParams


class WhisperComputeType(str, Enum):
    """The data type used by whisper models.

    See [cTranslate2 docs on quantization](https://opennmt.net/CTranslate2/quantization.html#quantize-on-model-conversion)
    for more information.

    Available types:
        - INT8
        - INT8_FLOAT32
        - INT8_FLOAT16
        - INT8_BFLOAT16
        - INT16
        - FLOAT16
        - BFLOAT16
        - FLOAT32
    """

    INT8 = "int8"
    INT8_FLOAT32 = "int8_float32"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class WhisperModelSize(str, Enum):
    """The whisper model.

    Available models:
        - TINY
        - TINY_EN
        - BASE
        - BASE_EN
        - SMALL
        - SMALL_EN
        - MEDIUM
        - MEDIUM_EN
        - LARGE_V1
        - LARGE_V2
        - LARGE
    """

    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE = "large"


class WhisperConfig(BaseModel):
    """The configuration for the whisper deployment from faster-whisper."""

    model_size: WhisperModelSize = Field(
        default=WhisperModelSize.BASE, description="The whisper model size."
    )
    compute_type: WhisperComputeType = Field(
        default=WhisperComputeType.FLOAT16, description="The compute type."
    )


class WhisperOutput(TypedDict):
    """The output of the whisper model.

    Attributes:
        segments (List[AsrSegment]): The ASR segments.
        transcription_info (AsrTranscriptionInfo): The ASR transcription info.
        transcription (AsrTranscription): The ASR transcription.
    """

    segments: list[AsrSegment]
    transcription_info: AsrTranscriptionInfo
    transcription: AsrTranscription


class WhisperBatchOutput(TypedDict):
    """The output of the whisper model for a batch of inputs.

    Attributes:
        segments (List[List[AsrSegment]]): The ASR segments for each media.
        transcription_info (List[AsrTranscriptionInfo]): The ASR transcription info for each media.
        transcription (List[AsrTranscription]): The ASR transcription for each media.
    """

    segments: list[list[AsrSegment]]
    transcription_info: list[AsrTranscriptionInfo]
    transcription: list[AsrTranscription]


@serve.deployment
class WhisperDeployment(BaseDeployment):
    """Deployment to serve Whisper models from faster-whisper."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the HFBlip2Config schema.
        """
        config_obj = WhisperConfig(**config)
        self.model_size = config_obj.model_size
        self.model_name = "whisper_" + self.model_size
        self.compute_type = config_obj.compute_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    # TODO: add audio support
    async def transcribe(
        self, media: Video, params: WhisperParams | None = None
    ) -> WhisperOutput:
        """Transcribe the media with the whisper model.

        Args:
            media (Video): The media to transcribe.
            params (WhisperParams): The parameters for the whisper model.

        Returns:
            WhisperOutput: The transcription output as a dictionary:
                segments (List[AsrSegment]): The ASR segments.
                transcription_info (AsrTranscriptionInfo): The ASR transcription info.
                transcription (AsrTranscription): The ASR transcription.

        Raises:
            InferenceException: If the inference fails.
        """
        if not params:
            params = WhisperParams()
        media_path: str = str(media.path)
        try:
            segments, info = self.model.transcribe(media_path, **params.dict())
        except Exception as e:
            raise InferenceException(self.model_name) from e

        asr_segments = [AsrSegment.from_whisper(seg) for seg in segments]
        asr_transcription_info = AsrTranscriptionInfo.from_whisper(info)
        transcription = "".join([seg.text for seg in asr_segments])
        asr_transcription = AsrTranscription(text=transcription)

        return WhisperOutput(
            segments=asr_segments,
            transcription_info=asr_transcription_info,
            transcription=asr_transcription,
        )

    async def transcribe_stream(
        self, media: Video, params: WhisperParams | None = None
    ) -> AsyncGenerator[WhisperOutput, None]:
        """Transcribe the media with the whisper model in a streaming fashion.

        Right now this is the same as transcribe, but we will add support for
        streaming in the future to support larger media and to make the ASR more responsive.

        Args:
            media (Video): The media to transcribe.
            params (WhisperParams): The parameters for the whisper model.

        Yields:
            WhisperOutput: The transcription output as a dictionary:
                segments (List[AsrSegment]): The ASR segments.
                transcription_info (AsrTranscriptionInfo): The ASR transcription info.
                transcription (AsrTranscription): The ASR transcription.
        """
        # TODO: add streaming support
        output = await self.transcribe(media, params)
        yield output

    async def transcribe_batch(
        self, media_batch: list[Video], params: WhisperParams = None
    ) -> WhisperBatchOutput:
        """Transcribe the batch of media with the Whisper model.

        Args:
            media_batch (list[Video]): The batch of media to transcribe.
            params (WhisperParams): The parameters for the whisper model.

        Returns:
            WhisperBatchOutput: The transcription output as a dictionary:
                segments (list[list[AsrSegment]]): The ASR segments for each media.
                transcription_info (list[AsrTranscriptionInfo]): The ASR transcription info for each media.
                transcription (list[AsrTranscription]): The ASR transcription for each media.

        Raises:
            InferenceException: If the inference fails.
        """
        if not params:
            params = WhisperParams()
        segments: list[list[AsrSegment]] = []
        infos: list[AsrTranscriptionInfo] = []
        transcriptions: list[AsrTranscription] = []
        for media in media_batch:
            output = await self.transcribe(media, params)
            segments.append(cast(list[AsrSegment], output["segments"]))
            infos.append(cast(AsrTranscriptionInfo, output["transcription_info"]))
            transcriptions.append(cast(AsrTranscription, output["transcription"]))

        return WhisperBatchOutput(
            segments=segments, transcription_info=infos, transcription=transcriptions
        )