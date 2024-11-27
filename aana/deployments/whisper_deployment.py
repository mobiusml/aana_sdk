import asyncio
import os
from collections.abc import AsyncGenerator
from enum import Enum
from pathlib import Path
from typing import Any, cast

import nvidia.cudnn.lib
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from pydantic import BaseModel, ConfigDict, Field
from ray import serve
from typing_extensions import TypedDict

from aana.core.models.asr import (
    AsrSegment,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_protected_fields
from aana.core.models.vad import VadSegment
from aana.core.models.whisper import BatchedWhisperParams, WhisperParams
from aana.deployments.base_deployment import BaseDeployment, exception_handler
from aana.exceptions.runtime import InferenceException

# Workaround for CUDNN issue with cTranslate2:
cudnn_path = str(Path(nvidia.cudnn.lib.__file__).parent)
os.environ["LD_LIBRARY_PATH"] = (
    cudnn_path + "/:" + os.environ.get("LD_LIBRARY_PATH", "")
)


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
        - LARGE_V3
        - LARGE
        - TURBO
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
    LARGE_V3 = "large-v3"
    LARGE = "large"
    TURBO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"  # SYSTRAN/faster-whisper do not have turbo model in HF.


class WhisperConfig(BaseModel):
    """The configuration for the whisper deployment from faster-whisper.

    Attributes:
        model_size (WhisperModelSize | str): The whisper model size. Defaults to WhisperModelSize.TURBO.
        compute_type (WhisperComputeType): The compute type. Defaults to WhisperComputeType.FLOAT16.
    """

    model_size: WhisperModelSize | str = Field(
        default=WhisperModelSize.TURBO,
        description="The whisper model size. Either one of WhisperModelSize or HuggingFace model ID in Ctranslate2 format.",
    )
    compute_type: WhisperComputeType = Field(
        default=WhisperComputeType.FLOAT16, description="The compute type."
    )

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


class WhisperOutput(TypedDict):
    """The output of the whisper model.

    Attributes:
        segments (list[AsrSegment]): The ASR segments.
        transcription_info (AsrTranscriptionInfo): The ASR transcription info.
        transcription (AsrTranscription): The ASR transcription.
    """

    segments: list[AsrSegment]
    transcription_info: AsrTranscriptionInfo
    transcription: AsrTranscription


class WhisperBatchOutput(TypedDict):
    """The output of the whisper model for a batch of inputs.

    Attributes:
        segments (list[list[AsrSegment]]): The ASR segments for each audio.
        transcription_info (list[AsrTranscriptionInfo]): The ASR transcription info for each audio.
        transcription (list[AsrTranscription]): The ASR transcription for each audio.
    """

    segments: list[list[AsrSegment]]
    transcription_info: list[AsrTranscriptionInfo]
    transcription: list[AsrTranscription]


@serve.deployment(max_ongoing_requests=1)
class WhisperDeployment(BaseDeployment):
    """Deployment to serve Whisper models from faster-whisper."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        The configuration should conform to the WhisperConfig schema.
        """
        config_obj = WhisperConfig(**config)
        self.model_size = config_obj.model_size
        self.model_name = "whisper_" + self.model_size
        self.compute_type = config_obj.compute_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        self.batched_model = BatchedInferencePipeline(
            model=self.model,
        )

    async def check_health(self):
        """Check the health of the deployment and clear torch cache to prevent memory leaks."""
        torch.cuda.empty_cache()
        await super().check_health()

    async def transcribe(
        self, audio: Audio, params: WhisperParams | None = None
    ) -> WhisperOutput:
        """Transcribe the audio with the whisper model.

        Args:
            audio (Audio): The audio to transcribe.
            params (WhisperParams): The parameters for the whisper model.

        Returns:
            WhisperOutput: The transcription output as a dictionary:

                - segments (List[AsrSegment]): The ASR segments.
                - transcription_info (AsrTranscriptionInfo): The ASR transcription info.
                - transcription (AsrTranscription): The ASR transcription.

        Raises:
            InferenceException: If the inference fails.
        """
        asr_segments = []
        asr_transcription_info = None
        async for output in self.transcribe_stream(audio, params):
            asr_segments.extend(output["segments"])
            asr_transcription_info = output["transcription_info"]

        transcription = "".join([seg.text for seg in asr_segments])
        asr_transcription = AsrTranscription(text=transcription)

        return WhisperOutput(
            segments=asr_segments,
            transcription_info=asr_transcription_info,
            transcription=asr_transcription,
        )

    @exception_handler
    async def transcribe_stream(
        self, audio: Audio, params: WhisperParams | None = None
    ) -> AsyncGenerator[WhisperOutput, None]:
        """Transcribe the audio with the whisper model in a streaming fashion.

        Args:
            audio (Audio): The audio to transcribe.
            params (WhisperParams): The parameters for the whisper model.

        Yields:
            WhisperOutput: The transcription output as a dictionary:

                - segments (list[AsrSegment]): The ASR segments.
                - transcription_info (AsrTranscriptionInfo): The ASR transcription info.
                - transcription (AsrTranscription): The ASR transcription.
        """
        if not params:
            params = WhisperParams()
        audio_array = audio.get_numpy()
        if not audio_array.any():
            # For silent audios/no audio tracks, return empty output with language as silence
            yield WhisperOutput(
                segments=[],
                transcription_info=AsrTranscriptionInfo(
                    language="silence", language_confidence=1.0
                ),
                transcription=AsrTranscription(text=""),
            )
        else:
            try:
                segments, info = self.model.transcribe(
                    audio_array, **params.model_dump()
                )
                asr_transcription_info = AsrTranscriptionInfo.from_whisper(info)
                for segment in segments:
                    await asyncio.sleep(0)
                    asr_segments = [AsrSegment.from_whisper(segment)]
                    asr_transcription = AsrTranscription(text=segment.text)

                    yield WhisperOutput(
                        segments=asr_segments,
                        transcription_info=asr_transcription_info,
                        transcription=asr_transcription,
                    )
            except Exception as e:
                raise InferenceException(self.model_name) from e

    async def transcribe_batch(
        self, audio_batch: list[Audio], params: WhisperParams | None = None
    ) -> WhisperBatchOutput:
        """Transcribe the batch of audios with the Whisper model.

        Args:
            audio_batch (list[Audio]): The batch of audios to transcribe.
            params (WhisperParams): The parameters for the whisper model.

        Returns:
            WhisperBatchOutput: The transcription output as a dictionary:

                - segments (list[list[AsrSegment]]): The ASR segments for each audio.
                - transcription_info (list[AsrTranscriptionInfo]): The ASR transcription info for each audio.
                - transcription (list[AsrTranscription]): The ASR transcription for each audio.

        Raises:
            InferenceException: If the inference fails.
        """
        if not params:
            params = WhisperParams()
        segments: list[list[AsrSegment]] = []
        infos: list[AsrTranscriptionInfo] = []
        transcriptions: list[AsrTranscription] = []
        for audio in audio_batch:
            output = await self.transcribe(audio, params)
            segments.append(cast(list[AsrSegment], output["segments"]))
            infos.append(cast(AsrTranscriptionInfo, output["transcription_info"]))
            transcriptions.append(cast(AsrTranscription, output["transcription"]))

        return WhisperBatchOutput(
            segments=segments, transcription_info=infos, transcription=transcriptions
        )

    @exception_handler
    async def transcribe_in_chunks(
        self,
        audio: Audio,
        vad_segments: list[VadSegment] | None = None,
        batch_size: int = 4,
        params: BatchedWhisperParams | None = None,
    ) -> AsyncGenerator[WhisperOutput, None]:
        """Transcribe a single audio by segmenting it into chunks (4x faster) in streaming mode.

        Args:
            audio (Audio): The audio to transcribe.
            vad_segments (list[VadSegment]| None): List of vad segments to guide batching of the audio data.
            batch_size (int): Maximum batch size for the batched inference.
            params (BatchedWhisperParams | None): The parameters for the batched inference with whisper model.

        Yields:
            WhisperOutput: The transcription output as a dictionary:
                segments (list[AsrSegment]): The ASR segments.
                transcription_info (AsrTranscriptionInfo): The ASR transcription info.
                transcription (AsrTranscription): The ASR transcription.

        Raises:
            InferenceException: If the inference fails.
        """
        if not params:
            params = BatchedWhisperParams()
        audio_array = audio.get_numpy()
        if vad_segments:
            vad_input = [seg.to_whisper_dict() for seg in vad_segments]
        if not audio_array.any():
            # For silent audios/no audio tracks, return empty output with language as silence
            yield WhisperOutput(
                segments=[],
                transcription_info=AsrTranscriptionInfo(
                    language="silence", language_confidence=1.0
                ),
                transcription=AsrTranscription(text=""),
            )
        else:
            try:
                segments, info = self.batched_model.transcribe(
                    audio_array,
                    vad_segments=vad_input if vad_segments else None,
                    batch_size=batch_size,
                    **params.model_dump(),
                )
                asr_transcription_info = AsrTranscriptionInfo.from_whisper(info)
                for segment in segments:
                    await asyncio.sleep(0)
                    asr_segments = [AsrSegment.from_whisper(segment)]
                    asr_transcription = AsrTranscription(text=segment.text)
                    yield WhisperOutput(
                        segments=asr_segments,
                        transcription_info=asr_transcription_info,
                        transcription=asr_transcription,
                    )
            except Exception as e:
                raise InferenceException(self.model_name) from e
