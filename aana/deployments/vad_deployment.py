from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from pyannote.audio import Model
from pydantic import BaseModel, Field
from ray import serve

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import InferenceException
from aana.models.core.audio import Audio
from aana.models.pydantic.time_interval import TimeInterval
from aana.models.pydantic.vad_output import VadSegment
from aana.models.pydantic.vad_params import VadParams
from aana.utils.audio import BinarizeVadScores, VoiceActivitySegmentation
from aana.utils.general import download_model
from aana.utils.test import test_cache


@dataclass
class SegmentX:
    """The vad segment format with optional speaker."""

    start: float
    end: float
    speaker: str = ""


class VadOutput(TypedDict):
    """The output of the VAD model.

    Attributes:
        segments (list[VadSegment]): The VAD segments.
    """

    segments: list[VadSegment]


class VadConfig(BaseModel):
    """The configuration for the vad deployment.

    Attributes:
        model (str): Model file url
        onset (float): Threshold for voice activity
        offset (float): Thereshold for silence
        min_duration_on (float): Minimum voiced duration
        min_duration_off (float): Minimum silence duration
        sample_rate (int): The sample rate of the audio

    """

    model: str = Field(
        description="The VAD model url.",
    )

    # default_parameters for the model on initialization:
    onset: float = Field(
        default=0.500,
        ge=0.0,
        description="Threshold to decide a positive voice activity.",
    )

    offset: float = Field(
        default=0.363,
        ge=0.0,
        description="Threshold to consider as a silence region.",
    )

    min_duration_on: float = Field(
        default=0.1, ge=0.0, description="Minimum voiced duration."
    )

    min_duration_off: float = Field(
        default=0.1, ge=0.0, description="Minimum duration to consider as silence."
    )

    sample_rate: int = Field(default=16000, description="Sample rate of the audio.")


@serve.deployment
class VadDeployment(BaseDeployment):
    """Deployment to serve VAD models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and instantiate vad_pipeline.

        The configuration should conform to the VadConfig schema.

        """
        config_obj = VadConfig(**config)
        self.hyperparameters = {
            "onset": config_obj.onset,
            "offset": config_obj.offset,
            "min_duration_on": config_obj.min_duration_on,
            "min_duration_off": config_obj.min_duration_off,
        }
        self.sample_rate = config_obj.sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # model hash for vad_model
        model_hash = str(config_obj.model).split("/")[-2]
        self.filepath = download_model(config_obj.model, model_hash)

        self.vad_model = Model.from_pretrained(self.filepath, use_auth_token=None)

        # vad_pipeline
        self.vad_pipeline = VoiceActivitySegmentation(
            segmentation=self.vad_model, device=torch.device(self.device)
        )
        self.vad_pipeline.instantiate(self.hyperparameters)

    async def __merge(
        self, segments: list[dict], params: VadParams
    ) -> list[VadSegment]:
        """Merge operation to combine vad segments into chunk_size segments.

        Args:
            segments (list[dict]): The list of segments to merge/split.
            params (VadParams): The parameters for the vad model.  #dict[str, Any]

        Returns:
            merged_segments list(VadSegment): The list of segments.

        """
        curr_end = 0
        merged_segments: list[VadSegment] = []
        seg_idxs: list[tuple[float, float]] = []
        speaker_idxs = [str]

        if not params.chunk_size > 0:
            chunk_error = "Expected positive value."
            raise ValueError(f"{chunk_error}")

        binarize = BinarizeVadScores(
            max_duration=params.chunk_size,
            onset=self.hyperparameters["onset"],
            offset=self.hyperparameters["offset"],
        )
        segments = binarize(segments)

        segments_list = [
            SegmentX(
                max(0.0, speech_turn.start - 0.1), speech_turn.end + 0.1, "UNKNOWN"
            )
            for speech_turn in segments.itersegments()
        ]
        if len(segments_list) == 0:
            # No active speech found in audio.
            return []

        # To make sure the starting point is the start of the segment.
        curr_start = segments_list[0].start

        for seg in segments_list:
            if seg.end - curr_start > params.chunk_size and curr_end - curr_start > 0:
                merged_segments.append(
                    VadSegment(
                        time_interval=TimeInterval(start=curr_start, end=curr_end),
                        segments=seg_idxs,
                    )
                )
                curr_start = seg.start
                seg_idxs = []
                speaker_idxs = []
            curr_end = seg.end
            seg_idxs.append((seg.start, seg.end))
            speaker_idxs.append(seg.speaker)

        merged_segments.append(
            VadSegment(
                time_interval=TimeInterval(start=curr_start, end=curr_end),
                segments=seg_idxs,
            )
        )
        return merged_segments

    async def __inference(self, audio: Audio) -> list[dict]:
        """Perform voice activity detection inference on the Audio with the vad model.

        Args:
            audio (Audio): The audio to perform vad.

        Returns:
            vad_segments (list[dict]): The list of vad segments.

        Raises:
            InferenceException: If the vad inference fails.
        """
        audio_array = audio.get_numpy()
        vad_input = {
            "waveform": torch.from_numpy(audio_array).unsqueeze(0),
            "sample_rate": self.sample_rate,
        }

        try:
            vad_segments = self.vad_pipeline(vad_input)
        except Exception as e:
            raise InferenceException(self.filepath.name) from e

        return vad_segments

    @test_cache
    async def asr_preprocess_vad(
        self, audio: Audio, params: VadParams | None = None
    ) -> VadOutput:
        """Perform vad inference to get vad segments and further processing (split and merge) to get segments for batched asr inference.

        Args:
            audio (Audio): The audio to perform vad.
            params (VadParams): The parameters for the vad model.

        Returns:
            egments (VadOutput): Output vad segments to the asr model.
        """
        if not params:
            params = VadParams()

        intermediate_vad_segments = await self.__inference(audio)
        asr_vad_segments = await self.__merge(
            intermediate_vad_segments,
            params,
        )
        return VadOutput(segments=asr_vad_segments)