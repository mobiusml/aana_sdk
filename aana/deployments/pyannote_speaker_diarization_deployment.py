from typing import Any, TypedDict

import torch
from huggingface_hub.utils import GatedRepoError
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pydantic import BaseModel, ConfigDict, Field
from ray import serve

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_protected_fields
from aana.core.models.speaker import (
    PyannoteSpeakerDiarizationParams,
    SpeakerDiarizationSegment,
)
from aana.core.models.time import TimeInterval
from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.runtime import InferenceException
from aana.processors.speaker import combine_homogeneous_speaker_diarization_segments


class SpeakerDiarizationOutput(TypedDict):
    """The output of the Speaker Diarization model.

    Attributes:
        segments (list[SpeakerDiarizationSegment]): The Speaker Diarization segments.
    """

    segments: list[SpeakerDiarizationSegment]


class PyannoteSpeakerDiarizationConfig(BaseModel):
    """The configuration for the Pyannote Speaker Diarization deployment.

    Attributes:
        model_id (str): name of the speaker diarization pipeline.
        sample_rate (int): The sample rate of the audio. Defaults to 16000.
    """

    model_id: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="The Speaker Diarization model ID.",
    )

    sample_rate: int = Field(default=16000, description="Sample rate of the audio.")

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


@serve.deployment
class PyannoteSpeakerDiarizationDeployment(BaseDeployment):
    """Deployment to serve Pyannote Speaker Diarization (SD) models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and instantiate speaker diarization pipeline.

        The configuration should conform to the PyannoteSpeakerDiarizationConfig schema.

        """
        config_obj = PyannoteSpeakerDiarizationConfig(**config)

        self.sample_rate = config_obj.sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = config_obj.model_id

        # for consistency across multiple runs
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        try:
            # load model using pyannote Pipeline
            self.diarize_model = Pipeline.from_pretrained(self.model_id)

            if self.diarize_model:
                self.diarize_model.to(torch.device(self.device))

        except Exception as e:
            raise GatedRepoError(
                message=f"This repository is private and requires a token to accept user conditions and access models in {self.model_id} pipeline."
            ) from e

    async def __inference(
        self, audio: Audio, params: PyannoteSpeakerDiarizationParams
    ) -> Annotation:
        """Perform inference on the Audio with the Pyannote Speaker Diarization model.

        Args:
            audio (Audio): The audio to perform Speaker Diarization.
            params (PyannoteSpeakerDiarizationParams): Parameters for the pyannote speaker diarization model.

        Returns:
            speaker_segments (Annotation): The list of speaker diarized segments.

        Raises:
            InferenceException: If the Speaker Diarization inference fails.
        """
        audio_array = audio.get_numpy()
        speaker_diarization_input = {
            "waveform": torch.from_numpy(audio_array).unsqueeze(0),
            "sample_rate": self.sample_rate,
        }

        try:
            speaker_segments = self.diarize_model(
                speaker_diarization_input,
                min_speakers=params.min_speakers,
                max_speakers=params.max_speakers,
            )

        except Exception as e:
            raise InferenceException(self.model_id) from e

        return speaker_segments

    async def diarize(
        self, audio: Audio, params: PyannoteSpeakerDiarizationParams | None = None
    ) -> SpeakerDiarizationOutput:
        """Perform Speaker Diarization inference to get speaker segments.

        Args:
            audio (Audio): The audio to perform Speaker Diarization.
            params (PyannoteSpeakerDiarizationParams): Parameters for the speaker diarization model.

        Returns:
            SpeakerDiarizationOutput: Output speaker segments from the Speaker Diarization pipeline.

        """
        if not params:
            params = PyannoteSpeakerDiarizationParams()

        speaker_segments = await self.__inference(audio, params)
        speaker_diarization_segments = []
        for speech_turn, _, speaker in speaker_segments.itertracks(yield_label=True):
            speaker_diarization_segments.append(
                SpeakerDiarizationSegment(
                    time_interval=TimeInterval(
                        start=speech_turn.start, end=speech_turn.end
                    ),
                    speaker=speaker,
                )
            )

        # Combine homogeneous speaker segments.
        processed_speaker_diarization_segments = (
            combine_homogeneous_speaker_diarization_segments(
                speaker_diarization_segments
            )
        )
        return SpeakerDiarizationOutput(segments=processed_speaker_diarization_segments)
