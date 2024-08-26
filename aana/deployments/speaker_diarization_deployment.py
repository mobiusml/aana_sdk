from typing import Any, TypedDict

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pydantic import BaseModel, ConfigDict, Field
from ray import serve

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_protected_fields
from aana.core.models.time import TimeInterval
from aana.core.models.vad import SDSegment
from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.runtime import InferenceException


class SDOutput(TypedDict):
    """The output of the SD model.

    Attributes:
        segments (list[SDSegment]): The SD segments.
    """

    segments: list[SDSegment]


class SDConfig(BaseModel):
    """The configuration for the Speaker Diarization deployment.

    Attributes:
        model_name (str): name of the speaker diarization pipeline.
        sample_rate (int): The sample rate of the audio. Defaults to 16000.
    """

    model_name: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="The SD model name.",
    )

    sample_rate: int = Field(default=16000, description="Sample rate of the audio.")
    # TODO: add min speakers, max speakers, see default values@ https://github.com/pyannote/pyannote-audio

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


@serve.deployment
class SpeakerDiarizationDeployment(BaseDeployment):
    """Deployment to serve Speaker Diarization (SD) models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and instantiate speaker diarization pipeline.

        The configuration should conform to the SDConfig schema.

        """
        config_obj = SDConfig(**config)

        self.sample_rate = config_obj.sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config_obj.model_name

        # for consistency across multiple runs
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # load model using pyannote Pipeline
        # TODO: Discuss feasibility/usefulness of having the model locally

        self.diarize_model = Pipeline.from_pretrained(self.model_name)
        self.diarize_model.to(torch.device(self.device))

    async def __inference(self, audio: Audio) -> Annotation:
        """Perform speaker diarization inference on the Audio with the SD model.

        Args:
            audio (Audio): The audio to perform SD.

        Returns:
            speaker_segments (Annotation): The list of speaker diarized segments.

        Raises:
            InferenceException: If the vad inference fails.
        """
        audio_array = audio.get_numpy()
        sd_input = {
            "waveform": torch.from_numpy(audio_array).unsqueeze(0),
            "sample_rate": self.sample_rate,
        }

        try:
            speaker_segments = self.diarize_model(sd_input)

        except Exception as e:
            raise InferenceException(self.model_name) from e

        return speaker_segments

    async def diarize_segments(self, audio: Audio) -> SDOutput:
        """Perform SD inference to get speaker segments.

        Args:
            audio (Audio): The audio to perform SD.

        Returns:
            SDOutput: Output speaker segments to the SD model.

        Raises:
            InferenceException: If the SD inference fails.

        TODO: Add user defined params (SDParams).
        TODO: add further processing with VAD if needed.

        """
        speaker_segments = await self.__inference(audio)
        sd_segments = []
        for speech_turn, _, speaker in speaker_segments.itertracks(yield_label=True):
            sd_segments.append(
                SDSegment(
                    time_interval=TimeInterval(
                        start=speech_turn.start, end=speech_turn.end
                    ),
                    speaker=speaker,
                )
            )

        if not sd_segments:
            print("no speaker segments detected.")

        return SDOutput(segments=sd_segments)
