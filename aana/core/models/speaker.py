from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.time import TimeInterval

__all__ = [
    "PyannoteSpeakerDiarizationParams",
    "SpeakerDiarizationSegment",
    "SpeakerDiarizationSegments",
]


class PyannoteSpeakerDiarizationParams(BaseModel):
    """A model for the pyannote Speaker Diarization model parameters.

    Attributes:
        min_speakers (int | None): The minimum number of speakers present in the audio.
        max_speakers (int | None): The maximum number of speakers present in the audio.
    """

    min_speakers: int | None = Field(
        default=None,
        description="The minimum number of speakers present in the audio.",
    )

    max_speakers: int | None = Field(
        default=None,
        description="The maximum number of speakers present in the audio.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Parameters for the pyannote speaker diarization model.",
        }
    )


class SpeakerDiarizationSegment(BaseModel):
    """Pydantic schema for Segment from Speaker Diarization model.

    Attributes:
        time_interval (TimeInterval): The start and end time of the segment
        speaker (str): speaker assignment of the model in the format "SPEAKER_XX"
    """

    time_interval: TimeInterval = Field(description="Time interval of the segment")
    speaker: str = Field(description="speaker assignment from the model")

    def to_dict(self) -> dict:
        """Generate dictionary with start, end and speaker keys from SpeakerDiarizationSegment.

        Returns:
            dict: Dictionary with start, end and speaker keys
        """
        return {
            "start": self.time_interval.start,
            "end": self.time_interval.end,
            "speaker": self.speaker,
        }

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Speaker Diarization Segment",
        }
    )


SpeakerDiarizationSegments = Annotated[
    list[SpeakerDiarizationSegment],
    Field(description="List of Speaker Diarization segments", default_factory=list),
]
"""
List of SpeakerDiarizationSegment objects.
"""
