from enum import Enum
from typing import Any, Dict
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field
from ray import serve
import torch

from aana.deployments.base_deployment import BaseDeployment
from aana.models.core.video import Video
from aana.models.pydantic.whisper_params import WhisperParams


class WhisperComputeType(str, Enum):
    """
    The data type used by whisper models.

    Available types:
        - INT8 (int8)
        - INT8_FLOAT32 (int8_float32)
        - INT8_FLOAT16 (int8_float16)
        - INT8_BFLOAT16 (int8_bfloat16)
        - INT16 (int16)
        - FLOAT16 (float16)
        - BFLOAT16 (bfloat16)
        - FLOAT32 (float32)
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
    """
    The whisper model.

    Available models:
        - TINY (tiny)
        - TINY_EN (tiny.en)
        - BASE (base)
        - BASE_EN (base.en)
        - SMALL (small)
        - SMALL_EN (small.en)
        - MEDIUM (medium)
        - MEDIUM_EN (medium.en)
        - LARGE_V1 (large-v1)
        - LARGE_V2 (large-v2)
        - LARGE (large)
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
    """
    The configuration for the whisper deployment from faster-whisper.

    """

    model_size: WhisperModelSize = Field(
        default=WhisperModelSize.BASE, description="The whisper model size."
    )
    compute_type: WhisperComputeType = Field(
        default=WhisperComputeType.FLOAT16, description="The compute type."
    )


@serve.deployment
class WhisperDeployment(BaseDeployment):
    """
    Deployment to serve Whisper models from faster-whisper.
    """

    async def apply_config(self, config: Dict[str, Any]):
        """
        Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the HFBlip2Config schema.
        """

        config_obj = WhisperConfig(**config)
        self.model_size = config_obj.model_size
        self.compute_type = config_obj.compute_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    # TODO: add audio support
    async def transcribe(self, media: Video, params: WhisperParams):
        """
        Transcribe the media with the whisper model.

        Args:
            media (Video): The media to transcribe.
            params (WhisperParams): The parameters for the whisper model.
        """

        media_path: str = str(media.path)
        segments, info = self.model.transcribe(media_path, **params.dict())
