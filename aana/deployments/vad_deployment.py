import hashlib
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypedDict

import torch
from pyannote.audio import Model
from pydantic import BaseModel, Field
from ray import serve
from tqdm import tqdm

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import InferenceException
from aana.models.core.audio import Audio
from aana.models.pydantic.time_interval import TimeInterval
from aana.models.pydantic.vad_output import VadSegment
from aana.models.pydantic.vad_params import VadParams
from aana.utils.audio import BinarizeVadScores, VoiceActivitySegmentation
from aana.utils.test import test_cache


class SegmentX:
    """The vad segment format with optional speaker."""

    def __init__(self, start, end, speaker=None):  # noqa: D107
        self.start = start
        self.end = end
        self.speaker = speaker


class VadOutput(TypedDict):
    """The output of the VAD model.

    Attributes:
        segments (list[VadSegment]): The VAD segments.
    """

    segments: list[VadSegment]


class VadConfig(BaseModel):
    """The configuration for the vad deployment."""

    # TODO: Can also be model ID on HF or a local model registry path
    model: str = Field(
        default="https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin",
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

    sample_rate: int = Field(default=16000)


@serve.deployment
class VadDeployment(BaseDeployment):
    """Deployment to serve VAD models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model from from the default open url.

        The configuration should conform to the VadConfig schema.

        """

        def download_vad_model(model_fp, vad_segmentation_url):
            if model_fp is None:
                model_dir = torch.hub._get_torch_home()
                if not Path(model_dir).exists():
                    Path(model_dir).mkdir(parents=True)
                model_fp = Path(model_dir) / "whisperx-vad-segmentation.bin"

            if Path(model_fp).exists() and not Path(model_fp).is_file():
                raise RuntimeError(f"{model_fp}")  # exists and is not a regular file

            if not Path(model_fp).is_file():
                with ExitStack() as stack:
                    source = stack.enter_context(
                        urllib.request.urlopen(vad_segmentation_url)
                    )
                    output = stack.enter_context(Path.open(model_fp, "wb"))

                    loop = tqdm(
                        total=int(source.info().get("Content-Length")),
                        ncols=80,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    )

                    with loop:
                        while True:
                            buffer = source.read(8192)
                            if not buffer:
                                break

                            output.write(buffer)
                            loop.update(len(buffer))

            model_bytes = Path.open(model_fp, "rb").read()

            if (
                hashlib.sha256(model_bytes).hexdigest()
                != str(vad_segmentation_url).split("/")[-2]
            ):
                checksum_error = "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
                raise RuntimeError(f"{checksum_error}")

            return Path(model_fp)

        config_obj = VadConfig(**config)
        self.hyperparameters = {
            "onset": config_obj.onset,
            "offset": config_obj.offset,
            "min_duration_on": config_obj.min_duration_on,
            "min_duration_off": config_obj.min_duration_off,
        }
        self.sample_rate = config_obj.sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.filepath = download_vad_model(
            None, config_obj.model
        )  # TODO: TO change this:just for now

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
        segments_list = []
        for speech_turn in segments.itersegments():  # .get_timeline():
            # .get_timeline(): #Annotation object supports iteration.
            segments_list.append(  # noqa: PERF401
                SegmentX(
                    max(0.0, speech_turn.start - 0.1), speech_turn.end + 0.1, "UNKNOWN"
                )
            )  # Additional 100ms padding to account for edge errors.

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
            "sample_rate": self.sample_rate,  # TODO:VadConfig?
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
