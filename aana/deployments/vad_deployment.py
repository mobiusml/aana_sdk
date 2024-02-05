from pathlib import Path
from contextlib import ExitStack
from collections.abc import Callable
import urllib.request, hashlib
from tqdm import tqdm
from typing import Callable, TypedDict, Optional, Text, Union, Any

import torch
import subprocess
import numpy as np

# faster whisper here? or load audio?
from pyannote.audio import Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, Segment, SlidingWindowFeature

from aana.models.core.video import Video
from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import InferenceException
from aana.models.pydantic.vad_params import VadParams
from aana.models.pydantic.vad_output import VadSegment
from aana.models.pydantic.time_interval import TimeInterval

from ray import serve
from pydantic import BaseModel, Field


class SegmentX:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


class VadOutput(TypedDict):
    """The output of the VAD model.

    Attributes:
        segments (list[VadSegment]): The VAD segments.
    """

    vad_segments: list[VadSegment]


class VadConfig(BaseModel):
    """The configuration for the whisper deployment from faster-whisper."""

    model_fp: Path | None = Field(default=None, description="The VAD model path.")


class VoiceActivitySegmentation(VoiceActivityDetection):
    """Pipeline for performing Voice Activity Segmentation based on the detection from the vad model.

    Args:
        dict parameters of VoiceActivityDetection class from pyannote:
        segmentation: loaded model.
        device: torch.device to perform the segmentation.

    Returns:
        segmentations: segmented speech regions
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__(
            segmentation=segmentation,
            fscore=fscore,
            use_auth_token=use_auth_token,
            **inference_kwargs,
        )

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection.

        Args:
            file (AudioFile): Processed file.
            hook (callable, optional): Hook called after each major step of the pipeline with the following signature: hook("step_name", step_artefact, file=file)

        Returns:
            speech (Annotation): Speech regions.
        """
        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations


class Binarize:
    """Binarize detection scores using hysteresis thresholding, with min-cut operation to ensure no segments are longer than max_duration.

    Args:
        onset (float, optional):
            Onset threshold. Defaults to 0.5.
        offset (float, optional):
            Offset threshold. Defaults to `onset`.
        min_duration_on (float, optional):
            Remove active regions shorter than that many seconds. Defaults to 0s.
        min_duration_off (float, optional):
            Fill inactive regions shorter than that many seconds. Defaults to 0s.
        pad_onset (float, optional):
            Extend active regions by moving their start time by that many seconds.
            Defaults to 0s.
        pad_offset (float, optional):
            Extend active regions by moving their end time by that many seconds.
            Defaults to 0s.
        max_duration (float):
            The maximum length of an active segment, divides segment at timestamp with lowest score.

    Reference:
        Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
        RNN-based Voice Activity Detection", InterSpeech 2015.

        Modified by Max Bain to include WhisperX's min-cut operation
        https://arxiv.org/abs/2303.00747

    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float("inf"),
    ):
        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores.

        Args:
            scores : SlidingWindowFeature
                Detection scores.

        Returns:
            active : Annotation
                Binarized scores.
        """
        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        # annotation meant to store 'active' regions
        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):
            label = k if scores.labels is None else scores.labels[k]

            # initial state
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            t = start
            for t, y in zip(timestamps[1:], k_scores[1:]):
                # currently active
                if is_active:
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        search_after = len(curr_scores) // 2
                        # divide segment
                        min_score_div_idx = search_after + np.argmin(
                            curr_scores[search_after:]
                        )
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(
                            start - self.pad_onset, min_score_t + self.pad_offset
                        )
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx + 1 :]
                        curr_timestamps = curr_timestamps[min_score_div_idx + 1 :]
                    # switching from active to inactive
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(y)
                    curr_timestamps.append(t)
                # currently inactive
                else:
                    # switching from inactive to active
                    if y > self.onset:
                        start = t
                        is_active = True

            # if active at the end, add final region
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError("This would break current max_duration param")
            active = active.support(collar=self.min_duration_off)

        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


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
                != vad_segmentation_url.split("/")[-2]
            ):
                checksum_error = "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
                raise RuntimeError(f"{checksum_error}")

            return model_fp

        config_obj = VadParams(**config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        filepath = download_vad_model(
            config_obj.model_fp, config_obj.vad_segmentation_url
        )

        self.vad_model = Model.from_pretrained(filepath, use_auth_token=None)
        hyperparameters = {
            "onset": config_obj.onset,
            "offset": config_obj.offset,
            "min_duration_on": config_obj.min_duration_on,
            "min_duration_off": config_obj.min_duration_off,
        }

        # vad_pipeline
        self.vad_pipeline = VoiceActivitySegmentation(
            segmentation=self.vad_model, device=torch.device(self.device)
        )
        self.vad_pipeline.instantiate(hyperparameters)

    # OPTIONAL: audio loader function if asr model is not used in deployment.
    async def load_audio(self, file: str, sr: int = 16000):
        """Open an audio file and read as mono waveform, resampling as necessary.

        Args:
            file (str): The audio file to open.
            sr (int): The sample rate to resample the audio if necessary.

        Returns:
            A NumPy array containing the audio waveform, in float32 dtype.

        Raises:
            RuntimeError: if ffmpeg fails to convert and load the audio.
        """
        try:
            # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI to be installed.
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i",
                file,
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-",
            ]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    async def merge_chunks(
        self, segments: list[dict], params: dict[str, Any]
    ) -> list[VadSegment]:
        """Merge operation to chunk_size segments.

        Args:
            segments (List(dict)): The list of segments to merge/split.
            params (dict[str, Any]): The parameters for the vad model as a dictionary

        Returns:
            merged_segments list(VadSegment): The list of segments.

        """
        chunk_size = params["chunk_size"]
        onset = params["merge_onset"]
        offset = params["merge_offset"]

        curr_end = 0
        merged_segments: list[VadSegment] = []
        seg_idxs: list[tuple[float, float]] = []
        speaker_idxs = [str]

        if not chunk_size > 0:
            chunk_error = "Expected positive value."
            raise ValueError(f"{chunk_error}")

        binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
        segments = binarize(segments)
        segments_list = []
        for speech_turn in segments.get_timeline():
            segments_list.append(  # noqa: PERF401
                SegmentX(
                    max(0.0, speech_turn.start - 0.1), speech_turn.end + 0.1, "UNKNOWN"
                )
            )  # To account for edge errors additional 100ms padding

        if len(segments_list) == 0:
            # No active speech found in audio.
            return []

        # Make sure the starting point is the start of the segment.
        curr_start = segments_list[0].start

        for seg in segments_list:
            if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
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

    async def vad_inference(self, media: Video) -> list[dict]:
        """Perform voice activity detection on the media with the vad model.

        Args:
            media (Video): The media to perform vad.

        Returns:
            vad_segments (list[dict]): The list of vad segments.

        Raises:
            InferenceException: If the vad inference fails.
        """
        media_path: str = str(media.path)

        # load audio data
        audio_array = (
            np.frombuffer(open(media_path, "rb").read(), dtype=np.int16)
            .flatten()
            .astype(np.float32)
            / 32768.0
        )
        # OPTIONAL: Use load_audio function if audio extraction is not performed.
        # audio_array = self.load_audio(media_path)

        try:
            vad_segments = self.vad_pipeline(
                {
                    "waveform": torch.from_numpy(audio_array).unsqueeze(0),
                    "sample_rate": 16000,
                }
            )
        except Exception as e:
            raise InferenceException(self.vad_model) from e

        return vad_segments

    async def asr_preprocess_vad(
        self, media: Video, params: VadParams | None = None
    ) -> VadOutput:
        """Perform vad inference and further processing for batched asr inference.

        Args:
            media (Video): The media to perform vad.
            params (VadParams): The parameters for the vad model.

        Returns:
            vad_segments (VadOutput): Output vad segments to the asr model.
        """
        if not params:
            params = VadParams()

        intermediate_vad_segments = await self.vad_inference(media)
        asr_vad_segments = await self.merge_chunks(
            intermediate_vad_segments, params.__dict__
        )
        return VadOutput(vad_segments=asr_vad_segments)
