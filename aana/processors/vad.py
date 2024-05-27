from collections.abc import Callable

import numpy as np
import torch
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, Segment, SlidingWindowFeature


class VoiceActivitySegmentation(VoiceActivityDetection):
    """Pipeline wrapper class for performing Voice Activity Segmentation based on the detection from the VAD model."""

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        device: torch.device | None = None,
        fscore: bool = False,
        use_auth_token: str | None = None,
        **inference_kwargs,
    ):
        """Initialize the pipeline with the model name and the optional device.

        Args:
            dict parameters of VoiceActivityDetection class from pyannote:
            segmentation (PipelineModel): Loaded model name.
            device (torch.device | None): Device to perform the segmentation.
            fscore (bool): Flag indicating whether to compute F-score during inference.
            use_auth_token (str | None): Optional authentication token for model access.
            inference_kwargs (dict): Optional additional arguments from VoiceActivityDetection pipeline.
        """
        super().__init__(
            segmentation=segmentation,
            device=device,
            fscore=fscore,
            use_auth_token=use_auth_token,
            **inference_kwargs,
        )

    def apply(self, file: AudioFile, hook: Callable | None = None) -> Annotation:
        """Apply voice activity detection on the audio file.

        Args:
            file (AudioFile): Processed file.
            hook (callable, optional): Hook called after each major step of the pipeline with the following signature: hook("step_name", step_artefact, file=file)

        Returns:
            segmentations (Annotation): Voice activity segmentation.
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


class BinarizeVadScores:
    """Binarize detection scores using hysteresis thresholding, with min-cut operation to ensure no segments are longer than max_duration.

    Reference:
        Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
        RNN-based Voice Activity Detection", InterSpeech 2015.

        Modified by Max Bain to include WhisperX's min-cut operation
        https://arxiv.org/abs/2303.00747

    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: float | None = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float("inf"),
    ):
        """Initializes the parameters for Binarizing the VAD scores.

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
        """
        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def __get_active_regions(self, scores: SlidingWindowFeature) -> Annotation:
        """Extract active regions from VAD scores.

        Args:
            scores (SlidingWindowFeature): Detection scores.

        Returns:
            active (Annotation): Active regions.
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
            for t, y in zip(timestamps[1:], k_scores[1:], strict=False):
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

        return active

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores.

        Args:
            scores (SlidingWindowFeature): Detection scores.

        Returns:
            active (Annotation): Binarized scores.
        """
        active = self.__get_active_regions(scores)
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
