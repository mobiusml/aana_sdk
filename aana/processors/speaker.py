from collections import defaultdict

from aana.core.models.asr import (
    AsrSegment,
    AsrWord,
)
from aana.core.models.speaker import SpeakerDiarizationSegment
from aana.core.models.time import TimeInterval

# Define sentence ending punctuations to split segments at sentence endings:
sentence_ending_punctuations = ".?!"


class PostProcessingForDiarizedAsr:
    """Class to handle post-processing for diarized ASR output by combining diarization and transcription segments.

    The post-processing involves assigning speaker labels to transcription segments and words, aligning speakers
    with punctuation, optionally merging homogeneous speaker segments, and reassigning confidence information to the segments.

    Attributes:
        diarized_segments (list[SpeakerDiarizationSegment]): Contains speaker diarization segments.
        transcription_segments (list[AsrSegment]): Transcription segments.
        merge (bool): Whether to merge the same speaker segments in the final output.
    """

    @classmethod
    def process(
        cls,
        diarized_segments: list[SpeakerDiarizationSegment],
        transcription_segments: list[AsrSegment],
        merge: bool = False,
    ) -> list[AsrSegment]:
        """Executes the post-processing pipeline that combines diarization and transcription segments.

        This method performs the following steps:
        1. Assign speaker labels to each segment and word in the transcription based on the diarization output.
        2. Align speakers with punctuation.
        3. Create new transcription segments by combining the speaker-labeled words.
        4. Optionally, merge consecutive speaker segments.
        5. Add confidence and no_speech_confidence to the new segments.

        Args:
            diarized_segments (list[SpeakerDiarizationSegment]): Contains speaker diarization segments.
            transcription_segments (list[AsrSegment]): Transcription segments.
            merge (bool): If True, merges consecutive speaker segments in the final output. Defaults to False.

        Returns:
            list[AsrSegment]: Updated transcription segments with speaker information per segment and word.
        """
        # intro: validation checks
        if not transcription_segments or not diarized_segments:
            return transcription_segments

        # Check if inputs are valid:
        for segment in transcription_segments:
            if segment.text and not segment.words:
                raise ValueError("Word-level timestamps are required for diarized ASR.")  # noqa: TRY003

        # 1. Assign speaker labels to each segment and word
        speaker_labelled_transcription = cls._assign_word_speakers(
            diarized_segments, transcription_segments
        )

        # 2. Align speakers with punctuation
        word_speaker_mapping = cls._align_with_punctuation(
            speaker_labelled_transcription
        )

        # 3. Create new transcription segments with speaker information
        segments = cls._create_speaker_segments(word_speaker_mapping)

        # 4. Assign confidence variables to the new segments
        segments = cls._add_segment_variables(segments, transcription_segments)

        # Optional: Merge consecutive speaker segments
        if merge:
            segments = cls._combine_homeogeneous_speaker_asr_segments(segments)

        return segments

    @classmethod
    def _create_speaker_segments(
        cls, word_list: list[AsrWord], max_words_per_segment: int = 50
    ) -> list[AsrSegment]:
        """Creates speaker segments from a list of words with speaker annotations.

        Args:
            word_list (list[AsrWord]): A list of words with associated speaker and timing details.
            max_words_per_segment (int): The maximum number of words per segment. If the segment exceeds this,
                                        it will be split at previous or next sentence-ending punctuation.

        Returns:
            list[AsrSegment]: A list of segments where each segment groups words spoken by the same speaker.
        """
        if not word_list:
            return []

        current_speaker = None
        current_segment = None
        final_segments: list[AsrSegment] = []
        word_count = 0

        for word_info in word_list:
            speaker = word_info.speaker or current_speaker

            # Handle speaker change
            if speaker != current_speaker:
                if current_segment:
                    final_segments.append(current_segment)
                current_segment = cls._create_new_segment(word_info, speaker)
                current_speaker = speaker
                word_count = 1
            else:
                if current_segment:
                    # Handle word count and punctuation splitting
                    (
                        current_segment,
                        word_count,
                    ) = cls._split_segment_on_length_punctuation(
                        current_segment,
                        word_info,
                        word_count,
                        max_words_per_segment,
                        final_segments,
                    )

        # Add the final segment if it exists
        if current_segment and current_segment.words:
            final_segments.append(current_segment)

        return final_segments

    @classmethod
    def _align_with_punctuation(
        cls, transcription_segments: list[AsrSegment], max_words_in_sentence: int = 50
    ) -> list[AsrWord]:
        """Aligns speaker labels with sentence boundaries defined by punctuation.

        Args:
            transcription_segments (list[AsrSegment]): transcription segments with speaker information.
            max_words_in_sentence (int): Maximum number of words allowed in a sentence.

        Returns:
            word_speaker_mapping: (list[AsrWord]): Realigned word-speaker mappings.
        """
        new_segments = [segment.words for segment in transcription_segments]
        word_speaker_mapping = [word for segment in new_segments for word in segment]
        words_list = [item.word for item in word_speaker_mapping]
        speaker_list = [item.speaker for item in word_speaker_mapping]

        # Fill missing speaker labels by finding the nearest speaker
        for i, item in enumerate(word_speaker_mapping):
            if item.speaker is None:
                item.speaker = cls._find_nearest_speaker(
                    i, word_speaker_mapping, reverse=i > 0
                )
                speaker_list[i] = item.speaker

        # Align speakers with sentence boundaries
        k = 0
        while k < len(word_speaker_mapping):
            if (
                k < len(word_speaker_mapping) - 1
                and speaker_list[k] != speaker_list[k + 1]
                and words_list[k][-1] not in sentence_ending_punctuations
            ):
                left_idx = cls._get_first_word_idx_of_sentence(
                    k, words_list, speaker_list, max_words_in_sentence
                )
                right_idx = cls._get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - (k - left_idx)
                )

                if min(left_idx, right_idx) == -1:
                    k += 1
                    continue

                spk_labels = speaker_list[left_idx : right_idx + 1]
                mod_speaker = max(set(spk_labels), key=spk_labels.count)

                if spk_labels.count(mod_speaker) >= len(spk_labels) // 2:
                    speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                        right_idx - left_idx + 1
                    )
                    k = right_idx

            k += 1

        # Realign the speaker labels in the original word_speaker_mapping
        for i, item in enumerate(word_speaker_mapping):
            item.speaker = speaker_list[i]

        return word_speaker_mapping

    @classmethod
    def _add_segment_variables(
        cls, segments: list[AsrSegment], transcription_segments: list[AsrSegment]
    ) -> list[AsrSegment]:
        """Adds confidence and no_speech_confidence variables to each segment.

        Args:
            segments (list[AsrSegment]): A list of segments to which the confidence values will be added.
            transcription_segments (list[AsrSegment]): The original transcription segments to help determine segment confidence.

        Returns:
            list[AsrSegment]: Segments with confidence and no_speech_confidence added.
        """
        for segment in segments:
            confidence, no_speech_confidence = cls._determine_major_segment_confidence(
                segment, transcription_segments
            )
            segment.confidence = confidence
            segment.no_speech_confidence = no_speech_confidence
        return segments

    @classmethod
    def _split_segment_on_length_punctuation(
        cls,
        current_segment: AsrSegment,
        word_info: AsrWord,
        word_count: int,
        max_words_per_segment: int,
        final_segments: list[AsrSegment],
    ) -> tuple[AsrSegment, int]:
        """Splits segments based on length and sentence-ending punctuation.

        Args:
            current_segment (AsrSegment): The current speaker segment being processed.
            word_info (AsrWord): Word information containing timing and text.
            word_count (int): The current word count in the segment.
            max_words_per_segment (int): Maximum number of words allowed in a segment before splitting.
            final_segments (list[AsrSegment]): List of segments to which the completed segment will be added.

        Returns:
            Tuple[AsrSegment, int]: The updated segment and word count.
        """
        # Check if word count exceeds the limit and if punctuation exists to split
        if word_count >= max_words_per_segment and any(
            p in word_info.word for p in sentence_ending_punctuations
        ):
            # update current segment and then append it
            current_segment.time_interval.end = word_info.time_interval.end
            current_segment.text += f"{word_info.word}"
            current_segment.words.append(word_info)
            final_segments.append(current_segment)
            current_segment = cls._create_new_segment(
                word_info, current_segment.speaker, is_empty=True
            )
            word_count = 0  # Reset word count

        else:
            # Append word to the current segment
            current_segment.time_interval.end = word_info.time_interval.end
            current_segment.text += f"{word_info.word}"
            current_segment.words.append(word_info)
            word_count += 1

        return current_segment, word_count

    @staticmethod
    def _assign_word_speakers(
        diarized_segments: list[SpeakerDiarizationSegment],
        transcription_segments: list[AsrSegment],
        fill_nearest: bool = False,
    ) -> list[AsrSegment]:
        """Assigns speaker labels to each segment and word in the transcription based on diarized output.

        Args:
            diarized_segments (list[SpeakerDiarizationSegment]): Contains speaker diarization segments.
            transcription_segments (list[AsrSegment]): Transcription data with segments, text, and language_info.
            fill_nearest (bool): If True, assigns the closest speaker even if there's no positive overlap. Default is False.

        Returns:
            transcription_segments (list[AsrSegment]): Transcription updated in-place with the assigned speaker labels.
        """

        def get_speaker_for_interval(
            sd_segments: list[SpeakerDiarizationSegment],
            start_time: float,
            end_time: float,
            fill_nearest: bool,
        ) -> str | None:
            """Determines the speaker for a given time interval based on diarized segments.

            Args:
                sd_segments (list[SpeakerDiarizationSegment]): List of speaker diarization segments.
                start_time (float): Start time of the interval.
                end_time (float): End time of the interval.
                fill_nearest (bool): If True, selects the closest speaker even with no overlap.

            Returns:
                str | None: The identified speaker label, or None if no speaker is found.
            """
            overlaps = []

            for sd_segment in sd_segments:
                interval_start = sd_segment.time_interval.start
                interval_end = sd_segment.time_interval.end

                # Calculate overlap duration
                overlap_start = max(start_time, interval_start)
                overlap_end = min(end_time, interval_end)
                overlap_duration = max(0.0, overlap_end - overlap_start)

                if overlap_duration > 0 or fill_nearest:
                    distance = float(
                        min(
                            abs(start_time - interval_end),
                            abs(end_time - interval_start),
                        )
                    )

                    overlaps.append(
                        {
                            "speaker": sd_segment.speaker,
                            "overlap_duration": overlap_duration,
                            "distance": distance,
                        }
                    )

            if not overlaps:
                return None
            else:
                # Select the speaker with the maximum overlap duration (or minimal distance)
                best_match = max(
                    overlaps,
                    key=lambda x: (x["overlap_duration"], -x["distance"])
                    if not fill_nearest
                    else (-x["distance"]),
                )
                return best_match["speaker"]

        for segment in transcription_segments:
            # Assign speaker to segment
            segment.speaker = get_speaker_for_interval(
                diarized_segments,
                segment.time_interval.start,
                segment.time_interval.end,
                fill_nearest,
            )

            # Assign speakers to words within the segment
            if segment.words:
                for word in segment.words:
                    word.speaker = get_speaker_for_interval(
                        diarized_segments,
                        word.time_interval.start,
                        word.time_interval.end,
                        fill_nearest,
                    )

        return transcription_segments

    @staticmethod
    def _get_first_word_idx_of_sentence(
        word_idx: int,
        word_list: list[str],
        speaker_list: list[str | None],
        max_words: int,
    ) -> int:
        """Get the index of the first word of the sentence in the given range."""
        left_idx = word_idx
        while (
            left_idx > 0
            and word_idx - left_idx < max_words
            and speaker_list[left_idx - 1] == speaker_list[left_idx]
            and word_list[left_idx - 1][-1] not in sentence_ending_punctuations
        ):
            left_idx -= 1

        return (
            left_idx
            if left_idx == 0
            or word_list[left_idx - 1][-1] in sentence_ending_punctuations
            else -1
        )

    @staticmethod
    def _get_last_word_idx_of_sentence(
        word_idx: int, word_list: list[str], max_words: int
    ) -> int:
        """Get the index of the last word of the sentence in the given range."""
        right_idx = word_idx
        while (
            right_idx < len(word_list)
            and right_idx - word_idx < max_words
            and word_list[right_idx][-1] not in sentence_ending_punctuations
        ):
            right_idx += 1

        return (
            right_idx
            if right_idx == len(word_list) - 1
            or word_list[right_idx][-1] in sentence_ending_punctuations
            else -1
        )

    @staticmethod
    def _find_nearest_speaker(
        index: int, word_speaker_mapping: list[AsrWord], reverse: bool = False
    ) -> str | None:
        """Find the nearest speaker label in the word_speaker_mapping either forward or backward.

        Args:
            index (int): The index to start searching from.
            word_speaker_mapping (list[AsrWord]): List of word-speaker mappings.
            reverse (bool): Search backwards if True; forwards if False. Default is False.

        Returns:
            str | None: The nearest speaker found or None if not found.
        """
        step = -1 if reverse else 1
        for i in range(index, len(word_speaker_mapping) if not reverse else -1, step):
            if word_speaker_mapping[i].speaker:
                return word_speaker_mapping[i].speaker
        return None

    @staticmethod
    def _create_new_segment(
        word_info: AsrWord, speaker: str | None, is_empty: bool = False
    ) -> AsrSegment:
        """Creates a new segment based on word information.

        Args:
            word_info (AsrWord): The word information containing text, timing, etc.
            speaker (str | None): The speaker associated with this word.
            is_empty (bool): If True, creates an empty segment (for punctuation-only segments).

        Returns:
            AsrSegment: A new segment with the provided word information and speaker details.
        """
        return AsrSegment(
            time_interval=TimeInterval(
                start=word_info.time_interval.start
                if not is_empty
                else word_info.time_interval.end,
                end=word_info.time_interval.end,
            ),
            text=word_info.word if not is_empty else "",
            speaker=speaker,
            words=[word_info] if not is_empty else [],
            confidence=None,
            no_speech_confidence=None,
        )

    @staticmethod
    def _determine_major_segment_confidence(
        segment: AsrSegment, transcription_segments: list[AsrSegment]
    ) -> tuple[float | None, float | None]:
        """Determines the confidence and no_speech_confidence based on the major segment (which contributes the most time or words).

        Args:
            segment (AsrSegment): New ASR segment.
            transcription_segments (list[AsrSegment]): Original transcription segments with confidence.

        Returns:
            tuple[Optional[float], Optional[float]]: Confidence and no_speech_confidence from the major segment.
        """

        def find_closest_segment(
            word_start: float, word_end: float
        ) -> AsrSegment | None:
            """Finds the closest segment in the transcription for the given word start and end times."""
            closest_segment = min(
                transcription_segments,
                key=lambda segment: abs(segment.time_interval.start - word_start)
                + abs(segment.time_interval.end - word_end),
                default=None,
            )
            return closest_segment

        def update_segment_contribution(
            contributions: dict, segment: AsrSegment, word_duration: float
        ) -> None:
            """Updates the contribution data for the given segment."""
            segment_id = id(segment)
            if segment_id not in contributions:
                contributions[segment_id] = {
                    "segment": segment,
                    "contribution_time": 0.0,
                    "word_count": 0,
                }
            contributions[segment_id]["contribution_time"] += word_duration
            contributions[segment_id]["word_count"] += 1

        segment_contributions: defaultdict = defaultdict(
            lambda: {"segment": None, "contribution_time": 0.0, "word_count": 0}
        )

        for word in segment.words:
            word_start, word_end = word.time_interval.start, word.time_interval.end
            word_duration = word_end - word_start

            closest_segment = find_closest_segment(word_start, word_end)

            if closest_segment:
                update_segment_contribution(
                    segment_contributions, closest_segment, word_duration
                )

        if not segment_contributions:
            return None, None

        # Determine the segment with the highest word count or contribution time
        major_segment_data = max(
            segment_contributions.values(),
            key=lambda data: data[
                "word_count"
            ],  # Change this to 'contribution_time' if needed
        )

        major_segment = major_segment_data["segment"]
        return major_segment.confidence, major_segment.no_speech_confidence

    @staticmethod
    def _combine_homeogeneous_speaker_asr_segments(
        segments: list[AsrSegment],
    ) -> list[AsrSegment]:
        """Merges consecutive segments that have the same speaker into a single segment.

        Args:
            segments (list[AsrSegment]): The initial list of segments.

        Returns:
            merged_segments (list[AsrSegment]): A new list of merged segments.
        """
        if not segments:
            return []

        merged_segments: list[AsrSegment] = []
        mapping: defaultdict[str, str] = defaultdict(str)

        current_segment = segments[0]
        speaker_counter = 0

        for next_segment in segments[1:]:
            if next_segment.speaker == current_segment.speaker:
                # Merge segments
                current_segment.time_interval.end = next_segment.time_interval.end
                current_segment.text += f" {next_segment.text}"
                current_segment.words.extend(next_segment.words)
            else:
                # Assign unique speaker labels and finalize the current segment
                if current_segment.speaker:
                    current_segment.speaker = mapping.setdefault(
                        current_segment.speaker, f"SPEAKER_{speaker_counter:02d}"
                    )
                    for word in current_segment.words:
                        word.speaker = current_segment.speaker
                    merged_segments.append(current_segment)
                    current_segment = next_segment
                    speaker_counter += 1

        # Handle the last segment
        if current_segment.speaker:
            current_segment.speaker = mapping.setdefault(
                current_segment.speaker, f"SPEAKER_{speaker_counter:02d}"
            )
            for word in current_segment.words:
                word.speaker = current_segment.speaker
            merged_segments.append(current_segment)

        return merged_segments


def combine_homogeneous_speaker_diarization_segments(
    diarized_segments: list[SpeakerDiarizationSegment],
) -> list[SpeakerDiarizationSegment]:
    """Combines same speaker segments in a diarization output.

    Speaker diarization model occationally produce overlapping chunks or segments belonging to the same speaker.
    This method combines segments with the same speaker into homogeneous speaker segments, ensuring no overlapping times.

    Args:
        diarized_segments (list(SpeakerDiarizationSegment)): Input with segments that may have overlapping times.

    Returns:
        list(SpeakerDiarizationSegment): Output with combined homogeneous speaker segments.
    """
    combined_segments: list = []
    current_speaker = None
    current_segment = None

    for segment in sorted(diarized_segments, key=lambda x: x.time_interval.start):
        speaker = segment.speaker

        # If there's a speaker change or current_segment is None, finalize current and start a new one
        if current_speaker != speaker:
            # Finalize the current segment if it exists
            if current_segment:
                combined_segments.append(current_segment)

            current_speaker = speaker

            # Start a new segment for the current speaker
            current_segment = SpeakerDiarizationSegment(
                time_interval=TimeInterval(
                    start=segment.time_interval.start, end=segment.time_interval.end
                ),
                speaker=current_speaker,
            )
        else:
            if current_segment:
                # Extend the current segment for the same speaker
                # Ensure there is no overlap; take the maximum of current end and incoming end
                current_segment.time_interval.end = max(
                    current_segment.time_interval.end, segment.time_interval.end
                )

        # Adjust the start of the next segment if there's any overlap
        if (
            current_segment
            and len(combined_segments) > 0
            and combined_segments[-1].time_interval.end
            > current_segment.time_interval.start
        ):
            current_segment.time_interval.start = combined_segments[
                -1
            ].time_interval.end

    # Add the last segment if it exists
    if current_segment:
        combined_segments.append(current_segment)

    return combined_segments
