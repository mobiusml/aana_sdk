from collections import defaultdict
from typing import TypedDict

from aana.core.models.asr import (
    AsrSegment,
    AsrTranscription,
    AsrTranscriptionInfo,
    AsrWord,
)
from aana.core.models.speaker import SpeakerDiarizationSegment
from aana.core.models.time import TimeInterval

# Utility functions for speaker-related processing


# redefine SpeakerDiarizationOutput and WhisperOutput to prevent circular imports
class SpeakerDiarizationOutput(TypedDict):
    """The output of the Speaker Diarization model.

    Attributes:
        segments (list[SpeakerDiarizationSegment]): The Speaker Diarization segments.
    """

    segments: list[SpeakerDiarizationSegment]


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


# Define sentence ending punctuations:
sentence_ending_punctuations = ".?!"


# AsrSegment and AsrWord has a speaker label that defaults to None.
def assign_word_speakers(
    diarized_output: SpeakerDiarizationOutput,
    transcription: WhisperOutput,
    fill_nearest: bool = False,
) -> WhisperOutput:
    """Assigns speaker labels to each segment and word in the transcription based on diarized output.

    Parameters:
    - diarized_output (SpeakerDiarizationOutput): Contains speaker diarization segments.
    - transcription (WhisperOutput): Transcription data with segments, text, and language_info.
    - fill_nearest (bool): If True, assigns the closest speaker even if there's no positive overlap. Default is False.

    Returns:
    - transcription (WhisperOutput): Transcription updated in-place with the assigned speaker labels.
    """
    for segment in transcription["segments"]:
        # Assign speaker to segment
        segment.speaker = get_speaker_for_interval(
            diarized_output["segments"],
            segment.time_interval.start,
            segment.time_interval.end,
            fill_nearest,
        )

        # Assign speakers to words within the segment
        if segment.words:
            for word in segment.words:
                word.speaker = get_speaker_for_interval(
                    diarized_output["segments"],
                    word.time_interval.start,
                    word.time_interval.end,
                    fill_nearest,
                )

    return transcription


def get_speaker_for_interval(
    sd_segments: list[SpeakerDiarizationSegment],
    start_time: float,
    end_time: float,
    fill_nearest: bool,
) -> str | None:
    """Determines the speaker for a given time interval based on diarized segments.

    Parameters:
    - sd_segments (list[SpeakerDiarizationSegment]): List of speaker diarization segments.
    - start_time (float): Start time of the interval.
    - end_time (float): End time of the interval.
    - fill_nearest (bool): If True, selects the closest speaker even with no overlap.

    Returns:
    - str | None: The identified speaker label, or None if no speaker is found.
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
            # Calculate union duration for potential future use
            # union_duration = max(end_time, interval_end) - min(start_time, interval_start)
            distance = float(
                min(abs(start_time - interval_end), abs(end_time - interval_start))
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


def get_first_word_idx_of_sentence(
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
        if left_idx == 0 or word_list[left_idx - 1][-1] in sentence_ending_punctuations
        else -1
    )


def get_last_word_idx_of_sentence(
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


def find_nearest_speaker(
    index: int, word_speaker_mapping: list[AsrWord], reverse: bool = False
) -> str | None:
    """Find the nearest speaker label in the word_speaker_mapping either forward or backward.

    Parameters:
    - index (int): The index to start searching from.
    - word_speaker_mapping (list[AsrWord]): List of word-speaker mappings.
    - reverse (bool): Search backwards if True; forwards if False. Default is False.

    Returns:
    - str | None: The nearest speaker found or None if not found.
    """
    step = -1 if reverse else 1
    for i in range(index, len(word_speaker_mapping) if not reverse else -1, step):
        if word_speaker_mapping[i].speaker:
            return word_speaker_mapping[i].speaker
    return None


def align_with_punctuation(
    transcription: WhisperOutput, max_words_in_sentence: int = 50
) -> list[AsrWord]:
    """Aligns speaker labels with sentence boundaries defined by punctuation.

    Parameters:
    - transcription (WhisperOutput): transcription with speaker information.
    - max_words_in_sentence (int): Maximum number of words allowed in a sentence.

    Returns:
    - list[AsrWord]: Realigned word-speaker mappings.
    """
    new_segments = [segment.words for segment in transcription["segments"]]
    word_speaker_mapping = [word for segment in new_segments for word in segment]
    words_list = [item.word for item in word_speaker_mapping]
    speaker_list = [item.speaker for item in word_speaker_mapping]

    # Fill missing speaker labels by finding the nearest speaker
    for i, item in enumerate(word_speaker_mapping):
        if item.speaker is None:
            item.speaker = find_nearest_speaker(i, word_speaker_mapping, reverse=i > 0)
            speaker_list[i] = item.speaker

    # Align speakers with sentence boundaries
    k = 0
    while k < len(word_speaker_mapping):
        if (
            k < len(word_speaker_mapping) - 1
            and speaker_list[k] != speaker_list[k + 1]
            and words_list[k][-1] not in sentence_ending_punctuations
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = get_last_word_idx_of_sentence(
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


def create_speaker_segments(
    word_list: list[AsrWord], merge: bool = False
) -> list[AsrSegment]:
    """Creates speaker segments from a list of words with speaker annotations and timing information.

    Parameters:
    - word_list (List[AsrWord]): A list of words with associated speaker and timing details.
    - merge (bool): If True, merges consecutive segments from the same speaker into one segment.

    Returns:
    - List[AsrSegment]: A list of segments where each segment groups words spoken by the same speaker.
    """
    if not word_list:
        return []

    # Initialize variables to track current speaker and segment
    current_speaker = None
    current_segment = None
    final_segments: list[AsrSegment] = []

    for word_info in word_list:
        # Default speaker assignment to the last known speaker if missing
        speaker = word_info.speaker or current_speaker

        # Check for speaker change or start of a new segment
        if speaker != current_speaker:
            if current_segment:
                final_segments.append(current_segment)

            # Start a new segment
            # TODO: Also get the original confidence measurements from segments and update it.
            current_segment = AsrSegment(
                time_interval=TimeInterval(
                    start=word_info.time_interval.start, end=word_info.time_interval.end
                ),
                text=word_info.word,
                speaker=speaker,
                words=[word_info],
                confidence=None,
                no_speech_confidence=None,
            )
            current_speaker = speaker
        else:
            if current_segment:
                # Update the current segment for continuous speaker
                current_segment.time_interval.end = word_info.time_interval.end
                current_segment.text += f"{word_info.word}"  # Add space between words
                current_segment.words.append(word_info)

    # Append the last segment if it exists
    if current_segment:
        final_segments.append(current_segment)

    # Optional merging of consecutive segments by the same speaker
    if merge:
        final_segments = merge_consecutive_speaker_segments(final_segments)

    return final_segments


def merge_consecutive_speaker_segments(
    segments: list[AsrSegment],
) -> list[AsrSegment]:
    """Merges consecutive segments that have the same speaker into a single segment.

    Parameters:
    - segments (List[AsrSegment]): The initial list of segments.

    Returns:
    - List[AsrSegment]: A new list of merged segments.
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


# Full Method
def asr_postprocessing_for_diarization(
    diarized_output: SpeakerDiarizationOutput, transcription: WhisperOutput
) -> WhisperOutput:
    """Perform diarized transcription based on individual deployments.

    Parameters:
    - diarized_output (SpeakerDiarizationOutput): Contains speaker diarization segments.
    - transcription (WhisperOutput): Transcription data with segments, text, and language_info.

    Returns:
    - transcription (WhisperOutput): Updated transcription with diarized information per segment/word.

    """
    # 1. Assign speaker labels to each segment and each word in WhisperOutput based on SpeakerDiarizationOutput.

    speaker_labelled_transcription = assign_word_speakers(
        diarized_output, transcription
    )
    # 2. Aligns the speakers with the punctuations:
    word_speaker_mapping = align_with_punctuation(speaker_labelled_transcription)

    # 3. Create ASR segments by combining the AsrWord with speaker information
    # and optionally combine segments based on speaker info.
    updated_transcription = create_speaker_segments(word_speaker_mapping)
    transcription["segments"] = updated_transcription
    return transcription


# speaker diarization model occationally produce overlapping chunks/ same speaker segments,
# below function combines them properly


def combine_homogeneous_speaker_segs(
    diarized_output: SpeakerDiarizationOutput,
) -> SpeakerDiarizationOutput:
    """Combines segments with the same speaker into homogeneous speaker segments, ensuring no overlapping times.

    Parameters:
    - diarized_output (SpeakerDiarizationOutput): Input with segments that may have overlapping times.

    Returns:
    - SpeakerDiarizationOutput: Output with combined homogeneous speaker segments.
    """
    combined_segments: list = []
    current_speaker = None
    current_segment = None

    for segment in sorted(
        diarized_output["segments"], key=lambda x: x.time_interval.start
    ):
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

    return SpeakerDiarizationOutput(segments=combined_segments)
