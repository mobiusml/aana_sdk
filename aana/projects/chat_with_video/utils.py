from aana.core.models.chat import ChatDialog, ChatMessage, Question
from aana.core.models.video import VideoMetadata


def generate_dialog(
    metadata: VideoMetadata,
    timeline: str,
    question: Question,
) -> ChatDialog:
    """Generates a dialog from the metadata and timeline of a video.

    Args:
        metadata (VideoMetadata): the metadata of the video
        timeline (str): the timeline of the video
        question (Question): the question to ask

    Returns:
        ChatDialog: the generated dialog
    """
    system_prompt_preamble = """You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while ensuring safety. You will be provided with a script in json format for a video containing information from visual captions and audio transcripts. Each entry in the script follows the format:

    {{
    "start_time":"start_time_in_seconds",
    "end_time": "end_time_in_seconds",
    "audio_transcript": "the_transcript_from_automatic_speech_recognition_system",
    "visual_caption": "the_caption_of_the_visuals_using_computer_vision_system"
    }}
    Note that the audio_transcript can sometimes be empty.

    Ensure you do not introduce any new named entities in your output and maintain the utmost factual accuracy in your responses.

    In the addition you will be provided with title of video extracted.
    """
    instruction = (
        "Provide a short and concise answer to the following user's question. "
        "Avoid mentioning any details about the script in JSON format. "
        "For example, a good response would be: 'Based on the analysis, "
        "here are the most relevant/useful/aesthetic moments.' "
        "A less effective response would be: "
        "'Based on the provided visual caption/audio transcript, "
        "here are the most relevant/useful/aesthetic moments. The user question is "
    )

    user_prompt_template = (
        "{instruction}"
        "Given the timeline of audio and visual activities in the video below "
        "I want to find out the following: {question}"
        "The timeline is: "
        "{timeline}"
        "\n"
        "The title of the video is {video_title}"
    )

    messages = []
    messages.append(ChatMessage(content=system_prompt_preamble, role="system"))
    messages.append(
        ChatMessage(
            content=user_prompt_template.format(
                instruction=instruction,
                question=question,
                timeline=timeline,
                video_title=metadata.title,
            ),
            role="user",
        )
    )

    dialog = ChatDialog(messages=messages)
    return dialog
