from aana.api.api_generation import Endpoint, EndpointOutput

endpoints = {
    "llama2": [
        Endpoint(
            name="llm_generate",
            path="/llm/generate",
            summary="Generate text using LLaMa2 7B Chat",
            outputs=[
                EndpointOutput(name="completion", output="vllm_llama2_7b_chat_output")
            ],
        ),
        Endpoint(
            name="llm_generate_stream",
            path="/llm/generate_stream",
            summary="Generate text using LLaMa2 7B Chat (streaming)",
            outputs=[
                EndpointOutput(
                    name="completion",
                    output="vllm_llama2_7b_chat_output_stream",
                    streaming=True,
                )
            ],
            streaming=True,
        ),
        Endpoint(
            name="llm_chat",
            path="/llm/chat",
            summary="Chat with LLaMa2 7B Chat",
            outputs=[
                EndpointOutput(
                    name="message", output="vllm_llama2_7b_chat_output_message"
                )
            ],
        ),
        Endpoint(
            name="llm_chat_stream",
            path="/llm/chat_stream",
            summary="Chat with LLaMa2 7B Chat (streaming)",
            outputs=[
                EndpointOutput(
                    name="completion",
                    output="vllm_llama2_7b_chat_output_dialog_stream",
                    streaming=True,
                )
            ],
            streaming=True,
        ),
    ],
    "blip2": [
        Endpoint(
            name="blip2_generate",
            path="/image/generate_captions",
            summary="Generate captions for images using BLIP2 OPT-2.7B",
            outputs=[
                EndpointOutput(name="captions", output="captions_hf_blip2_opt_2_7b")
            ],
        ),
        Endpoint(
            name="blip2_video_generate",
            path="/video/generate_captions",
            summary="Generate captions for videos using BLIP2 OPT-2.7B",
            outputs=[
                EndpointOutput(
                    name="captions", output="video_captions_hf_blip2_opt_2_7b"
                ),
                EndpointOutput(name="timestamps", output="video_timestamps"),
                EndpointOutput(name="caption_ids", output="caption_ids"),
            ],
        ),
    ],
    "whisper": [
        Endpoint(
            name="whisper_transcribe",
            path="/video/transcribe",
            summary="Transcribe a video using Whisper Medium",
            outputs=[
                EndpointOutput(
                    name="transcription",
                    output="video_transcriptions_whisper_medium",
                    streaming=True,
                ),
                EndpointOutput(
                    name="segments",
                    output="video_transcriptions_segments_whisper_medium",
                    streaming=True,
                ),
                EndpointOutput(
                    name="info",
                    output="video_transcriptions_info_whisper_medium",
                    streaming=True,
                ),
                EndpointOutput(name="transcription_id", output="transcription_id"),
            ],
            streaming=True,
        ),
        Endpoint(
            name="load_transcription",
            path="/video/get_transcription",
            summary="Load a transcription",
            outputs=[
                EndpointOutput(
                    name="transcription",
                    output="video_transcriptions_whisper_medium_from_db",
                ),
                EndpointOutput(
                    name="segments",
                    output="video_transcriptions_segments_whisper_medium_from_db",
                ),
                EndpointOutput(
                    name="info",
                    output="video_transcriptions_info_whisper_medium_from_db",
                ),
            ],
        ),
        Endpoint(
            name="delete_media_id",
            path="/video/delete",
            summary="Delete a video",
            outputs=[
                EndpointOutput(name="deleted_media_id", output="deleted_media_id")
            ],
        ),
    ],
    "chat_with_video": [
        Endpoint(
            name="index_video_stream",
            path="/video/index_stream",
            summary="Index a video and return the captions and transcriptions as a stream",
            outputs=[
                EndpointOutput(
                    name="transcription",
                    output="video_transcriptions_whisper_medium",
                    streaming=True,
                ),
                EndpointOutput(
                    name="segments",
                    output="video_transcriptions_segments_whisper_medium",
                    streaming=True,
                ),
                EndpointOutput(
                    name="info",
                    output="video_transcriptions_info_whisper_medium",
                    streaming=True,
                ),
                EndpointOutput(
                    name="captions",
                    output="video_captions_hf_blip2_opt_2_7b",
                    streaming=True,
                ),
                EndpointOutput(
                    name="timestamps", output="video_timestamps", streaming=True
                ),
                EndpointOutput(name="caption_ids", output="caption_ids"),
                EndpointOutput(name="transcription_id", output="transcription_id"),
            ],
            streaming=True,
        ),
        Endpoint(
            name="video_chat_stream",
            path="/video/chat_stream",
            summary="Chat with video using LLaMa2 7B Chat (streaming)",
            outputs=[
                EndpointOutput(
                    name="completion",
                    output="vllm_llama2_7b_chat_output_dialog_stream_video",
                    streaming=True,
                )
            ],
            streaming=True,
        ),
        Endpoint(
            name="video_metadata",
            path="/video/metadata",
            summary="Load video metadata",
            outputs=[
                EndpointOutput(name="metadata", output="video_metadata"),
            ],
        ),
        Endpoint(
            name="delete_media_id",
            path="/video/delete",
            summary="Delete a video",
            outputs=[
                EndpointOutput(name="deleted_media_id", output="deleted_media_id")
            ],
        ),
    ],
    "stablediffusion2": [
        Endpoint(
            name="imagegen",
            path="/generate_image",
            summary="Generates an image from a text prompt",
            outputs=[
                EndpointOutput(
                    name="image_path_stablediffusion2",
                    output="image_path_stablediffusion2",
                )
            ],
        )
    ],
    "standardconceptsv2": [
        Endpoint(
            name="image_tagging",
            path="/image/tagging/standard_concepts",
            summary="Tags an image",
            outputs=[
                EndpointOutput(
                    name="standard_concepts_v2_predictions",
                    output="standard_concepts_v2_predictions",
                )
            ],
        )
    ],
}
