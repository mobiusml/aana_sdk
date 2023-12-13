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
                    name="captions", output="videos_captions_hf_blip2_opt_2_7b"
                ),
                EndpointOutput(name="timestamps", output="timestamps"),
            ],
        ),
    ],
    "video": [
        Endpoint(
            name="video_extract_frames",
            path="/video/extract_frames",
            summary="Extract frames from a video",
            outputs=[
                EndpointOutput(name="timestamps", output="timestamps"),
                EndpointOutput(name="duration", output="duration"),
            ],
        )
    ],
    "whisper": [
        Endpoint(
            name="whisper_transcribe",
            path="/video/transcribe",
            summary="Transcribe a video using Whisper Medium",
            outputs=[
                EndpointOutput(
                    name="transcription", output="videos_transcriptions_whisper_medium"
                ),
                EndpointOutput(
                    name="segments",
                    output="videos_transcriptions_segments_whisper_medium",
                ),
                EndpointOutput(
                    name="info", output="videos_transcriptions_info_whisper_medium"
                ),
            ],
        )
    ],
    "chat_with_video": [
        Endpoint(
            name="blip2_video_generate",
            path="/video/generate_captions",
            summary="Generate captions for videos using BLIP2 OPT-2.7B",
            outputs=[
                EndpointOutput(
                    name="captions",
                    output="video_captions_hf_blip2_opt_2_7b",
                    streaming=True,
                ),
                EndpointOutput(
                    name="timestamps", output="video_timestamps", streaming=True
                ),
                EndpointOutput(
                    name="video_captions_path", output="video_captions_path"
                ),
            ],
            streaming=True,
        ),
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
                EndpointOutput(name="transcription_path", output="transcription_path"),
            ],
            streaming=True,
        ),
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
                EndpointOutput(name="transcription_path", output="transcription_path"),
                EndpointOutput(
                    name="captions",
                    output="video_captions_hf_blip2_opt_2_7b",
                    streaming=True,
                ),
                EndpointOutput(
                    name="timestamps", output="video_timestamps", streaming=True
                ),
                EndpointOutput(name="combined_timeline", output="combined_timeline"),
                EndpointOutput(
                    name="combined_timeline_path", output="combined_timeline_path"
                ),
                EndpointOutput(
                    name="video_metadata_path", output="video_metadata_path"
                ),
                EndpointOutput(
                    name="video_captions_path", output="video_captions_path"
                ),
                EndpointOutput(name="transcription_path", output="transcription_path"),
            ],
            streaming=True,
        ),
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
    ],
}
