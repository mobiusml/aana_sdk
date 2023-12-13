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
                    name="completion", output="vllm_llama2_7b_chat_output_stream"
                )
            ],
            streaming=True,
        ),
    ],
    "zephyr": [
        Endpoint(
            name="zephyr_generate",
            path="/llm/generate",
            summary="Generate text using Zephyr 7B Beta",
            outputs=[
                EndpointOutput(name="completion", output="vllm_zephyr_7b_beta_output")
            ],
        )
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
                    name="captions", output="video_captions_hf_blip2_opt_2_7b"
                ),
                EndpointOutput(name="timestamps", output="video_timestamps"),
            ],
            streaming=True,
        ),
        Endpoint(
            name="whisper_transcribe",
            path="/video/transcribe",
            summary="Transcribe a video using Whisper Medium",
            outputs=[
                EndpointOutput(
                    name="transcription", output="video_transcriptions_whisper_medium"
                ),
                EndpointOutput(
                    name="segments",
                    output="video_transcriptions_segments_whisper_medium",
                ),
                EndpointOutput(
                    name="info", output="video_transcriptions_info_whisper_medium"
                ),
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
                    name="completion", output="vllm_llama2_7b_chat_output_stream"
                )
            ],
            streaming=True,
        ),
    ],
}
