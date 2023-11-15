from aana.api.api_generation import Endpoint


endpoints = {
    "llama2": [
        Endpoint(
            name="llm_generate",
            path="/llm/generate",
            summary="Generate text using LLaMa2 7B Chat",
            outputs=["vllm_llama2_7b_chat_output"],
        ),
        Endpoint(
            name="llm_generate_stream",
            path="/llm/generate_stream",
            summary="Generate text using LLaMa2 7B Chat (streaming)",
            outputs=["vllm_llama2_7b_chat_output_stream"],
            streaming=True,
        ),
    ],
    "zephyr": [
        Endpoint(
            name="zephyr_generate",
            path="/llm/generate",
            summary="Generate text using Zephyr 7B Beta",
            outputs=["vllm_zephyr_7b_beta_output"],
        )
    ],
    "blip2": [
        Endpoint(
            name="blip2_generate",
            path="/image/generate_captions",
            summary="Generate captions for images using BLIP2 OPT-2.7B",
            outputs=["captions_hf_blip2_opt_2_7b"],
        ),
        Endpoint(
            name="blip2_video_generate",
            path="/video/generate_captions",
            summary="Generate captions for videos using BLIP2 OPT-2.7B",
            outputs=["video_captions_hf_blip2_opt_2_7b", "timestamps"],
        ),
    ],
    "video": [
        Endpoint(
            name="video_extract_frames",
            path="/video/extract_frames",
            summary="Extract frames from a video",
            outputs=["timestamps", "duration"],
        )
    ],
}
