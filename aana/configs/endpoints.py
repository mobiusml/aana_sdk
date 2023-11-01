from aana.api.api_generation import Endpoint


endpoints = {
    "llama2": [
        Endpoint(
            name="llm_generate",
            path="/llm/generate",
            summary="Generate text using LLaMa2 7B Chat",
            outputs=["vllm_llama2_7b_chat_output"],
        )
    ],
    "zephyr": [
        Endpoint(
            name="zephyr_generate",
            path="/zephyr/generate",
            summary="Generate text using Zephyr 7B Beta",
            outputs=["vllm_zephyr_7b_beta_output"],
        )
    ],
}
