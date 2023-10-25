from aana.api.api_generation import Endpoint


endpoints = [
    Endpoint(
        name="llm_generate",
        path="/llm/generate",
        summary="Generate text using LLM",
        outputs=["vllm_llama2_7b_chat_output"],
    )
]
