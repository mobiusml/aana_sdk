import argparse

from aana.configs.deployments import (
    vllm_llama2_7b_chat_deployment,
)
from aana.projects.llama2.endpoints import (
    LlmChatEndpoint,
    LlmChatStreamEndpoint,
    LlmGenerateEndpoint,
    LlmGenerateStreamEndpoint,
)
from aana.sdk import AanaSDK

deployments = [
    {
        "name": "llm_deployment",
        "instance": vllm_llama2_7b_chat_deployment,
    },
]

endpoints = [
    {
        "name": "llm_generate",
        "path": "/llm/generate",
        "summary": "Generate text using LLaMa2 7B Chat",
        "endpoint_cls": LlmGenerateEndpoint,
    },
    {
        "name": "llm_generate_stream",
        "path": "/llm/generate_stream",
        "summary": "Generate text using LLaMa2 7B Chat (streaming)",
        "endpoint_cls": LlmGenerateStreamEndpoint,
    },
    {
        "name": "llm_chat",
        "path": "/llm/chat",
        "summary": "Chat with LLaMa2 7B Chat",
        "endpoint_cls": LlmChatEndpoint,
    },
    {
        "name": "llm_chat_stream",
        "path": "/llm/chat_stream",
        "summary": "Chat with LLaMa2 7B Chat (streaming)",
        "endpoint_cls": LlmChatStreamEndpoint,
    },
]

if __name__ == "__main__":
    """Runs the application."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8000)
    arg_parser.add_argument("--host", type=str, default="127.0.0.1")
    args = arg_parser.parse_args()

    aana_app = AanaSDK(port=args.port, host=args.host, show_logs=True)

    for deployment in deployments:
        aana_app.register_deployment(
            name=deployment["name"],
            instance=deployment["instance"],
        )

    for endpoint in endpoints:
        aana_app.register_endpoint(
            name=endpoint["name"],
            path=endpoint["path"],
            summary=endpoint["summary"],
            endpoint_cls=endpoint["endpoint_cls"],
        )

    aana_app.deploy(blocking=True)