import json
import time
from uuid import uuid4

import ray
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from aana.api.responses import AanaJSONResponse
from aana.configs.settings import settings as aana_settings
from aana.core.models.chat import ChatCompletion, ChatCompletionRequest, ChatDialog
from aana.core.models.sampling import SamplingParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle

router = APIRouter(
    tags=["openai-api"], include_in_schema=aana_settings.openai_endpoint_enabled
)


@router.post(
    "/chat/completions",
    response_model=ChatCompletion,
)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions requests for OpenAI compatible API."""
    if not aana_settings.openai_endpoint_enabled:
        return AanaJSONResponse(
            content={
                "error": {"message": "The OpenAI-compatible endpoint is not enabled."}
            },
            status_code=404,
        )

    async def _async_chat_completions(
        handle: AanaDeploymentHandle,
        dialog: ChatDialog,
        sampling_params: SamplingParams,
    ):
        async for response in handle.chat_stream(
            dialog=dialog, sampling_params=sampling_params
        ):
            chunk = {
                "id": f"chatcmpl-{uuid4().hex}",
                "object": "chat.completion.chunk",
                "model": request.model,
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": response["text"], "role": "assistant"},
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # Check if the deployment exists
    try:
        handle = await AanaDeploymentHandle.create(request.model)
    except ray.serve.exceptions.RayServeException:
        return AanaJSONResponse(
            content={
                "error": {"message": f"The model `{request.model}` does not exist."}
            },
            status_code=404,
        )

    # Check if the deployment is a chat model
    if not hasattr(handle, "chat") or not hasattr(handle, "chat_stream"):
        return AanaJSONResponse(
            content={
                "error": {"message": f"The model `{request.model}` does not exist."}
            },
            status_code=404,
        )

    dialog = ChatDialog(
        messages=request.messages,
    )

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
    )

    if request.stream:
        return StreamingResponse(
            _async_chat_completions(handle, dialog, sampling_params),
            media_type="application/x-ndjson",
        )
    else:
        response = await handle.chat(dialog=dialog, sampling_params=sampling_params)
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "model": request.model,
            "created": int(time.time()),
            "choices": [{"index": 0, "message": response["message"]}],
        }
