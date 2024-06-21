import json
import time
from typing import Any
from uuid import uuid4

import ray
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from ray import serve

from aana.api.api_generation import Endpoint, add_custom_schemas_to_openapi_schema
from aana.api.app import app
from aana.api.event_handlers.event_manager import EventManager
from aana.api.responses import AanaJSONResponse
from aana.configs.settings import settings as aana_settings
from aana.core.models.chat import ChatCompletetion, ChatCompletionRequest, ChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.task import TaskId
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.storage.services.task import TaskInfo, delete_task, get_task_info


@serve.deployment(ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(self, endpoints: list[Endpoint]):
        """Constructor.

        Args:
            endpoints (dict): List of endpoints for the request
        """
        self.endpoints = endpoints

        self.event_manager = EventManager()
        self.custom_schemas: dict[str, dict] = {}
        for endpoint in self.endpoints:
            endpoint.register(
                app=app,
                custom_schemas=self.custom_schemas,
                event_manager=self.event_manager,
            )

        app.openapi = self.custom_openapi
        self.ready = True

    def custom_openapi(self) -> dict[str, Any]:
        """Returns OpenAPI schema, generating it if necessary."""
        if app.openapi_schema:
            return app.openapi_schema
        # TODO: populate title and version from package info
        openapi_schema = get_openapi(title="Aana", version="0.1.0", routes=app.routes)
        openapi_schema = add_custom_schemas_to_openapi_schema(
            openapi_schema=openapi_schema, custom_schemas=self.custom_schemas
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    @app.get("/api/ready")
    async def is_ready(self):
        """The endpoint for checking if the application is ready.

        Real reason for this endpoint is to make automatic endpoint generation work.
        If RequestHandler doesn't have any endpoints defined manually,
        then the automatic endpoint generation doesn't work.
        #TODO: Find a better solution for this.

        Returns:
            AanaJSONResponse: The response containing the ready status.
        """
        return AanaJSONResponse(content={"ready": self.ready})

    async def call_endpoint(self, path: str, **kwargs: dict[str, Any]) -> Any:
        """Call the endpoint from FastAPI with the given name.

        Args:
            path (str): The path of the endpoint.
            **kwargs: The arguments to pass to the endpoint.

        Returns:
            Any: The response from the endpoint.
        """
        for e in self.endpoints:
            if e.path == path:
                endpoint = e
                break
        else:
            raise ValueError(f"Endpoint {path} not found")  # noqa: TRY003

        if not endpoint.initialized:
            await endpoint.initialize()

        if endpoint.is_streaming_response():
            return [item async for item in endpoint.run(**kwargs)]
        else:
            return await endpoint.run(**kwargs)

    @app.get(
        "/tasks/get/{task_id}",
        summary="Get Task Status",
        description="Get the task status by task ID.",
        include_in_schema=aana_settings.task_queue.enabled,
    )
    async def get_task_endpoint(self, task_id: str) -> TaskInfo:
        """Get the task with the given ID.

        Args:
            task_id (str): The ID of the task.

        Returns:
            TaskInfo: The status of the task.
        """
        return get_task_info(task_id)

    @app.get(
        "/tasks/delete/{task_id}",
        summary="Delete Task",
        description="Delete the task by task ID.",
        include_in_schema=aana_settings.task_queue.enabled,
    )
    async def delete_task_endpoint(self, task_id: str) -> TaskId:
        """Delete the task with the given ID.

        Args:
            task_id (str): The ID of the task.

        Returns:
            TaskInfo: The deleted task.
        """
        task = delete_task(task_id)
        return TaskId(task_id=str(task.id))

    @app.post("/chat/completions", response_model=ChatCompletetion)
    async def chat_completions(self, request: ChatCompletionRequest):
        """Handle chat completions requests for OpenAI compatible API."""

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
