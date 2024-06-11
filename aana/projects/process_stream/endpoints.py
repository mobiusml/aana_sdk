from collections.abc import AsyncGenerator
from typing import Annotated, TypedDict

from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.stream import StreamInput
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.integrations.external.av import fetch_stream_frames
from aana.processors.remote import run_remote


class CaptionStreamOutput(TypedDict):
    """The output of the transcribe video endpoint."""

    captions: Annotated[list[str], Field(..., description="Captions")]
    timestamps: Annotated[
        list[float], Field(..., description="Timestamps for each caption in seconds")
    ]


class CaptionStreamEndpoint(Endpoint):
    """Transcribe video in chunks endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.captioning_handle = await AanaDeploymentHandle.create(
            "captioning_deployment"
        )

    async def run(
        self,
        stream: StreamInput,
    ) -> AsyncGenerator[CaptionStreamOutput, None]:
        """Transcribe video in chunks."""
        async for frames_dict in run_remote(fetch_stream_frames)(
            stream_input=stream, batch_size=2
        ):
            captioning_output = await self.captioning_handle.generate_batch(
                images=frames_dict["frames"]
            )

            yield {
                "captions": captioning_output["captions"],
                "timestamps": frames_dict["timestamps"],
            }