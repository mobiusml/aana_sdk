from typing import Any

from fastapi.openapi.utils import get_openapi
from mobius_pipeline.pipeline import Pipeline
from ray import serve

from aana.api.api_generation import Endpoint, add_custom_schemas_to_openapi_schema
from aana.api.app import app
from aana.api.event_handlers.event_manager import EventManager
from aana.api.responses import AanaJSONResponse

# TODO: improve type annotations


@serve.deployment(ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(
        self,
        endpoints: list[Endpoint],
        pipeline_nodes: list[dict[str, Any]],
        context: dict[str, Any],
    ):
        """Constructor.

        Args:
            endpoints (dict): List of endpoints for the request
            pipeline_nodes (list[dict[str, Any]]]): List of nodes for the pipeline
            context (dict): Pipeline context so pipeline can use  deployment handles.
        """
        self.context = context
        self.endpoints = endpoints
        self.pipeline = Pipeline(pipeline_nodes, context)
        self.event_manager = EventManager()

        self.custom_schemas: dict[str, dict] = {}
        for endpoint in self.endpoints:
            endpoint.register(
                app=app,
                pipeline=self.pipeline,
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

    def get_context(self) -> dict:
        """Getter for pipeline context.

        Returns:
            dict: The context of the pipeline.
        """
        return self.context

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
