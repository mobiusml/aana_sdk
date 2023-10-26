from typing import Any, Dict, List
from ray import serve
from fastapi.openapi.utils import get_openapi

from mobius_pipeline.pipeline import Pipeline
from aana.api.api_generation import Endpoint, add_custom_schemas_to_openapi_schema

from aana.api.app import app
from aana.api.responses import AanaJSONResponse


@serve.deployment(route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    ready = False

    def __init__(
        self,
        endpoints: List[Endpoint],
        pipeline_nodes: List[Dict[str, Any]],
        context: Dict[str, Any],
    ):
        """
        Args:
            deployments (Dict): The dictionary of deployments.
                It is passed to the context to the pipeline so the pipeline can access the deployments handles.
        """

        self.context = context
        self.endpoints = endpoints
        self.pipeline = Pipeline(pipeline_nodes, context)

        self.custom_schemas = {}
        for endpoint in self.endpoints:
            endpoint.register(app=app, pipeline=self.pipeline)
            # get schema for endpoint to add to openapi schema
            schema = endpoint.get_request_schema(self.pipeline)
            self.custom_schemas[endpoint.name] = schema

        app.openapi = self.custom_openapi
        self.ready = True

    def custom_openapi(self) -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        # TODO: populate title and version from package info
        openapi_schema = get_openapi(title="Aana", version="0.1.0", routes=app.routes)
        openapi_schema = add_custom_schemas_to_openapi_schema(
            openapi_schema=openapi_schema, custom_schemas=self.custom_schemas
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    def get_context(self):
        """
        Returns:
            dict: The context of the pipeline.
        """
        return self.context

    @app.get("/api/ready")
    async def is_ready(self):
        """
        The endpoint for checking if the application is ready.

        Real reason for this endpoint is to make automatic endpoint generation work.
        If RequestHandler doesn't have any endpoints defined manually,
        then the automatic endpoint generation doesn't work.
        #TODO: Find a better solution for this.

        Returns:
            AanaJSONResponse: The response containing the ready status.
        """

        return AanaJSONResponse(content={"ready": self.ready})
