from aana.sdk import AanaSDK

from .deployments import stablediffusion2_deployment
from .endpoints import ImageGenerationEndpoint, IMAGEGEN_DEPLOYMENT_NAME

aana_app = AanaSDK(name="stablediffusion2")

aana_app.register_deployment(
    name=IMAGEGEN_DEPLOYMENT_NAME,
    instance=stablediffusion2_deployment,
)

aana_app.register_endpoint(
    name="generate_image,
    path="/generate_image",
    summary="Generates an image from a text prompt",
    endpoint_cls=ImageGenerationEndpoint,
)
