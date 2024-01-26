from ray import serve

from aana.tests.utils import ray_init


def start_ray(deployment):
    """Setup Ray instance for the test."""
    # Setup ray environment and serve
    ray_init()
    app = deployment.bind()
    port = 34422
    test_name = deployment.name
    route_prefix = f"/{test_name}"
    handle = serve.run(app, port=port, name=test_name, route_prefix=route_prefix)
    return handle
