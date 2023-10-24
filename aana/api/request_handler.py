from typing import Dict, List, Tuple
from ray import serve

from mobius_pipeline.pipeline import Pipeline

from aana.api.app import app
from aana.api.responses import AanaJSONResponse
from aana.configs.pipeline import nodes
from aana.models.pydantic.llm_request import LLMRequest


async def run_pipeline(
    pipeline: Pipeline, data: Dict, required_outputs: List[str]
) -> Tuple[Dict, Dict[str, float]]:
    """
    This function is used to run a Mobius Pipeline.
    It creates a container from the data, runs the pipeline and returns the output.

    Args:
        pipeline (Pipeline): The pipeline to run.
        data (dict): The data to create the container from.
        required_outputs (List[str]): The required outputs of the pipeline.

    Returns:
        tuple[dict, dict[str, float]]: The output of the pipeline and the execution time of the pipeline.
    """

    # create a container from the data
    container = pipeline.parse_dict(data)

    # run the pipeline
    output, execution_time = await pipeline.run(
        container, required_outputs, return_execution_time=True
    )
    return output, execution_time


@serve.deployment(route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 0.1})
@serve.ingress(app)
class RequestHandler:
    """This class is used to handle requests to the Aana application."""

    def __init__(self, deployments: Dict):
        """
        Args:
            deployments (Dict): The dictionary of deployments.
                It is passed to the context to the pipeline so the pipeline can access the deployments handles.
        """
        self.context = {
            "deployments": deployments,
        }
        self.pipeline = Pipeline(nodes, self.context)

    @app.post("/llm/generate")
    async def generate_llm(self, llm_request: LLMRequest) -> AanaJSONResponse:
        """
        The endpoint for running the LLM.
        It is running the pipeline with the given prompt and sampling parameters.
        This is here as an example and will be replace with automatic endpoint generation.

        Args:
            llm_request (LLMRequest): The LLM request. It contains the prompt and sampling parameters.

        Returns:
            AanaJSONResponse: The response containing the output of the pipeline and the execution time.
        """
        prompt = llm_request.prompt
        sampling_params = llm_request.sampling_params

        output, execution_time = await run_pipeline(
            self.pipeline,
            {"prompt": prompt, "sampling_params": sampling_params},
            ["vllm_llama2_7b_chat_output"],
        )
        output["execution_time"] = execution_time
        return AanaJSONResponse(content=output)
