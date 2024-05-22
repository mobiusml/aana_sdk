# ruff: noqa: S101
import haystack
import haystack.components.preprocessors
import PIL
import pytest

from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.integrations.haystack.deployment_component import AanaDeploymentComponent
from aana.tests.deployments.test_stablediffusion2_deployment import (
    setup_deployment as setup_stablediffusion2_deployment,  # noqa: F401
)
from aana.tests.utils import is_gpu_available, is_using_deployment_cache


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
async def test_haystack_wrapper(setup_stablediffusion2_deployment):  # noqa: F811
    """Tests haystack wrapper for deployments."""
    deployment_name = "sd2_deployment"
    method_name = "generate"
    result_key = "image"
    deployment_handle = await AanaDeploymentHandle.create(deployment_name)
    component = AanaDeploymentComponent(deployment_handle, method_name)
    result = component.run(prompt="foo")
    assert result_key in result, result


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
async def test_haystack_pipeline(setup_stablediffusion2_deployment):  # noqa: F811
    """Tests haystack wrapper in a pipeline."""

    # Haystack components generally take lists of things
    # so we need a component to turn a list of strings into a single string
    @haystack.component
    class TextCombinerComponent:
        @haystack.component.output_types(text=str)
        def run(self, texts: list[str]):
            return {"text": " ".join(texts)}

    # Haystack also doesn't have any simple components that
    # take images as inputs, so let's have one of those as well
    @haystack.component
    class ImageSizeComponent:
        @haystack.component.output_types(size=tuple[int, int])
        def run(self, image: PIL.Image.Image):
            return {"size": image.size}

    deployment_name = "sd2_deployment"
    method_name = "generate"
    # result_key = "image"
    deployment_handle = await AanaDeploymentHandle.create(deployment_name)
    aana_component = AanaDeploymentComponent(deployment_handle, method_name)
    aana_component.warm_up()
    text_cleaner = haystack.components.preprocessors.TextCleaner(
        convert_to_lowercase=False, remove_punctuation=True, remove_numbers=True
    )
    text_combiner = TextCombinerComponent()
    image_sizer = ImageSizeComponent()

    pipeline = haystack.Pipeline()
    pipeline.add_component("stablediffusion2", aana_component)
    pipeline.add_component("text_cleaner", text_cleaner)
    pipeline.add_component("text_combiner", text_combiner)
    pipeline.add_component("image_sizer", image_sizer)
    pipeline.connect("text_cleaner.texts", "text_combiner.texts")
    pipeline.connect("text_combiner.text", "stablediffusion2.prompt")
    pipeline.connect("stablediffusion2.image", "image_sizer.image")
    result = pipeline.run({"text_cleaner": {"texts": ["A? dog!"]}})

    assert "image_sizer" in result
    assert "size" in result["image_sizer"]
    assert len(result["image_sizer"]["size"]) == 2


@pytest.mark.asyncio
async def test_haystack_wrapper_fails(setup_stablediffusion2_deployment):  # noqa: F811
    """Tests that haystack wrapper raises if method_name is missing."""
    deployment_name = "sd2_deployment"
    missing_method_name = "does_not_exist"
    deployment_handle = await AanaDeploymentHandle.create(deployment_name)
    with pytest.raises(AttributeError):
        _component = AanaDeploymentComponent(deployment_handle, missing_method_name)
