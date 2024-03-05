from pathlib import Path
from uuid import uuid4

import PIL.Image

from aana.configs.settings import settings
from aana.models.core.file import PathResult
from aana.models.pydantic.image_input import ImageInput


def save_image(image: PIL.Image.Image, full_path: Path | None = None) -> PathResult:
    """Saves an image to the given full path, or randomely generates one if no path is supplied.

    Arguments:
        image (Image): the image to save
        full_path (Path|None): the path to save the image to. If None, will generate one randomly.

    Returns:
        PathResult: contains the path to the saved image.
    """
    if not full_path:
        full_path = settings.image_dir / f"{uuid4()}.png"
    image.save(full_path)
    return {"path": full_path}


def load_image_input(image_input: ImageInput) -> dict:
    """Loads an image_input and turns it into an Image object."""
    return {"image": image_input.convert_input_to_object()}
