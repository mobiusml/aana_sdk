from pathlib import Path
from uuid import uuid4

from aana.configs.settings import settings
from aana.models.core.file import PathResult
from aana.models.core.image import Image


def save_image(image: Image, full_path: Path | None = None) -> PathResult:
    """Saves an image to the given full path, or randomely generates one if no path is supplied.

    Arguments:
        image (Image): the image to save
        full_path (Path|None): the path to save the image to. If None, will generate one randomly.

    Returns:
        PathResult: contains the path to the saved image.
    """
    if not full_path:
        full_path = settings.tmp_image_dir / f"{uuid4()}.png"
    image.save_from_content(full_path)
    return {"path": full_path}
