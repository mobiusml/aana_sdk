from pathlib import Path
from aana.models.core.file import PathResult
from aana.models.core.image import Image


def save_image(image: Image, full_path: Path) -> PathResult
    image.save_from_content(full_path)
    return {"path": full_path}
