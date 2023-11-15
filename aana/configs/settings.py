from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    A pydantic model for SDK settings.

    """

    tmp_data_dir: Path = Path("/tmp/aana_data")
    youtube_video_dir = tmp_data_dir / "youtube_videos"
    image_dir = tmp_data_dir / "images"


settings = Settings()
