from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    A pydantic model for SDK settings.

    """

    tmp_data_dir: Path = Path("/tmp/aana_data")


settings = Settings()
