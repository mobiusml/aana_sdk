from pathlib import Path
from typing import TypedDict


class PathResult(TypedDict):
    """Represents a path result describing a file on disk."""

    path: Path
