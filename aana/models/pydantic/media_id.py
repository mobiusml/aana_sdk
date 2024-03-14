from types import MappingProxyType

from pydantic import BaseModel


class MediaId(BaseModel):
    """A model for a media ID."""

    __root__: str

    def __str__(self):
        """Convert to a string."""
        return self.__root__

    class Config:
        schema_extra = MappingProxyType({"description": "A media ID."})
