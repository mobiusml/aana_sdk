from types import MappingProxyType

from pydantic import BaseModel


class Prompt(BaseModel):
    """A model for a user prompt to LLM."""

    __root__: str

    def __str__(self):
        """Convert to a string."""
        return self.__root__

    class Config:
        schema_extra = MappingProxyType({"description": "A prompt to LLM."})
