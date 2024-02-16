from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

from aana.models.pydantic.base import BaseStringModel


class Prompt(BaseStringModel):
    """A model for a user prompt to LLM."""

    model_config = ConfigDict(
        json_schema_extra=MappingProxyType({"description": "A prompt to LLM."})
    )
