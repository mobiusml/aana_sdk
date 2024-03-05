from pydantic import ConfigDict

from aana.models.pydantic.base import BaseStringModel


class Prompt(BaseStringModel):
    """A model for a user prompt to LLM."""

    model_config = ConfigDict(json_schema_extra={"description": "A prompt to LLM."})
