from pydantic import ConfigDict

from aana.models.pydantic.base import BaseStringModel


class Question(BaseStringModel):
    """A model for a question."""

    model_config = ConfigDict(json_schema_extra={"description": "A question."})
