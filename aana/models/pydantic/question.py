from types import MappingProxyType

from aana.models.pydantic.base import BaseStringModel


class Question(BaseStringModel):
    """A model for a question."""

    class Config:
        schema_extra = MappingProxyType({"description": "A question."})
