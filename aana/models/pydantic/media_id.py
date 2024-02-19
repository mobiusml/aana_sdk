import uuid
from types import MappingProxyType

from pydantic import root_validator

from aana.models.pydantic.base import BaseStringModel


class MediaId(BaseStringModel):
    """A model for a media id."""

    # method to generate a random media id with uuid
    @classmethod
    def random(cls) -> "MediaId":
        """Generate a random media id."""
        return cls(__root__=str(uuid.uuid4()))

    # check if the media id is not empty string
    @root_validator
    def media_id_must_not_be_empty(cls, values):
        """Validates that the media_id is not an empty string.

        Args:
            values (dict): The values of the fields.

        Returns:
            dict: The values of the fields.

        Raises:
            ValueError: if the media_id is an empty string.
        """
        if "__root__" not in values:
            raise ValueError("media_id is not provided")  # noqa: TRY003
        if values["__root__"] == "":
            raise ValueError("media_id cannot be an empty string")  # noqa: TRY003
        return values

    class Config:
        schema_extra = MappingProxyType({"description": "Media ID"})
