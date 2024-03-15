import uuid

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from aana.models.pydantic.base import BaseStringModel


class MediaId(BaseStringModel):
    """A model for a media id."""

    # method to generate a random media id with uuid
    @classmethod
    def random(cls) -> "MediaId":
        """Generate a random media id."""
        return cls(str(uuid.uuid4()))

    # check if the media id is not empty string
    @model_validator(mode="after")
    def verify_media_id_must_not_be_empty(self) -> Self:
        """Validates that the media_id is not an empty string.

        Args:
            values (dict): The values of the fields.

        Returns:
            dict: The values of the fields.

        Raises:
            ValueError: if the media_id is an empty string.
        """
        if self.root == "":
            raise ValueError("media_id cannot be an empty string")  # noqa: TRY003
        return self

    model_config = ConfigDict(json_schema_extra={"description": "Media ID"})
