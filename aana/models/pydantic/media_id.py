from typing import Annotated, Any

from pydantic import Field, ValidationInfo, ValidatorFunctionWrapHandler
from pydantic.functional_validators import WrapValidator


def verify_media_id_must_not_be_empty(
    v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> str:
    """Validates that the media_id is not an empty string."""
    assert v != "", "media_id cannot be an empty string"  # noqa: S101
    return v


MediaId = Annotated[
    str,
    Field(description="The media ID."),
    WrapValidator(verify_media_id_must_not_be_empty),
]
