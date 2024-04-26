from typing import Annotated

from pydantic import Field

Caption = Annotated[str, Field(description="A caption.")]

CaptionsList = Annotated[list[Caption], Field(description="A list of captions.")]
