from typing import Annotated

from pydantic import Field

__all__ = ["Caption", "CaptionsList"]

Caption = Annotated[str, Field(description="A caption.")]
"""
A caption.
"""

CaptionsList = Annotated[list[Caption], Field(description="A list of captions.")]
"""
A list of captions.
"""
