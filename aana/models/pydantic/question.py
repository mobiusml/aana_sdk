from typing import Annotated

from pydantic import Field

Question = Annotated[str, Field(description="The question.")]
