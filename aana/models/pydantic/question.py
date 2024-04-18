from typing import Annotated

from pydantic import Field

Question = Annotated[str, Field(alias="question", description="The question.")]
