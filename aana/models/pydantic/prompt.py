from typing import Annotated

from pydantic import Field

Prompt = Annotated[str, Field(description="The prompt for the LLM.")]
