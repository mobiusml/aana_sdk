from typing import Dict, List, Optional
from pydantic import BaseModel, Extra, Field

from aana.models.pydantic.prompt import Prompt
from aana.models.pydantic.sampling_params import SamplingParams


class LLMRequest(BaseModel):
    """
    This class is used to represent a request to LLM.

    Attributes:
        prompt (Prompt): A prompt to LLM.
        sampling_params (SamplingParams): Sampling parameters for generating text.
    """

    prompt: Prompt = Field(..., description="A prompt to LLM.")
    sampling_params: Optional[SamplingParams] = Field(
        None, description="Sampling parameters for generating text."
    )

    class Config:
        extra = Extra.forbid
        schema_extra = {"description": "A request to LLM."}
