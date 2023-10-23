from typing import Optional

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """
    A model for sampling parameters of LLM.

    Attributes:
        temperature (float): The temperature.
        top_p (float): Top-p.
        top_k (int): Top-k.
        max_tokens (int): The maximum number of tokens to generate.
    """

    temperature: Optional[float] = Field(default=None, description="The temperature.")
    top_p: Optional[float] = Field(default=None, description="Top-p.")
    top_k: Optional[int] = Field(default=None, description="Top-k.")
    max_tokens: Optional[int] = Field(
        default=None, description="The maximum number of tokens to generate."
    )

    class Config:
        schema_extra = {"description": "Sampling parameters for generating text."}
