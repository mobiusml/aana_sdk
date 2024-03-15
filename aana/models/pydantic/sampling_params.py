from pydantic import BaseModel, ConfigDict, Field, field_validator


class SamplingParams(BaseModel):
    """A model for sampling parameters of LLM.

    Attributes:
        temperature (float): Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p (float): Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k (int): Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        max_tokens (int): The maximum number of tokens to generate.
    """

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Float that controls the randomness of the sampling. "
            "Lower values make the model more deterministic, "
            "while higher values make the model more random. "
            "Zero means greedy sampling."
        ),
    )
    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Float that controls the cumulative probability of the top tokens to consider. "
            "Must be in (0, 1]. Set to 1 to consider all tokens."
        ),
    )
    top_k: int | None = Field(
        default=None,
        description=(
            "Integer that controls the number of top tokens to consider. "
            "Set to -1 to consider all tokens."
        ),
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="The maximum number of tokens to generate."
    )

    @field_validator("top_k")
    def check_top_k(cls, v: int):
        """Validates a top_k argument.

        Makes sure it is either -1, or at least 1.

        Args:
            v (int): Value to validate.

        Raises:
            ValueError: The value is not valid.

        Returns:
            The top_k value.
        """
        if v is None:
            return v
        if v < -1 or v == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, got {v}.")  # noqa: TRY003
        return v

    model_config = ConfigDict(
        json_schema_extra={"description": "Sampling parameters for generating text."}
    )
