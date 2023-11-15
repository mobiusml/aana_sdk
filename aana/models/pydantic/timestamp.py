from pydantic import BaseModel, Field


class Timestamp(BaseModel):
    """
    Pydantic schema for Timestamp.

    Attributes:
        start (float): Start time
        end (float): End time
    """

    start: float = Field(ge=0.0, description="Start time")
    end: float = Field(ge=0.0, description="End time")

    class Config:
        schema_extra = {
            "description": "Timestamp",
        }
