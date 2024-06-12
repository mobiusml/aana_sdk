from typing import Any

from pydantic import BaseModel, Field

from aana.storage.models.task import Status as TaskStatus


class TaskInfo(BaseModel):
    """Task information.

    Attributes:
        id (str): The task ID.
        status (TaskStatus): The task status.
        result (Any): The task result.
    """

    id: str = Field(..., description="The task ID.")
    status: TaskStatus = Field(..., description="The task status.")
    result: Any = Field(None, description="The task result.")


class TaskId(BaseModel):
    """Task ID.

    Attributes:
        id (str): The task ID.
    """

    task_id: str = Field(..., description="The task ID.")
