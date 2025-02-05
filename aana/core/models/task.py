from typing import Any

from pydantic import BaseModel, Field

from aana.core.models.api_service import ApiKey
from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity


class TaskInfo(BaseModel):
    """Task information.

    Attributes:
        id (str): The task ID.
        endpoint (str): The endpoint to which the task is assigned.
        data (Any): The task data.
        status (TaskStatus): The task status.
        result (Any): The task result.
    """

    id: str = Field(..., description="The task ID.")
    endpoint: str = Field(
        ..., description="The endpoint to which the task is assigned."
    )
    data: Any = Field(..., description="The task data.")
    status: TaskStatus = Field(..., description="The task status.")
    result: Any = Field(None, description="The task result.")

    @classmethod
    def from_entity(cls, task: TaskEntity) -> "TaskInfo":
        """Create a TaskInfo from a TaskEntity."""
        # Prepare data (remove ApiKey, None values, etc.)
        task_data = {}
        for key, value in task.data.items():
            if value is None or isinstance(value, ApiKey):
                continue
            task_data[key] = value

        return TaskInfo(
            id=str(task.id),
            endpoint=task.endpoint,
            data=task_data,
            status=task.status,
            result=task.result,
        )
