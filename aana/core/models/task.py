from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.api_service import ApiKey
from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity

TaskId = Annotated[
    str,
    Field(description="The task ID.", example="11111111-1111-1111-1111-111111111111"),
]


class TaskInfo(BaseModel):
    """Task information.

    Attributes:
        id (str): The task ID.
        endpoint (str): The endpoint to which the task is assigned.
        data (Any): The task data.
        status (TaskStatus): The task status.
        result (Any): The task result.
    """

    id: TaskId
    endpoint: str = Field(
        ..., description="The endpoint to which the task is assigned."
    )
    data: Any = Field(..., description="The task data.")
    status: TaskStatus = Field(..., description="The task status.")
    result: Any = Field(None, description="The task result.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "endpoint": "/index",
                    "data": {
                        "image": {
                            "url": "https://example.com/image.jpg",
                            "media_id": "abc123",
                        }
                    },
                    "status": "running",
                    "result": None,
                }
            ]
        },
        extra="forbid",
    )

    @classmethod
    def from_entity(cls, task: TaskEntity, is_admin: bool = False) -> "TaskInfo":
        """Create a TaskInfo from a TaskEntity."""
        # Prepare data (remove ApiKey, None values, etc.)
        task_data = {}
        for key, value in task.data.items():
            if value is None or isinstance(value, ApiKey):
                continue
            task_data[key] = value

        # Remove stacktrace from result if not admin
        if not is_admin and "stacktrace" in task.result:
            task.result.pop("stacktrace", None)

        return TaskInfo(
            id=str(task.id),
            endpoint=task.endpoint,
            data=task_data,
            status=task.status,
            result=task.result,
        )
