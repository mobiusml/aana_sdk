from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from aana.storage.engine import engine
from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity
from aana.storage.repository.task import TaskRepository


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


def create_task(
    endpoint: str,
    data: Any,
    priority: int = 0,
) -> str:
    """Create a task.

    Args:
        endpoint: The endpoint to which the task is assigned.
        data: Data for the task.
        priority: Priority of the task (0 is the lowest).

    Returns:
        str: The task ID.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = TaskEntity(endpoint=endpoint, data=data, priority=priority)
        task_repo.create(task)
        return str(task.id)


def get_task_info(task_id: str) -> TaskInfo:
    """Get a task info.

    Args:
        task_id (str): The task ID.

    Returns:
        TaskStatus: The task status.

    Raises:
        NotFoundException: If the task is not found.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = task_repo.read(task_id)
        return TaskInfo(
            id=str(task.id),
            status=task.status,
            result=task.result,
        )
