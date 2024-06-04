from typing import Any

from sqlalchemy.orm import Session

from aana.storage.engine import engine
from aana.storage.models.task import TaskEntity
from aana.storage.repository.task import TaskRepository


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
        The task ID.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = TaskEntity(endpoint=endpoint, data=data, priority=priority)
        task_repo.create(task)
        return task.id
