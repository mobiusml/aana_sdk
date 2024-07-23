from typing import Any

from sqlalchemy.orm import Session

from aana.storage.engine import engine
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
        str: The task ID.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = task_repo.save(endpoint=endpoint, data=data, priority=priority)
        return str(task.id)
