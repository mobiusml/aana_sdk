from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from aana.core.models.task import TaskInfo
from aana.storage.engine import engine
from aana.storage.models.task import Status as TaskStatus
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


def get_task(task_id: str) -> TaskEntity:
    """Get a task.

    Args:
        task_id (str): The task ID.

    Returns:
        TaskEntity: The task.

    Raises:
        NotFoundException: If the task is not found.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = task_repo.read(task_id)
        return task


def delete_task(task_id: str) -> TaskEntity:
    """Delete a task.

    Args:
        task_id (str): The task ID.

    Returns:
        TaskEntity: The deleted task.

    Raises:
        NotFoundException: If the task is not found.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = task_repo.delete(task_id, check=True)
        return task


def update_task_status(
    task_id: str, status: TaskStatus, progress: int | None = None, result: Any = None
):
    """Update the task status.

    Args:
        task_id (str): The task ID.
        status (TaskStatus): The new status.
        progress (int | None): The progress. If None, the progress will not be updated.
        result (Any): The result.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        task = task_repo.read(task_id)
        task.status = status
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            task.completed_at = datetime.now()  # noqa: DTZ005
        if progress is not None:
            task.progress = progress
        task.result = result
        session.commit()


def get_unprocessed_tasks(limit: int | None = None) -> list[TaskEntity]:
    """Fetches all unprocessed tasks.

    The task is considered unprocessed if it is in CREATED or NOT_FINISHED state or
    in RUNNING or ASSIGNED state and the update timestamp is older
    than the execution timeout (to handle stuck tasks).

    Args:
        limit (int | None): The maximum number of tasks to fetch. If None, fetch all.

    Returns:
        list[TaskEntity]: the unprocessed tasks.
    """
    with Session(engine) as session:
        task_repo = TaskRepository(session)
        return task_repo.get_unprocessed_tasks(limit)
