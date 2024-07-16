from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session

from aana.configs.settings import settings as aana_settings
from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity
from aana.storage.repository.base import BaseRepository


class TaskRepository(BaseRepository[TaskEntity]):
    """Repository for tasks."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, TaskEntity)

    def read(self, task_id: str | UUID, check: bool = True) -> TaskEntity:
        """Reads a single task by id from the database.

        Args:
            task_id (str | UUID): ID of the task to retrieve
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The corresponding task from the database if found.

        Raises:
            NotFoundException if the entity is not found and `check` is True.
        """
        if isinstance(task_id, str):
            task_id = UUID(task_id)
        return super().read(task_id, check=check)

    def delete(self, task_id: str | UUID, check: bool = False) -> TaskEntity | None:
        """Deletes a single task by id from the database.

        Args:
            task_id (str | UUID): ID of the task to delete
            check (bool): Whether to check if the task exists before deleting

        Returns:
            The deleted task from the database if found.

        Raises:
            NotFoundException: The id does not correspond to a record in the database.
        """
        if isinstance(task_id, str):
            task_id = UUID(task_id)
        return super().delete(task_id, check)

    def save(self, endpoint: str, data: Any, priority: int = 0):
        """Add a task to the database.

        Args:
            endpoint (str): The endpoint to which the task is assigned.
            data (Any): Data for the task.
            priority (int): Priority of the task (0 is the lowest).

        Returns:
            TaskEntity: The saved task.
        """
        task = TaskEntity(endpoint=endpoint, data=data, priority=priority)
        self.session.add(task)
        self.session.commit()
        return task

    def get_unprocessed_tasks(self, limit: int | None = None) -> list[TaskEntity]:
        """Fetches all unprocessed tasks.

        The task is considered unprocessed if it is in CREATED or NOT_FINISHED state or
        in RUNNING or ASSIGNED state and the update timestamp is older
        than the execution timeout (to handle stuck tasks).

        Args:
            limit (int | None): The maximum number of tasks to fetch. If None, fetch all.

        Returns:
            list[TaskEntity]: the unprocessed tasks.
        """
        execution_timeout = aana_settings.task_queue.execution_timeout
        cutoff_time = datetime.now() - timedelta(seconds=execution_timeout)  # noqa: DTZ005
        tasks = (
            self.session.query(TaskEntity)
            .filter(
                or_(
                    TaskEntity.status.in_(
                        [TaskStatus.CREATED, TaskStatus.NOT_FINISHED]
                    ),
                    and_(
                        TaskEntity.status.in_(
                            [TaskStatus.RUNNING, TaskStatus.ASSIGNED]
                        ),
                        TaskEntity.updated_at <= cutoff_time,
                    ),
                )
            )
            .order_by(desc(TaskEntity.priority), TaskEntity.created_at)
            .limit(limit)
            .all()
        )
        return tasks

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: int | None = None,
        result: Any = None,
    ):
        """Update the status of a task.

        Args:
            task_id (str): The ID of the task.
            status (TaskStatus): The new status.
            progress (int | None): The progress. If None, the progress will not be updated.
            result (Any): The result.
        """
        task = self.read(task_id)
        task.status = status
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            task.completed_at = datetime.now()  # noqa: DTZ005
        if progress is not None:
            task.progress = progress
        task.result = result
        self.session.commit()
