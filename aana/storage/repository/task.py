from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

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

    def fetch_unprocessed_tasks(self, limit: int | None = None) -> list[TaskEntity]:
        """Fetches unprocessed tasks and marks them as ASSIGNED.

        The task is considered unprocessed if it is in CREATED or NOT_FINISHED state.

        The function runs in a transaction and locks the rows to prevent race condition
        if multiple task queue deployments are running concurrently.

        IMPORTANT: The lock doesn't work with SQLite. If you are using SQLite, you should
        only run one task queue deployment at a time. Otherwise, you may encounter
        race conditions.

        Args:
            limit (int | None): The maximum number of tasks to fetch. If None, fetch all.

        Returns:
            list[TaskEntity]: the unprocessed tasks.
        """
        tasks = (
            self.session.query(TaskEntity)
            .filter(
                TaskEntity.status.in_([TaskStatus.CREATED, TaskStatus.NOT_FINISHED])
            )
            .order_by(desc(TaskEntity.priority), TaskEntity.created_at)
            .limit(limit)
            .populate_existing()
            .with_for_update(skip_locked=True)
            .all()
        )
        for task in tasks:
            self.update_status(
                task_id=task.id,
                status=TaskStatus.ASSIGNED,
                progress=0,
                commit=False,
            )
        self.session.commit()
        return tasks

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: int | None = None,
        result: Any = None,
        commit: bool = True,
    ):
        """Update the status of a task.

        Args:
            task_id (str): The ID of the task.
            status (TaskStatus): The new status.
            progress (int | None): The progress. If None, the progress will not be updated.
            result (Any): The result.
            commit (bool): Whether to commit the transaction.
        """
        task = self.read(task_id)
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            task.completed_at = datetime.now()  # noqa: DTZ005
        if status == TaskStatus.ASSIGNED:
            task.assigned_at = datetime.now()  # noqa: DTZ005
            task.num_retries += 1
        if progress is not None:
            task.progress = progress
        task.status = status
        task.result = result
        if commit:
            self.session.commit()

    def get_active_tasks(self) -> list[TaskEntity]:
        """Fetches all active tasks.

        The task is considered active if it is in RUNNING or ASSIGNED state.

        Returns:
            list[TaskEntity]: the active tasks.
        """
        tasks = (
            self.session.query(TaskEntity)
            .filter(TaskEntity.status.in_([TaskStatus.RUNNING, TaskStatus.ASSIGNED]))
            .all()
        )
        return tasks

    def filter_incomplete_tasks(self, task_ids: list[str]) -> list[str]:
        """Remove the task IDs that are already completed (COMPLETED or FAILED).

        Args:
            task_ids (list[str]): The task IDs to filter.

        Returns:
            list[str]: The task IDs that are not completed.
        """
        task_ids = [UUID(task_id) for task_id in task_ids]
        tasks = (
            self.session.query(TaskEntity)
            .filter(
                and_(
                    TaskEntity.id.in_(task_ids),
                    TaskEntity.status.not_in(
                        [
                            TaskStatus.COMPLETED,
                            TaskStatus.FAILED,
                            TaskStatus.NOT_FINISHED,
                        ]
                    ),
                )
            )
            .all()
        )
        incomplete_task_ids = [str(task.id) for task in tasks]
        return incomplete_task_ids

    def update_expired_tasks(
        self, execution_timeout: float, max_retries: int
    ) -> list[TaskEntity]:
        """Fetches all tasks that are expired and updates their status.

        The task is considered expired if it is in RUNNING or ASSIGNED state and the
        updated_at time is older than the execution_timeout.

        If the task has exceeded the maximum number of retries, it will be marked as FAILED.
        If the task has not exceeded the maximum number of retries, it will be marked as NOT_FINISHED and
        be retried again.

        The function runs in a transaction and locks the rows to prevent race condition
        if multiple task queue deployments are running concurrently.

        IMPORTANT: The lock doesn't work with SQLite. If you are using SQLite, you should
        only run one task queue deployment at a time. Otherwise, you may encounter
        race conditions.

        Args:
            execution_timeout (float): The maximum execution time for a task in seconds
            max_retries (int): The maximum number of retries for a task

        Returns:
            list[TaskEntity]: the expired tasks.
        """
        cutoff_time = datetime.now() - timedelta(seconds=execution_timeout)  # noqa: DTZ005
        tasks = (
            self.session.query(TaskEntity)
            .filter(
                and_(
                    TaskEntity.status.in_([TaskStatus.RUNNING, TaskStatus.ASSIGNED]),
                    TaskEntity.updated_at <= cutoff_time,
                ),
            )
            .populate_existing()
            .with_for_update(skip_locked=True)
            .all()
        )
        for task in tasks:
            if task.num_retries >= max_retries:
                self.update_status(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    progress=0,
                    result={
                        "error": "TimeoutError",
                        "message": (
                            f"Task execution timed out after {execution_timeout} seconds and "
                            f"exceeded the maximum number of retries ({max_retries})"
                        ),
                    },
                    commit=False,
                )
            else:
                self.update_status(
                    task_id=task.id,
                    status=TaskStatus.NOT_FINISHED,
                    progress=0,
                    commit=False,
                )
        self.session.commit()
        return tasks
