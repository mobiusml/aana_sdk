from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity
from aana.storage.repository.base import BaseRepository


class TaskRepository(BaseRepository[TaskEntity]):
    """Repository for tasks."""

    def __init__(self, session: AsyncSession):
        """Constructor."""
        super().__init__(session, TaskEntity)

    async def read(self, task_id: str | UUID, check: bool = True) -> TaskEntity | None:
        """Reads a single task by id from the database.

        Args:
            task_id (str | UUID): ID of the task to retrieve
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The corresponding task from the database if found.

        Raises:
            NotFoundException if the entity is not found and `check` is True.
        """
        task_id = self._convert_to_uuid(task_id)
        if task_id is None:
            return None
        return await super().read(task_id, check=check)

    async def delete(
        self, task_id: str | UUID, check: bool = False
    ) -> TaskEntity | None:
        """Deletes a single task by id from the database.

        Args:
            task_id (str | UUID): ID of the task to delete
            check (bool): Whether to check if the task exists before deleting

        Returns:
            The deleted task from the database if found.

        Raises:
            NotFoundException: The id does not correspond to a record in the database.
        """
        task_id = self._convert_to_uuid(task_id)
        if task_id is None:
            return None
        return await super().delete(task_id, check)

    async def save(
        self, endpoint: str, data: Any, user_id: str | None = None, priority: int = 0
    ):
        """Add a task to the database.

        Args:
            endpoint (str): The endpoint to which the task is assigned.
            data (Any): Data for the task.
            user_id (str | None): The ID of the user who created the task.
            priority (int): Priority of the task (0 is the lowest).

        Returns:
            TaskEntity: The saved task.
        """
        task = TaskEntity(
            endpoint=endpoint, data=data, priority=priority, user_id=user_id
        )
        self.session.add(task)
        await self.session.commit()
        return task

    async def fetch_unprocessed_tasks(
        self,
        limit: int | None = None,
        max_retries: int = 1,
        retryable_exceptions: list[str] | None = None,
        api_service_enabled: bool = False,
        maximum_active_tasks_per_user: int = 25,
    ) -> list[TaskEntity]:
        """Fetches unprocessed tasks and marks them as ASSIGNED.

        The task is considered unprocessed if it is in CREATED or NOT_FINISHED state.

        The function runs in a transaction and locks the rows to prevent race condition
        if multiple task queue deployments are running concurrently.

        IMPORTANT: The lock doesn't work with SQLite. If you are using SQLite, you should
        only run one task queue deployment at a time. Otherwise, you may encounter
        race conditions.

        Args:
            limit (int | None): The maximum number of tasks to fetch. If None, fetch all.
            max_retries (int): The maximum number of retries for a task.
            retryable_exceptions (list[str] | None): The list of exceptions that should be retried.
            api_service_enabled (bool): Whether the API service is enabled. If True, the tasks will be
                assigned to users with active subscriptions only. Defaults to False.
            maximum_active_tasks_per_user (int): The maximum number of active tasks per user.

        Returns:
            list[TaskEntity]: the unprocessed tasks.
        """
        if retryable_exceptions:
            # Convert the list of exceptions to a string for the query:
            # e.g., ["InferenceException", "ValueError"] -> "'InferenceException', 'ValueError'"
            exceptions_str = ", ".join([f"'{ex}'" for ex in retryable_exceptions])
            if self.session.bind.dialect.name == "postgresql":
                exception_name_query = f"result->>'error' IN ({exceptions_str})"
            elif self.session.bind.dialect.name == "sqlite":
                exception_name_query = (
                    f"json_extract(result, '$.error') IN ({exceptions_str})"
                )
            else:
                raise NotImplementedError(
                    f"Filtering by exception name is not supported for {self.session.bind.dialect.name}"
                )

            main_filter_expr = or_(
                TaskEntity.status.in_([TaskStatus.CREATED, TaskStatus.NOT_FINISHED]),
                and_(
                    TaskEntity.status == TaskStatus.FAILED,
                    text(exception_name_query),
                    TaskEntity.num_retries < max_retries,
                ),
            )
        else:
            main_filter_expr = TaskEntity.status.in_(
                [TaskStatus.CREATED, TaskStatus.NOT_FINISHED]
            )

        if api_service_enabled:
            # Fetch active users with active subscriptions from the API service database
            stmt = select(ApiKeyEntity.user_id).where(
                ApiKeyEntity.is_subscription_active == True
            )
            result = await self.session.execute(stmt)
            active_user_ids = [row[0] for row in result.all()]

            if not active_user_ids:
                return []

            # Subquery to count the number of active tasks for each user
            TaskAlias = aliased(TaskEntity)
            active_task_count = (
                select(func.count(TaskAlias.id))
                .where(
                    # Match user_id (including None) between outer task and subquery
                    TaskAlias.user_id == TaskEntity.user_id,
                    TaskAlias.status.in_([TaskStatus.RUNNING, TaskStatus.ASSIGNED]),
                )
                .correlate(TaskEntity)  # Correlate with the outer TaskEntity
                .scalar_subquery()
            )

            # Filter expression to prioritize users with fewer active tasks
            user_filter_expr = or_(
                TaskEntity.user_id.in_(active_user_ids),
                TaskEntity.user_id == None,
            )

            # Window function to enumerate tasks per user by priority and created_at
            window_stmt = (
                select(
                    TaskEntity.id.label("task_id"),
                    TaskEntity.user_id.label("user_id"),
                    TaskEntity.priority.label("priority"),
                    TaskEntity.created_at.label("created_at"),
                    # "How many tasks are active for the same user?"
                    active_task_count.label("active_count"),
                    # row_number() to limit tasks per user by priority
                    func.row_number()
                    .over(
                        partition_by=TaskEntity.user_id,
                        order_by=(desc(TaskEntity.priority), TaskEntity.created_at),
                    )
                    .label("row_num"),
                )
                .where(and_(main_filter_expr, user_filter_expr))
                .subquery()
            )

            # Query to fetch tasks with limited active tasks per user
            candidate_ids_stmt = (
                select(window_stmt.c.task_id)
                .where(
                    or_(
                        window_stmt.c.active_count + window_stmt.c.row_num
                        <= maximum_active_tasks_per_user,
                        window_stmt.c.user_id == None,
                    )
                )
                .order_by(
                    desc(window_stmt.c.priority),
                    window_stmt.c.created_at,
                    window_stmt.c.row_num,
                )
                .limit(limit)
            )

            # Fetch the task IDs from the query
            result = await self.session.execute(candidate_ids_stmt)
            candidate_ids = [row.task_id for row in result.all()]

            if not candidate_ids:
                # No tasks matched
                return []

            # Lock those specific tasks with FOR UPDATE SKIP LOCKED
            stmt = (
                select(TaskEntity)
                .where(TaskEntity.id.in_(candidate_ids))
                .with_for_update(skip_locked=True)
            )
            result = await self.session.execute(stmt)
            tasks = result.scalars().all()
        else:
            stmt = (
                select(TaskEntity)
                .where(main_filter_expr)
                .order_by(desc(TaskEntity.priority), TaskEntity.created_at)
                .limit(limit)
                .execution_options(populate_existing=True)
                .with_for_update(skip_locked=True)
            )
            result = await self.session.execute(stmt)
            tasks = result.scalars().all()

        for task in tasks:
            await self.update_status(
                task_id=task.id,
                status=TaskStatus.ASSIGNED,
                progress=0,
                commit=False,
            )
        await self.session.commit()
        return tasks

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: int | None = None,
        result: Any = None,
        commit: bool = True,
    ) -> TaskEntity:
        """Update the status of a task.

        Args:
            task_id (str): The ID of the task.
            status (TaskStatus): The new status.
            progress (int | None): The progress. If None, the progress will not be updated.
            result (Any): The result.
            commit (bool): Whether to commit the transaction.

        Returns:
            TaskEntity: The updated task.
        """
        task = await self.read(task_id)
        if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
            task.completed_at = datetime.now(timezone.utc)
        if status == TaskStatus.ASSIGNED:
            task.assigned_at = datetime.now(timezone.utc)
            task.num_retries += 1
        if progress is not None:
            task.progress = progress
        task.status = status
        task.result = result
        if commit:
            await self.session.commit()
        return task

    async def retry_task(self, task_id: str) -> TaskEntity:
        """Retry a task. The task will reset to CREATED status.

        Args:
            task_id (str): The ID of the task.

        Returns:
            TaskEntity: The updated task.
        """
        task = await self.read(task_id)
        task.status = TaskStatus.CREATED
        task.progress = 0
        task.result = None
        task.num_retries = 0
        task.assigned_at = None
        task.completed_at = None
        await self.session.commit()
        return task

    async def get_active_tasks(self) -> list[TaskEntity]:
        """Fetches all active tasks.

        The task is considered active if it is in RUNNING or ASSIGNED state.

        Returns:
            list[TaskEntity]: the active tasks.
        """
        stmt = select(TaskEntity).where(
            TaskEntity.status.in_([TaskStatus.RUNNING, TaskStatus.ASSIGNED])
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def filter_incomplete_tasks(self, task_ids: list[str]) -> list[str]:
        """Remove the task IDs that are already completed (COMPLETED or FAILED).

        Args:
            task_ids (list[str]): The task IDs to filter.

        Returns:
            list[str]: The task IDs that are not completed.
        """
        task_ids = [UUID(task_id) for task_id in task_ids]
        stmt = select(TaskEntity).where(
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
        result = await self.session.execute(stmt)
        tasks = result.scalars().all()
        incomplete_task_ids = [str(task.id) for task in tasks]
        return incomplete_task_ids

    async def update_expired_tasks(
        self, execution_timeout: float, heartbeat_timeout: float, max_retries: int
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
            heartbeat_timeout (float): The maximum time since the last heartbeat in seconds
            max_retries (int): The maximum number of retries for a task

        Returns:
            list[TaskEntity]: the expired tasks.
        """
        timeout_cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=execution_timeout
        )
        heartbeat_cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=heartbeat_timeout
        )

        stmt = (
            select(TaskEntity)
            .where(
                and_(
                    TaskEntity.status.in_([TaskStatus.RUNNING, TaskStatus.ASSIGNED]),
                    or_(
                        TaskEntity.assigned_at <= timeout_cutoff,
                        TaskEntity.updated_at <= heartbeat_cutoff,
                    ),
                )
            )
            .with_for_update(skip_locked=True)
        )

        result = await self.session.execute(stmt)
        tasks = result.scalars().all()

        for task in tasks:
            if task.num_retries >= max_retries:
                if task.assigned_at.astimezone(timezone.utc) <= timeout_cutoff:
                    result = {
                        "error": "TimeoutError",
                        "message": (
                            f"Task execution timed out after {execution_timeout} seconds and "
                            f"exceeded the maximum number of retries ({max_retries})"
                        ),
                    }
                else:
                    result = {
                        "error": "HeartbeatTimeoutError",
                        "message": (
                            f"The task has not received a heartbeat for {heartbeat_timeout} seconds and "
                            f"exceeded the maximum number of retries ({max_retries})"
                        ),
                    }

                await self.update_status(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    progress=0,
                    result=result,
                    commit=False,
                )
            else:
                await self.update_status(
                    task_id=task.id,
                    status=TaskStatus.NOT_FINISHED,
                    progress=0,
                    commit=False,
                )

        await self.session.commit()
        return tasks

    async def heartbeat(self, task_ids: list[str] | set[str]):
        """Updates the updated_at timestamp for multiple tasks.

        Args:
            task_ids (list[str] | set[str]): List or set of task IDs to update
        """
        task_ids = [
            UUID(task_id) if isinstance(task_id, str) else task_id
            for task_id in task_ids
        ]

        stmt = select(TaskEntity).where(TaskEntity.id.in_(task_ids))

        result = await self.session.execute(stmt)
        tasks = result.scalars().all()

        for task in tasks:
            task.updated_at = datetime.now(timezone.utc)

        await self.session.commit()

    async def get_tasks(
        self,
        user_id: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TaskEntity]:
        """Get tasks by user_id and status.

        Args:
            user_id (str | None): The user ID.
            status (TaskStatus | None): The task status.
            limit (int): The maximum number of tasks to fetch.
            offset (int): The offset.

        Returns:
            list[TaskEntity]: The list of tasks.
        """
        stmt = select(TaskEntity)
        if user_id:
            stmt = stmt.where(TaskEntity.user_id == user_id)
        if status:
            stmt = stmt.where(TaskEntity.status == status)

        stmt = stmt.order_by(TaskEntity.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, user_id: str | None = None) -> dict[str, int]:
        """Count tasks by status.

        Args:
            user_id (str | None): The user ID. If None, all tasks are counted.

        Returns:
            dict[str, int]: The count of tasks by status.
        """
        base_stmt = select(TaskEntity.status, func.count(TaskEntity.id))
        if user_id:
            base_stmt = base_stmt.where(TaskEntity.user_id == user_id)

        stmt = base_stmt.group_by(TaskEntity.status)
        result = await self.session.execute(stmt)
        counts = result.all()

        count_dict = {status.value: count for status, count in counts}
        count_dict["total"] = sum(count_dict.values())
        return count_dict
