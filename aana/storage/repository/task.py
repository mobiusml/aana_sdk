from datetime import datetime, timedelta

from sqlalchemy import and_, or_
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

    def get_unprocessed_tasks(self) -> list[TaskEntity]:
        """Fetches all unprocessed tasks.

        The task is considered unprocessed if it is in CREATED state or
        in RUNNING or ASSIGNED state and the update timestamp is older
        than the execution timeout (to handle stuck tasks).

        Returns:
            list[TaskEntity]: the unprocessed tasks.
        """
        execution_timeout = aana_settings.task_queue.execution_timeout
        cutoff_time = datetime.now() - timedelta(seconds=execution_timeout)  # noqa: DTZ005
        tasks = (
            self.session.query(TaskEntity)
            .filter(
                or_(
                    TaskEntity.status == TaskStatus.CREATED,
                    and_(
                        TaskEntity.status.in_(
                            [TaskStatus.RUNNING, TaskStatus.ASSIGNED]
                        ),
                        TaskEntity.update_ts <= cutoff_time,
                    ),
                )
            )
            .order_by(TaskEntity.priority, TaskEntity.create_ts)
            .all()
        )
        return tasks