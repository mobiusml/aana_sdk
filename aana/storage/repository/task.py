from sqlalchemy.orm import Session

from aana.storage.models.task import TaskEntity
from aana.storage.repository.base import BaseRepository


class TaskRepository(BaseRepository[TaskEntity]):
    """Repository for tasks."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, TaskEntity)

    def get_by_task_id(self, task_id: str) -> TaskEntity:
        """Fetches a task by task_id.

        Args:
            task_id (str): Task ID to query.

        Returns:
            TaskEntity: the task.
        """
        return self.read(task_id)
