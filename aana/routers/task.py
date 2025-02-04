import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from aana.api.security import UserIdDependency
from aana.configs.settings import settings as aana_settings
from aana.core.models.task import TaskInfo
from aana.storage.models.task import Status as TaskStatus
from aana.storage.repository.task import TaskRepository
from aana.storage.session import GetDbDependency

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"], include_in_schema=aana_settings.task_queue.enabled)

# Request models


class TaskListRequest(BaseModel):
    """Request to list tasks."""

    status: TaskStatus | None = Field(
        None, description="Filter tasks by status. If None, all tasks are returned."
    )
    limit: int = Field(100, description="The maximum number of tasks to return.")
    offset: int = Field(
        0, description="The number of tasks to skip before starting to return tasks."
    )


# Response models


class TaskList(BaseModel):
    """Response for a list of tasks."""

    tasks: list[TaskInfo] = Field(..., description="The list of tasks.")


# fmt: off
class TaskCount(BaseModel):
    """Response for a count of tasks by status."""

    created: int | None = Field(None, description="The number of tasks in the CREATED status.")
    assigned: int | None = Field(None, description="The number of tasks in the ASSIGNED status.")
    completed: int | None = Field(None, description="The number of tasks in the COMPLETED status.")
    running: int | None = Field(None, description="The number of tasks in the RUNNING status.")
    failed: int | None = Field(None, description="The number of tasks in the FAILED status.")
    not_finished: int | None = Field(None, description="The number of tasks in the NOT_FINISHED status.")
    total: int = Field(..., description="The total number of tasks.")
# fmt: on

# Endpoints


@router.get(
    "/tasks/count",
    summary="Count Tasks",
    description="Count tasks per status.",
    response_model_exclude_none=True,
)
async def count_tasks(db: GetDbDependency, user_id: UserIdDependency) -> TaskCount:
    """Count tasks by status."""
    task_repo = TaskRepository(db)
    counts = task_repo.count(user_id=user_id)
    return TaskCount(**counts)


@router.get(
    "/tasks/{task_id}",
    summary="Get Task Status",
    description="Get the task status by task ID.",
)
async def get_task(
    task_id: str, db: GetDbDependency, user_id: UserIdDependency
) -> TaskInfo:
    """Get the task with the given ID."""
    task_repo = TaskRepository(db)
    task = task_repo.read(task_id, check=False)
    if not task or task.user_id != user_id:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )
    return TaskInfo.from_entity(task)


@router.get(
    "/tasks",
    summary="List Tasks",
    description="List all tasks.",
)
async def list_tasks(
    db: GetDbDependency,
    user_id: UserIdDependency,
    status: Annotated[
        TaskStatus | None,
        Field(description="Filter tasks by status. If None, all tasks are returned."),
    ] = None,
    page: Annotated[int, Field(description="The page number.")] = 1,
    per_page: Annotated[int, Field(description="The number of tasks per page.")] = 100,
) -> TaskList:
    """List all tasks."""
    task_repo = TaskRepository(db)
    tasks = task_repo.get_tasks(
        user_id=user_id, status=status, limit=per_page, offset=(page - 1) * per_page
    )
    return TaskList(tasks=[TaskInfo.from_entity(task) for task in tasks])


@router.delete(
    "/tasks/{task_id}",
    summary="Delete Task",
    description="Delete the task by task ID.",
)
async def delete_task(
    task_id: str, db: GetDbDependency, user_id: UserIdDependency
) -> TaskInfo:
    """Delete the task with the given ID."""
    task_repo = TaskRepository(db)
    task = task_repo.read(task_id, check=False)
    if not task or task.user_id != user_id:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )
    task = task_repo.delete(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )
    return TaskInfo.from_entity(task)


# Legacy endpoints (to be removed in the future)


@router.get(
    "/tasks/get/{task_id}",
    summary="Get Task Status (Legacy)",
    description="Get the task status by task ID (Legacy endpoint).",
    deprecated=True,
)
async def get_task_legacy(
    task_id: str, db: GetDbDependency, user_id: UserIdDependency
) -> TaskInfo:
    """Get the task with the given ID (Legacy endpoint)."""
    task_repo = TaskRepository(db)
    task = task_repo.read(task_id)
    if not task or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskInfo.from_entity(task)


class TaskId(BaseModel):
    """Task ID (Legacy).

    Attributes:
        id (str): The task ID.
    """

    task_id: str = Field(..., description="The task ID.")


@router.get(
    "/tasks/delete/{task_id}",
    summary="Delete Task (Legacy)",
    description="Delete the task by task ID (Legacy endpoint).",
    deprecated=True,
)
async def delete_task_legacy(
    task_id: str, db: GetDbDependency, user_id: UserIdDependency
) -> TaskId:
    """Delete the task with the given ID (Legacy endpoint)."""
    task_repo = TaskRepository(db)
    task = task_repo.read(task_id)
    if not task or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    task = task_repo.delete(task_id)
    return TaskId(task_id=str(task.id))
