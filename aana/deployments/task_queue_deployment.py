import asyncio
import concurrent.futures
from datetime import datetime
from typing import Any

import orjson
from pydantic import BaseModel, Field
from ray import serve
from sqlalchemy.orm import Session

from aana.api.exception_handler import custom_exception_handler
from aana.configs.settings import settings as aana_settings
from aana.deployments.base_deployment import BaseDeployment
from aana.storage.models.task import Status as TaskStatus
from aana.storage.repository.task import TaskRepository
from aana.utils.asyncio import run_async


class TaskQueueConfig(BaseModel):
    """The configuration for the task queue deployment."""

    app_name: str = Field(description="The name of the Aana app")


@serve.deployment
class TaskQueueDeployment(BaseDeployment):
    """Deployment to serve the task queue."""

    def __init__(self):
        """Initialize the task queue deployment."""
        super().__init__()
        loop = asyncio.get_running_loop()
        self.loop_task = loop.create_task(self.loop())
        self.loop_task.add_done_callback(lambda fut: fut.result())

    def check_health(self):
        """Check the health of the deployment."""
        # if the loop is not running, the deployment is unhealthy
        if self.loop_task.done():
            raise RuntimeError("Task queue loop is not running")  # noqa: TRY003

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        The configuration should conform to the TaskQueueConfig schema.
        """
        config_obj = TaskQueueConfig(**config)
        self.handle = None
        self.app_name = config_obj.app_name
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=aana_settings.task_queue.num_workers,
        )

    async def loop(self):
        """The main loop for the task queue deployment.

        The loop will check the queue and assign tasks to workers.
        """
        from aana.storage.engine import engine

        async def handle_task(task_id: str):
            """Process a task."""
            with Session(engine) as session:
                task_repo = TaskRepository(session)
                task = task_repo.read(task_id)
                task.status = TaskStatus.RUNNING
                session.commit()
                try:
                    out = await self.handle.call_endpoint.remote(
                        task.endpoint, **task.data
                    )
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()  # noqa: DTZ005
                    task.progress = 100
                    task.result = out
                    session.commit()
                except Exception as e:
                    error_response = custom_exception_handler(None, e)
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()  # noqa: DTZ005
                    task.progress = 0
                    task.result = orjson.loads(error_response.body)
                    session.commit()

        def run_handle_task(task_id):
            """Wrapper to run the handle_task function."""
            run_async(handle_task(task_id))

        with Session(engine) as session:
            task_repo = TaskRepository(session)
            while True:
                if not self.configured:
                    # Wait for the deployment to be configured.
                    await asyncio.sleep(1)
                    continue

                tasks = task_repo.get_unprocessed_tasks(
                    limit=aana_settings.task_queue.num_workers * 2
                )
                if not tasks:
                    await asyncio.sleep(0.1)
                    continue

                if not self.handle:
                    self.handle = serve.get_app_handle(self.app_name)

                for task in tasks:
                    # Check if the thread pool has too many tasks.
                    # If so, stop assigning tasks.
                    # We do it to prevent the thread pool from being overwhelmed.
                    # We don't want to schedule all tasks from the task queue (could be millions).
                    if (
                        self.thread_pool._work_queue.qsize()
                        > aana_settings.task_queue.num_workers * 2
                    ):
                        # wait a bit to give the thread pool time to process tasks
                        await asyncio.sleep(0.1)
                        break
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_at = datetime.now()  # noqa: DTZ005
                    session.commit()
                    self.thread_pool.submit(run_handle_task, task.id)
