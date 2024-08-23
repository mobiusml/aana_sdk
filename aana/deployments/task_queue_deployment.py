import asyncio
from typing import Any

import ray
from pydantic import BaseModel, Field
from ray import serve

from aana.configs.settings import settings as aana_settings
from aana.deployments.base_deployment import BaseDeployment
from aana.storage.models.task import Status as TaskStatus
from aana.storage.repository.task import TaskRepository
from aana.storage.session import get_session


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
        self.loop_task.add_done_callback(
            lambda fut: fut.result() if not fut.cancelled() else None
        )
        self.session = get_session()
        self.task_repo = TaskRepository(self.session)

    def check_health(self):
        """Check the health of the deployment."""
        # if the loop is not running, the deployment is unhealthy
        if self.loop_task.done():
            raise RuntimeError(  # noqa: TRY003
                "Task queue loop is not running"
            ) from self.loop_task.exception()

    def __del__(self):
        """Clean up the deployment."""
        # Cancel the loop task to prevent tasks from being reassigned
        self.loop_task.cancel()
        # Set all non-completed tasks to NOT_FINISHED
        for task_id, future in self.futures.items():
            if not future.done():
                self.task_repo.update_status(task_id, TaskStatus.NOT_FINISHED, 0)

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        The configuration should conform to the TaskQueueConfig schema.
        """
        config_obj = TaskQueueConfig(**config)
        self.app_name = config_obj.app_name

    async def loop(self):  # noqa: C901
        """The main loop for the task queue deployment.

        The loop will check the queue and assign tasks to workers.
        """
        running_task_ids: list[str] = []
        handle = None

        active_tasks = self.task_repo.get_active_tasks()
        for task in active_tasks:
            if task.status == TaskStatus.RUNNING:
                running_task_ids.append(task.id)
            if task.status == TaskStatus.ASSIGNED:
                self.task_repo.update_status(
                    task_id=task.id,
                    status=TaskStatus.NOT_FINISHED,
                    progress=0,
                )

        while True:
            if not self._configured:
                # Wait for the deployment to be configured.
                await asyncio.sleep(1)
                continue

            # Remove completed tasks from the list of running tasks
            running_task_ids = self.task_repo.remove_completed_tasks(running_task_ids)

            # TODO: Add task timeout handling

            # If the queue is full, wait and retry
            if len(running_task_ids) >= aana_settings.task_queue.num_workers:
                await asyncio.sleep(0.1)
                continue

            # Get new tasks from the database
            num_tasks_to_assign = aana_settings.task_queue.num_workers - len(
                running_task_ids
            )
            tasks = self.task_repo.get_unprocessed_tasks(limit=num_tasks_to_assign)

            # If there are no tasks, wait and retry
            if not tasks:
                await asyncio.sleep(0.1)
                continue
            # else:
            #     print(f"num_tasks_to_assign: {num_tasks_to_assign}, tasks: {tasks}")

            if not handle:
                # Sometimes the app isn't available immediately after the deployment is created
                # so we need to wait for it to become available
                for _ in range(10):
                    try:
                        handle = serve.get_app_handle(self.app_name)
                        break
                    except ray.serve.exceptions.RayServeException as e:
                        print(
                            f"App {self.app_name} not available yet: {e}, retrying..."
                        )
                        await asyncio.sleep(1)
                else:
                    # If the app is not available after all retries, try again
                    # but without catching the exception
                    # (if it fails, the deployment will be unhealthy, and restart will be attempted)
                    handle = serve.get_app_handle(self.app_name)

            # Start processing the tasks
            for task in tasks:
                self.task_repo.update_status(
                    task_id=task.id,
                    status=TaskStatus.ASSIGNED,
                    progress=0,
                )
                handle.execute_task.remote(task_id=task.id)
                running_task_ids.append(str(task.id))
