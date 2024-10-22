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
    """Deployment to serve the task queue.

    IMPORTANT: If you are using SQLite, make sure to run only one instance
    of this deployment to avoid database race conditions.
    """

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
        self.running_task_ids: list[str] = []
        self.deployment_responses = {}

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
        # Cancel all deployment responses to stop the tasks
        # and set all non-completed tasks to NOT_FINISHED
        for task_id in self.running_task_ids:
            deployment_response = self.deployment_responses.get(task_id)
            if deployment_response:
                deployment_response.cancel()
            self.task_repo.update_status(
                task_id=task_id,
                status=TaskStatus.NOT_FINISHED,
                progress=0,
            )

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
        handle = None

        while True:
            if not self._configured:
                # Wait for the deployment to be configured.
                await asyncio.sleep(1)
                continue

            # Remove completed tasks from the list of running tasks
            self.running_task_ids = self.task_repo.filter_incomplete_tasks(
                self.running_task_ids
            )

            # Check for expired tasks
            execution_timeout = aana_settings.task_queue.execution_timeout
            max_retries = aana_settings.task_queue.max_retries
            expired_tasks = self.task_repo.update_expired_tasks(
                execution_timeout=execution_timeout, max_retries=max_retries
            )
            for task in expired_tasks:
                deployment_response = self.deployment_responses.get(task.id)
                if deployment_response:
                    deployment_response.cancel()

            # If the queue is full, wait and retry
            if len(self.running_task_ids) >= aana_settings.task_queue.num_workers:
                await asyncio.sleep(0.1)
                continue

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

            # Get new tasks from the database
            num_tasks_to_assign = aana_settings.task_queue.num_workers - len(
                self.running_task_ids
            )
            tasks = self.task_repo.fetch_unprocessed_tasks(limit=num_tasks_to_assign)

            # If there are no tasks, wait and retry
            if not tasks:
                await asyncio.sleep(0.1)
                continue

            # Start processing the tasks
            for task in tasks:
                deployment_response = handle.execute_task.remote(task_id=task.id)
                self.deployment_responses[task.id] = deployment_response
                self.running_task_ids.append(str(task.id))
