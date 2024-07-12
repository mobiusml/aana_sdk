import asyncio
import concurrent.futures
from typing import Any

import orjson
import ray
from pydantic import BaseModel, Field
from ray import serve

from aana.api.exception_handler import custom_exception_handler
from aana.configs.settings import settings as aana_settings
from aana.deployments.base_deployment import BaseDeployment
from aana.storage.models.task import Status as TaskStatus
from aana.storage.services.task import (
    get_task,
    get_unprocessed_tasks,
    update_task_status,
)
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
        self.futures = {}
        loop = asyncio.get_running_loop()
        self.loop_task = loop.create_task(self.loop())
        self.loop_task.add_done_callback(
            lambda fut: fut.result() if not fut.cancelled() else None
        )

    def check_health(self):
        """Check the health of the deployment."""
        # if the loop is not running, the deployment is unhealthy
        if self.loop_task.done():
            raise RuntimeError("Task queue loop is not running")  # noqa: TRY003

    def __del__(self):
        """Clean up the deployment."""
        # Cancel the loop task to prevent tasks from being reassigned
        self.loop_task.cancel()
        # Set all non-completed tasks to NOT_FINISHED
        for task_id, future in self.futures.items():
            if not future.done():
                update_task_status(task_id, TaskStatus.NOT_FINISHED, 0)

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

    async def loop(self):  # noqa: C901
        """The main loop for the task queue deployment.

        The loop will check the queue and assign tasks to workers.
        """

        async def handle_task(task_id: str):
            """Process a task."""
            # Fetch the task details
            task = get_task(task_id)
            # Initially set the task status to RUNNING
            update_task_status(task_id, TaskStatus.RUNNING, 0)
            try:
                # Call the endpoint asynchronously
                out = await self.handle.call_endpoint.remote(task.endpoint, **task.data)
                # Update the task status to COMPLETED
                update_task_status(task_id, TaskStatus.COMPLETED, 100, out)
            except Exception as e:
                # Handle the exception and update the task status to FAILED
                error_response = custom_exception_handler(None, e)
                error = orjson.loads(error_response.body)
                update_task_status(task_id, TaskStatus.FAILED, 0, error)

        def run_handle_task(task_id):
            """Wrapper to run the handle_task function."""
            run_async(handle_task(task_id))

        def is_thread_pool_full():
            """Check if the thread pool has too many tasks.

            We use it to stop assigning tasks to the thread pool if it's full
            to prevent the thread pool from being overwhelmed.
            We don't want to schedule all tasks from the task queue (could be millions).
            """
            return (
                self.thread_pool._work_queue.qsize()
                > aana_settings.task_queue.num_workers * 2
            )

        while True:
            if not self.configured:
                # Wait for the deployment to be configured.
                await asyncio.sleep(1)
                continue

            # Remove completed tasks from the futures dictionary
            for task_id in list(self.futures.keys()):
                if self.futures[task_id].done():
                    del self.futures[task_id]

            if is_thread_pool_full():
                # wait a bit to give the thread pool time to process tasks
                await asyncio.sleep(0.1)
                continue

            tasks = get_unprocessed_tasks(
                limit=aana_settings.task_queue.num_workers * 2
            )

            if not tasks:
                await asyncio.sleep(0.1)
                continue

            if not self.handle:
                # Sometimes the app isn't available immediately after the deployment is created
                # so we need to wait for it to become available
                for _ in range(10):
                    try:
                        self.handle = serve.get_app_handle(self.app_name)
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
                    self.handle = serve.get_app_handle(self.app_name)

            for task in tasks:
                if is_thread_pool_full():
                    # wait a bit to give the thread pool time to process tasks
                    await asyncio.sleep(0.1)
                    break
                update_task_status(task.id, TaskStatus.ASSIGNED, 0)
                future = self.thread_pool.submit(run_handle_task, task.id)
                self.futures[task.id] = future
