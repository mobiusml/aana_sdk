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
from aana.utils.core import sleep_exponential_backoff


class TaskQueueConfig(BaseModel):
    """The configuration for the task queue deployment."""

    app_name: str = Field(description="The name of the Aana app")
    retryable_exceptions: list[str] = Field(
        description="The list of exceptions that should be retried"
    )


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
            with get_session() as session:
                TaskRepository(session).update_status(
                    task_id=task_id,
                    status=TaskStatus.NOT_FINISHED,
                    progress=0,
                )

    async def app_health_check(self) -> bool:
        """Check the health of the app.

        The app is considered healthy if for every deployment, at least 50% of the replicas are running.
        The reason for this is that even if some replicas are not running, the app can still can process requests.
        And in the cluster setup, it is possible that some replicas on other nodes are not running or just starting up
        and it is not a reason to consider the app unhealthy.

        Returns:
            bool: True if the app is healthy, False otherwise
        """
        serve_status = serve.status()
        for app in serve_status.applications.values():
            for deployment in app.deployments.values():
                num_replicas = sum(deployment.replica_states.values())
                if num_replicas > 0:
                    health_ratio = (
                        deployment.replica_states.get("RUNNING", 0) / num_replicas
                    )
                    if health_ratio < 0.5:
                        return False
        return True

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        The configuration should conform to the TaskQueueConfig schema.
        """
        config_obj = TaskQueueConfig(**config)
        self.app_name = config_obj.app_name
        self.retryable_exceptions = config_obj.retryable_exceptions

    async def loop(self):  # noqa: C901
        """The main loop for the task queue deployment.

        The loop will check the queue and assign tasks to workers.
        """
        handle = None
        app_health_check_attempts = 0
        configuration_attempts = 0
        full_queue_attempts = 0
        no_tasks_attempts = 0
        no_app_attempts = 0

        while True:
            # Check the health of the app
            app_health = await self.app_health_check()
            if not app_health:
                # If the app is not healthy, wait and retry
                await sleep_exponential_backoff(1.0, 5.0, app_health_check_attempts)
                app_health_check_attempts += 1
                continue
            else:
                app_health_check_attempts = 0

            if not self._configured:
                # Wait for the deployment to be configured.
                await sleep_exponential_backoff(1.0, 5.0, configuration_attempts)
                configuration_attempts += 1
                continue
            else:
                configuration_attempts = 0

            with get_session() as session:
                # Remove completed tasks from the list of running tasks
                self.running_task_ids = TaskRepository(session).filter_incomplete_tasks(
                    self.running_task_ids
                )

                # Check for expired tasks
                expired_tasks = TaskRepository(session).update_expired_tasks(
                    execution_timeout=aana_settings.task_queue.execution_timeout,
                    heartbeat_timeout=aana_settings.task_queue.heartbeat_timeout,
                    max_retries=aana_settings.task_queue.max_retries,
                )
                for task in expired_tasks:
                    deployment_response = self.deployment_responses.get(task.id)
                    if deployment_response:
                        deployment_response.cancel()

            # If the queue is full, wait and retry
            if len(self.running_task_ids) >= aana_settings.task_queue.num_workers:
                await sleep_exponential_backoff(0.1, 5.0, full_queue_attempts)
                full_queue_attempts += 1
                continue
            else:
                full_queue_attempts = 0

            if not handle:
                # Try to get the app handle, if it fails, wait and retry later
                try:
                    handle = serve.get_app_handle(self.app_name)
                except ray.serve.exceptions.RayServeException as e:
                    print(f"App {self.app_name} not available yet: {e}, retrying...")
                    await sleep_exponential_backoff(1.0, 5.0, no_app_attempts)
                    no_app_attempts += 1
                    continue

            # Get new tasks from the database
            with get_session() as session:
                num_tasks_to_assign = aana_settings.task_queue.num_workers - len(
                    self.running_task_ids
                )
                tasks = TaskRepository(session).fetch_unprocessed_tasks(
                    limit=num_tasks_to_assign,
                    max_retries=aana_settings.task_queue.max_retries,
                    retryable_exceptions=self.retryable_exceptions,
                )

                # If there are no tasks, wait and retry
                if not tasks:
                    await sleep_exponential_backoff(0.1, 5.0, no_tasks_attempts)
                    no_tasks_attempts += 1
                    continue
                else:
                    no_tasks_attempts = 0

                # Start processing the tasks
                for task in tasks:
                    deployment_response = handle.execute_task.remote(task_id=task.id)
                    self.deployment_responses[task.id] = deployment_response
                    self.running_task_ids.append(str(task.id))
