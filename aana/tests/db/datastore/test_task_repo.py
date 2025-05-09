# ruff: noqa: S101

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete

from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity
from aana.storage.repository.task import TaskRepository


@pytest.mark.asyncio
async def test_save_task_repo(db_session_manager):
    """Test saving a task."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)
        task = await task_repo.save(endpoint="/test", data={"test": "test"})

        task_entity = await task_repo.read(task.id)
        assert task_entity
        assert task_entity.id == task.id
        assert task_entity.endpoint == "/test"
        assert task_entity.data == {"test": "test"}


@pytest.mark.asyncio
async def test_multiple_simultaneous_tasks(db_session_manager):
    """Test creating multiple tasks in parallel, each with its own session."""

    async def add_task(i):
        async with db_session_manager.session() as session:
            task_repo = TaskRepository(session)
            return await task_repo.save(endpoint="/test", data={"test": i})

    tasks = [asyncio.create_task(add_task(i)) for i in range(30)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 30


@pytest.mark.asyncio
async def test_get_unprocessed_tasks(db_session_manager):
    """Test fetching unprocessed tasks."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        async def _create_sample_tasks():
            # Remove all existing tasks
            await session.execute(delete(TaskEntity))
            await session.commit()

            # Create sample tasks with different statuses
            now = datetime.now(timezone.utc)

            # fmt: off
            task1 = TaskEntity(endpoint="/test1", data={"test": "data1"}, status=TaskStatus.CREATED, priority=1, created_at=now - timedelta(hours=10))
            task2 = TaskEntity(endpoint="/test2", data={"test": "data2"}, status=TaskStatus.NOT_FINISHED, priority=2, created_at=now - timedelta(hours=1))
            task3 = TaskEntity(endpoint="/test3", data={"test": "data3"}, status=TaskStatus.COMPLETED, priority=3, created_at=now - timedelta(hours=2))
            task4 = TaskEntity(endpoint="/test4", data={"test": "data4"}, status=TaskStatus.CREATED, priority=2, created_at=now - timedelta(hours=3))
            task5 = TaskEntity(endpoint="/test5", data={"test": "data5"}, status=TaskStatus.FAILED, priority=1, created_at=now - timedelta(minutes=1), result={"error": "InferenceException"})
            task6 = TaskEntity(endpoint="/test6", data={"test": "data6"}, status=TaskStatus.RUNNING, priority=3, created_at=now - timedelta(minutes=2))
            task7 = TaskEntity(endpoint="/test7", data={"test": "data7"}, status=TaskStatus.FAILED, priority=1, created_at=now - timedelta(minutes=3), result={"error": "NonRecoverableError"})
            # fmt: on

            session.add_all([task1, task2, task3, task4, task5, task6, task7])
            await session.commit()
            return task1, task2, task3, task4, task5, task6, task7

        # Create sample tasks
        task1, task2, task3, task4, task5, task6, task7 = await _create_sample_tasks()

        # Fetch unprocessed tasks without any limit
        unprocessed_tasks = await task_repo.fetch_unprocessed_tasks(
            max_retries=3, retryable_exceptions=["InferenceException"]
        )

        # Assert that only tasks with CREATED and NOT_FINISHED status are returned
        assert len(unprocessed_tasks) == 4
        assert task1 in unprocessed_tasks
        assert task2 in unprocessed_tasks
        assert task4 in unprocessed_tasks
        assert task5 in unprocessed_tasks

        # Ensure tasks are ordered by priority and then by created_at
        assert unprocessed_tasks[0].id == task4.id  # Highest priority
        assert unprocessed_tasks[1].id == task2.id  # Same priority, but a newer task
        assert unprocessed_tasks[2].id == task1.id  # Lowest priority
        assert unprocessed_tasks[3].id == task5.id  # Highest priority, but older

        # Create sample tasks
        task1, task2, task3, task4, task5, task6, task7 = await _create_sample_tasks()

        # Fetch unprocessed tasks with a limit
        limited_tasks = await task_repo.fetch_unprocessed_tasks(
            limit=2, max_retries=3, retryable_exceptions=["InferenceException"]
        )

        # Assert that only the specified number of tasks is returned
        assert len(limited_tasks) == 2
        assert limited_tasks[0].id == task4.id  # Highest priority
        assert limited_tasks[1].id == task2.id  # Same priority, but older


@pytest.mark.asyncio
async def test_get_unprocessed_tasks_with_api_key(db_session_manager_with_api_service):
    """Test fetching unprocessed tasks with an API key."""
    async with db_session_manager_with_api_service.session() as session:
        task_repo = TaskRepository(session)

        # Remove all existing tasks and API keys
        await session.execute(delete(TaskEntity))
        await session.execute(delete(ApiKeyEntity))
        await session.commit()

        # fmt: off
        session.add_all(
            [
                # Active users
                ApiKeyEntity(user_id="1", is_subscription_active=True, api_key="key1", key_id="key1", subscription_id="sub1", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
                ApiKeyEntity(user_id="2", is_subscription_active=True, api_key="key2", key_id="key2", subscription_id="sub2", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
                ApiKeyEntity(user_id="3", is_subscription_active=True, api_key="key3", key_id="key3", subscription_id="sub3", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
                # Inactive user (should be excluded)
                ApiKeyEntity(user_id="4", is_subscription_active=False, api_key="key4", key_id="key4", subscription_id="sub4", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
            ]
        )
        await session.commit()
        # fmt: on

        # fmt: off
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)

        session.add_all([
            # User 1 (3 RUNNING tasks - lower priority)
            TaskEntity(user_id="1", status=TaskStatus.RUNNING, priority=1, created_at=base_time, endpoint="/test1", data={"test": "data1"}),
            TaskEntity(user_id="1", status=TaskStatus.RUNNING, priority=2, created_at=base_time + timedelta(minutes=5), endpoint="/test2", data={"test": "data2"}),
            TaskEntity(user_id="1", status=TaskStatus.CREATED, priority=3, created_at=base_time + timedelta(minutes=10), endpoint="/test3", data={"test": "data3"}),
            TaskEntity(user_id="1", status=TaskStatus.RUNNING, priority=3, created_at=base_time + timedelta(minutes=15), endpoint="/test4", data={"test": "data4"}),

            # User 2 (1 RUNNING task - medium priority)
            TaskEntity(user_id="2", status=TaskStatus.RUNNING, priority=2, created_at=base_time + timedelta(minutes=20), endpoint="/test5", data={"test": "data5"}),
            TaskEntity(user_id="2", status=TaskStatus.NOT_FINISHED, priority=3, created_at=base_time + timedelta(minutes=25), endpoint="/test6", data={"test": "data6"}),
            TaskEntity(user_id="2", status=TaskStatus.CREATED, priority=1, created_at=base_time + timedelta(minutes=30), endpoint="/test7", data={"test": "data7"}),

            # User 3 (0 RUNNING tasks - should be prioritized first)
            TaskEntity(user_id="3", status=TaskStatus.CREATED, priority=1, created_at=base_time + timedelta(minutes=35), endpoint="/test8", data={"test": "data8"}),
            TaskEntity(user_id="3", status=TaskStatus.FAILED, priority=3, created_at=base_time + timedelta(minutes=40), endpoint="/test9", data={"test": "data9"}, result={"error": "InferenceException"},),

            # User 4 (Inactive subscription - should be excluded)
            TaskEntity(user_id="4", status=TaskStatus.CREATED, priority=3, created_at=base_time, endpoint="/test10", data={"test": "data10"}),

            # No user specified
            TaskEntity(user_id=None, status=TaskStatus.CREATED, priority=4, created_at=base_time + timedelta(minutes=5), endpoint="/test11", data={"test": "data11"}),
            TaskEntity(user_id=None, status=TaskStatus.RUNNING, priority=5, created_at=base_time + timedelta(minutes=10), endpoint="/test12", data={"test": "data12"}),
            TaskEntity(user_id=None, status=TaskStatus.NOT_FINISHED, priority=3, created_at=base_time + timedelta(minutes=15), endpoint="/test13", data={"test": "data13"}),
            TaskEntity(user_id=None, status=TaskStatus.CREATED, priority=2, created_at=base_time + timedelta(minutes=20), endpoint="/test14", data={"test": "data14"}),
        ])
        await session.commit()
        # fmt: on

        # Fetch unprocessed tasks without any limit
        unprocessed_tasks = await task_repo.fetch_unprocessed_tasks(
            max_retries=3,
            retryable_exceptions=["InferenceException"],
            api_service_enabled=True,
            maximum_active_tasks_per_user=2,
        )

        # Assert that the right tasks are returned in the right order
        assert len(unprocessed_tasks) == 6
        expected_endpoints = set(
            ["/test11", "/test13", "/test6", "/test9", "/test14", "/test8"]
        )
        assert expected_endpoints == {task.endpoint for task in unprocessed_tasks}


@pytest.mark.asyncio
async def test_update_status(db_session_manager):
    """Test updating the status of a task."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Create a task with an initial status
        task = TaskEntity(
            endpoint="/test", data={"key": "value"}, status=TaskStatus.CREATED
        )
        session.add(task)
        await session.commit()

        # Update the status to ASSIGNED and check fields
        await task_repo.update_status(task.id, TaskStatus.ASSIGNED, progress=50)

        updated_task = await task_repo.read(task.id)
        assert updated_task.status == TaskStatus.ASSIGNED
        assert updated_task.assigned_at is not None
        assert updated_task.num_retries == 1
        assert updated_task.progress == 50
        assert updated_task.result is None

        # Update the status to COMPLETED and check fields
        await task_repo.update_status(
            task.id,
            TaskStatus.COMPLETED,
            progress=100,
            result={"result": "final_result"},
        )

        updated_task = await task_repo.read(task.id)
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.completed_at is not None
        assert updated_task.progress == 100
        assert updated_task.result == {"result": "final_result"}

        # Ensure timestamps are reasonable
        assert updated_task.assigned_at < updated_task.completed_at
        assert updated_task.created_at < updated_task.assigned_at

        # Update the status to FAILED and check fields
        await task_repo.update_status(
            task.id, TaskStatus.FAILED, progress=0, result={"error": "error_message"}
        )
        updated_task = await task_repo.read(task.id)

        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.completed_at is not None
        assert updated_task.progress == 0
        assert updated_task.result == {"error": "error_message"}

        # Ensure timestamps are reasonable
        assert updated_task.assigned_at < updated_task.completed_at
        assert updated_task.created_at < updated_task.assigned_at


@pytest.mark.asyncio
async def test_get_active_tasks(db_session_manager):
    """Test fetching active tasks."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Remove all existing tasks
        await session.execute(delete(TaskEntity))
        await session.commit()

        # Create sample tasks with different statuses
        task1 = TaskEntity(
            endpoint="/task1", data={"test": "data1"}, status=TaskStatus.CREATED
        )
        task2 = TaskEntity(
            endpoint="/task2", data={"test": "data2"}, status=TaskStatus.RUNNING
        )
        task3 = TaskEntity(
            endpoint="/task3", data={"test": "data3"}, status=TaskStatus.ASSIGNED
        )
        task4 = TaskEntity(
            endpoint="/task4", data={"test": "data4"}, status=TaskStatus.COMPLETED
        )
        task5 = TaskEntity(
            endpoint="/task5", data={"test": "data5"}, status=TaskStatus.FAILED
        )
        task6 = TaskEntity(
            endpoint="/task6", data={"test": "data6"}, status=TaskStatus.NOT_FINISHED
        )

        session.add_all([task1, task2, task3, task4, task5, task6])
        await session.commit()

        # Fetch active tasks
        active_tasks = await task_repo.get_active_tasks()

        # Assert that only tasks with RUNNING and ASSIGNED status are returned
        assert len(active_tasks) == 2
        assert task2 in active_tasks
        assert task3 in active_tasks
        assert all(
            task.status in [TaskStatus.RUNNING, TaskStatus.ASSIGNED]
            for task in active_tasks
        )


@pytest.mark.asyncio
async def test_remove_completed_tasks(db_session_manager):
    """Test removing completed tasks."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Remove all existing tasks
        await session.execute(delete(TaskEntity))
        await session.commit()

        # Create sample tasks with different statuses
        task1 = TaskEntity(
            endpoint="/task1", data={"test": "data1"}, status=TaskStatus.COMPLETED
        )
        task2 = TaskEntity(
            endpoint="/task2", data={"test": "data2"}, status=TaskStatus.RUNNING
        )
        task3 = TaskEntity(
            endpoint="/task3", data={"test": "data3"}, status=TaskStatus.ASSIGNED
        )
        task4 = TaskEntity(
            endpoint="/task4", data={"test": "data4"}, status=TaskStatus.COMPLETED
        )
        task5 = TaskEntity(
            endpoint="/task5", data={"test": "data5"}, status=TaskStatus.FAILED
        )
        task6 = TaskEntity(
            endpoint="/task6", data={"test": "data6"}, status=TaskStatus.NOT_FINISHED
        )

        all_tasks = [task1, task2, task3, task4, task5, task6]
        unfinished_tasks = [task2, task3]

        session.add_all(all_tasks)
        await session.commit()

        # Remove completed tasks
        task_ids = [str(task.id) for task in all_tasks]
        non_completed_task_ids = await task_repo.filter_incomplete_tasks(task_ids)

        # Assert that only the task IDs that are not completed are returned
        assert set(non_completed_task_ids) == {
            str(task.id) for task in unfinished_tasks
        }


@pytest.mark.asyncio
async def test_update_expired_tasks(db_session_manager):
    """Test updating expired tasks."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Remove all existing tasks
        await session.execute(delete(TaskEntity))
        await session.commit()

        # Set up current time and a cutoff time
        current_time = datetime.now(timezone.utc)
        execution_timeout = 3600  # 1 hour in seconds
        heartbeat_timeout = 60  # 1 minute in seconds

        # Create tasks with different updated_at times and statuses
        task1 = TaskEntity(
            endpoint="/task1",
            data={"test": "data1"},
            status=TaskStatus.RUNNING,
            assigned_at=current_time - timedelta(hours=2),
            updated_at=current_time - timedelta(seconds=10),
        )
        task2 = TaskEntity(
            endpoint="/task2",
            data={"test": "data2"},
            status=TaskStatus.ASSIGNED,
            assigned_at=current_time - timedelta(seconds=2),
            updated_at=current_time - timedelta(seconds=5),
        )
        task3 = TaskEntity(
            endpoint="/task3",
            data={"test": "data3"},
            status=TaskStatus.RUNNING,
            assigned_at=current_time - timedelta(seconds=2),
            updated_at=current_time,
        )
        task4 = TaskEntity(
            endpoint="/task4",
            data={"test": "data4"},
            status=TaskStatus.COMPLETED,
            assigned_at=current_time - timedelta(hours=1),
            updated_at=current_time - timedelta(hours=2),
        )
        task5 = TaskEntity(
            endpoint="/task5",
            data={"test": "data5"},
            status=TaskStatus.FAILED,
            assigned_at=current_time - timedelta(minutes=1),
            updated_at=current_time - timedelta(seconds=4),
        )
        task6 = TaskEntity(
            endpoint="/task6",
            data={"test": "data6"},
            status=TaskStatus.RUNNING,
            assigned_at=current_time - timedelta(minutes=3),
            updated_at=current_time - timedelta(minutes=2),
        )

        session.add_all([task1, task2, task3, task4, task5, task6])
        await session.commit()

        # Fetch expired tasks
        expired_tasks = await task_repo.update_expired_tasks(
            execution_timeout=execution_timeout,
            heartbeat_timeout=heartbeat_timeout,
            max_retries=3,
        )

        # Assert that only tasks with RUNNING or ASSIGNED status and an assigned_at time older than the execution_timeout or
        # heartbeat_timeout are returned
        expected_task_ids = {str(task1.id), str(task6.id)}
        returned_task_ids = {str(task.id) for task in expired_tasks}

        assert returned_task_ids == expected_task_ids


@pytest.mark.asyncio
async def test_retry_task(db_session_manager):
    """Test retrying a failed task."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Create a failed task with some progress and results
        task = TaskEntity(
            endpoint="/test",
            data={"test": "data"},
            status=TaskStatus.FAILED,
            progress=50,
            result={"error": "some error"},
            num_retries=2,
            assigned_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        session.add(task)
        await session.commit()

        # Retry the task
        retried_task = await task_repo.retry_task(str(task.id))

        # Check that task was reset properly
        assert retried_task.status == TaskStatus.CREATED
        assert retried_task.progress == 0
        assert retried_task.result is None
        assert retried_task.num_retries == 0
        assert retried_task.assigned_at is None
        assert retried_task.completed_at is None

        # Original data should be preserved
        assert retried_task.endpoint == "/test"
        assert retried_task.data == {"test": "data"}


@pytest.mark.asyncio
async def test_get_tasks(db_session_manager):
    """Test fetching tasks with various filters."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Remove all existing tasks
        await session.execute(delete(TaskEntity))
        await session.commit()

        # Create sample tasks with different users and statuses
        base_time = datetime.now(timezone.utc)
        tasks = [
            TaskEntity(
                endpoint="/test1",
                data={"test": "data1"},
                status=TaskStatus.COMPLETED,
                user_id="user1",
                created_at=base_time - timedelta(hours=1),
            ),
            TaskEntity(
                endpoint="/test2",
                data={"test": "data2"},
                status=TaskStatus.RUNNING,
                user_id="user1",
                created_at=base_time - timedelta(minutes=30),
            ),
            TaskEntity(
                endpoint="/test3",
                data={"test": "data3"},
                status=TaskStatus.FAILED,
                user_id="user2",
                created_at=base_time - timedelta(minutes=20),
            ),
            TaskEntity(
                endpoint="/test4",
                data={"test": "data4"},
                status=TaskStatus.COMPLETED,
                user_id="user2",
                created_at=base_time - timedelta(minutes=10),
            ),
            TaskEntity(
                endpoint="/test5",
                data={"test": "data5"},
                status=TaskStatus.COMPLETED,
                user_id=None,
                created_at=base_time,
            ),
        ]
        session.add_all(tasks)
        await session.commit()

        # Test getting all tasks without filters
        all_tasks = await task_repo.get_tasks()
        assert len(all_tasks) == 5
        assert all_tasks[0].endpoint == "/test5"  # Most recent first

        # Test filtering by user_id
        user1_tasks = await task_repo.get_tasks(user_id="user1")
        assert len(user1_tasks) == 2
        assert all(task.user_id == "user1" for task in user1_tasks)

        # Test filtering by status
        completed_tasks = await task_repo.get_tasks(status=TaskStatus.COMPLETED)
        assert len(completed_tasks) == 3
        assert all(task.status == TaskStatus.COMPLETED for task in completed_tasks)

        # Test combined filters
        user2_completed = await task_repo.get_tasks(
            user_id="user2", status=TaskStatus.COMPLETED
        )
        assert len(user2_completed) == 1
        assert user2_completed[0].endpoint == "/test4"

        # Test limit and offset
        limited_tasks = await task_repo.get_tasks(limit=2)
        assert len(limited_tasks) == 2
        assert limited_tasks[0].endpoint == "/test5"
        assert limited_tasks[1].endpoint == "/test4"

        offset_tasks = await task_repo.get_tasks(limit=2, offset=2)
        assert len(offset_tasks) == 2
        assert offset_tasks[0].endpoint == "/test3"
        assert offset_tasks[1].endpoint == "/test2"


@pytest.mark.asyncio
async def test_count_tasks(db_session_manager):
    """Test counting tasks by status."""
    async with db_session_manager.session() as session:
        task_repo = TaskRepository(session)

        # Remove all existing tasks
        await session.execute(delete(TaskEntity))
        await session.commit()

        # Create sample tasks with different users and statuses
        tasks = [
            # User 1 tasks
            TaskEntity(
                endpoint="/test1",
                data={"test": "data1"},
                status=TaskStatus.COMPLETED,
                user_id="user1",
            ),
            TaskEntity(
                endpoint="/test2",
                data={"test": "data2"},
                status=TaskStatus.RUNNING,
                user_id="user1",
            ),
            TaskEntity(
                endpoint="/test3",
                data={"test": "data3"},
                status=TaskStatus.FAILED,
                user_id="user1",
            ),
            TaskEntity(
                endpoint="/test4",
                data={"test": "data4"},
                status=TaskStatus.COMPLETED,
                user_id="user1",
            ),
            # User 2 tasks
            TaskEntity(
                endpoint="/test5",
                data={"test": "data5"},
                status=TaskStatus.COMPLETED,
                user_id="user2",
            ),
            TaskEntity(
                endpoint="/test6",
                data={"test": "data6"},
                status=TaskStatus.RUNNING,
                user_id="user2",
            ),
            # Tasks with no user
            TaskEntity(
                endpoint="/test7",
                data={"test": "data7"},
                status=TaskStatus.CREATED,
                user_id=None,
            ),
            TaskEntity(
                endpoint="/test8",
                data={"test": "data8"},
                status=TaskStatus.NOT_FINISHED,
                user_id=None,
            ),
        ]
        session.add_all(tasks)
        await session.commit()

        # Test counting all tasks
        all_counts = await task_repo.count()
        assert all_counts["total"] == 8
        assert all_counts["completed"] == 3
        assert all_counts["running"] == 2
        assert all_counts["failed"] == 1
        assert all_counts["created"] == 1
        assert all_counts["not_finished"] == 1

        # Test counting user1's tasks
        user1_counts = await task_repo.count(user_id="user1")
        assert user1_counts["total"] == 4
        assert user1_counts["completed"] == 2
        assert user1_counts["running"] == 1
        assert user1_counts["failed"] == 1

        # Test counting user2's tasks
        user2_counts = await task_repo.count(user_id="user2")
        assert user2_counts["total"] == 2
        assert user2_counts["completed"] == 1
        assert user2_counts["running"] == 1
