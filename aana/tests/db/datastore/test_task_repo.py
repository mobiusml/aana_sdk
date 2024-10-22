# ruff: noqa: S101

import asyncio
from datetime import datetime, timedelta

import pytest

from aana.storage.models.task import Status as TaskStatus
from aana.storage.models.task import TaskEntity
from aana.storage.repository.task import TaskRepository


def test_save_task_repo(db_session):
    """Test saving a task."""
    task_repo = TaskRepository(db_session)
    task = task_repo.save(endpoint="/test", data={"test": "test"})

    task_entity = task_repo.read(task.id)
    assert task_entity
    assert task_entity.id == task.id
    assert task_entity.endpoint == "/test"
    assert task_entity.data == {"test": "test"}


@pytest.mark.asyncio
async def test_multiple_simultaneous_tasks(db_session):
    """Test creating multiple tasks in parallel."""
    task_repo = TaskRepository(db_session)

    # Create multiple tasks in parallel with asyncio
    async def add_task(i):
        task = task_repo.save(endpoint="/test", data={"test": i})
        return task

    async_tasks = []
    for i in range(30):
        async_task = asyncio.create_task(add_task(i))
        async_tasks.append(async_task)

    # Wait for all tasks to complete
    await asyncio.gather(*async_tasks)


def test_get_unprocessed_tasks(db_session):
    """Test fetching unprocessed tasks."""
    task_repo = TaskRepository(db_session)

    def _create_sample_tasks():
        # Remove all existing tasks
        db_session.query(TaskEntity).delete()
        db_session.commit()

        # Create sample tasks with different statuses
        now = datetime.now()  # noqa: DTZ005

        task1 = TaskEntity(
            endpoint="/test1",
            data={"test": "data1"},
            status=TaskStatus.CREATED,
            priority=1,
            created_at=now - timedelta(hours=10),
        )
        task2 = TaskEntity(
            endpoint="/test2",
            data={"test": "data2"},
            status=TaskStatus.NOT_FINISHED,
            priority=2,
            created_at=now - timedelta(hours=1),
        )
        task3 = TaskEntity(
            endpoint="/test3",
            data={"test": "data3"},
            status=TaskStatus.COMPLETED,
            priority=3,
            created_at=now - timedelta(hours=2),
        )
        task4 = TaskEntity(
            endpoint="/test4",
            data={"test": "data4"},
            status=TaskStatus.CREATED,
            priority=2,
            created_at=now - timedelta(hours=3),
        )

        db_session.add_all([task1, task2, task3, task4])
        db_session.commit()
        return task1, task2, task3, task4

    # Create sample tasks
    task1, task2, task3, task4 = _create_sample_tasks()

    # Fetch unprocessed tasks without any limit
    unprocessed_tasks = task_repo.fetch_unprocessed_tasks()

    # Assert that only tasks with CREATED and NOT_FINISHED status are returned
    assert len(unprocessed_tasks) == 3
    assert task1 in unprocessed_tasks
    assert task2 in unprocessed_tasks
    assert task4 in unprocessed_tasks

    # Ensure tasks are ordered by priority and then by created_at
    assert unprocessed_tasks[0].id == task4.id  # Highest priority
    assert unprocessed_tasks[1].id == task2.id  # Same priority, but a newer task
    assert unprocessed_tasks[2].id == task1.id  # Lowest priority

    # Create sample tasks
    task1, task2, task3, task4 = _create_sample_tasks()

    # Fetch unprocessed tasks with a limit
    limited_tasks = task_repo.fetch_unprocessed_tasks(limit=2)

    # Assert that only the specified number of tasks is returned
    assert len(limited_tasks) == 2
    assert limited_tasks[0].id == task4.id  # Highest priority
    assert limited_tasks[1].id == task2.id  # Same priority, but older


def test_update_status(db_session):
    """Test updating the status of a task."""
    task_repo = TaskRepository(db_session)

    # Create a task with an initial status
    task = TaskEntity(
        endpoint="/test", data={"key": "value"}, status=TaskStatus.CREATED
    )
    db_session.add(task)
    db_session.commit()

    # Update the status to ASSIGNED and check fields
    task_repo.update_status(task.id, TaskStatus.ASSIGNED, progress=50)

    updated_task = task_repo.read(task.id)
    assert updated_task.status == TaskStatus.ASSIGNED
    assert updated_task.assigned_at is not None
    assert updated_task.num_retries == 1
    assert updated_task.progress == 50
    assert updated_task.result is None

    # Update the status to COMPLETED and check fields
    task_repo.update_status(
        task.id, TaskStatus.COMPLETED, progress=100, result={"result": "final_result"}
    )

    updated_task = task_repo.read(task.id)
    assert updated_task.status == TaskStatus.COMPLETED
    assert updated_task.completed_at is not None
    assert updated_task.progress == 100
    assert updated_task.result == {"result": "final_result"}

    # Ensure timestamps are reasonable
    assert updated_task.assigned_at < updated_task.completed_at
    assert updated_task.created_at < updated_task.assigned_at

    # Update the status to FAILED and check fields
    task_repo.update_status(
        task.id, TaskStatus.FAILED, progress=0, result={"error": "error_message"}
    )
    updated_task = task_repo.read(task.id)

    assert updated_task.status == TaskStatus.FAILED
    assert updated_task.completed_at is not None
    assert updated_task.progress == 0
    assert updated_task.result == {"error": "error_message"}

    # Ensure timestamps are reasonable
    assert updated_task.assigned_at < updated_task.completed_at
    assert updated_task.created_at < updated_task.assigned_at


def test_get_active_tasks(db_session):
    """Test fetching active tasks."""
    task_repo = TaskRepository(db_session)

    # Remove all existing tasks
    db_session.query(TaskEntity).delete()
    db_session.commit()

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

    db_session.add_all([task1, task2, task3, task4, task5, task6])
    db_session.commit()

    # Fetch active tasks
    active_tasks = task_repo.get_active_tasks()

    # Assert that only tasks with RUNNING and ASSIGNED status are returned
    assert len(active_tasks) == 2
    assert task2 in active_tasks
    assert task3 in active_tasks
    assert all(
        task.status in [TaskStatus.RUNNING, TaskStatus.ASSIGNED]
        for task in active_tasks
    )


def test_remove_completed_tasks(db_session):
    """Test removing completed tasks."""
    task_repo = TaskRepository(db_session)

    # Remove all existing tasks
    db_session.query(TaskEntity).delete()
    db_session.commit()

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

    db_session.add_all(all_tasks)
    db_session.commit()

    # Remove completed tasks
    task_ids = [str(task.id) for task in all_tasks]
    non_completed_task_ids = task_repo.filter_incomplete_tasks(task_ids)

    # Assert that only the task IDs that are not completed are returned
    assert set(non_completed_task_ids) == {str(task.id) for task in unfinished_tasks}


def test_update_expired_tasks(db_session):
    """Test updating expired tasks."""
    task_repo = TaskRepository(db_session)

    # Remove all existing tasks
    db_session.query(TaskEntity).delete()
    db_session.commit()

    # Set up current time and a cutoff time
    current_time = datetime.now()  # noqa: DTZ005
    execution_timeout = 3600  # 1 hour in seconds

    # Create tasks with different updated_at times and statuses
    task1 = TaskEntity(
        endpoint="/task1",
        data={"test": "data1"},
        status=TaskStatus.RUNNING,
        updated_at=current_time - timedelta(hours=2),
    )
    task2 = TaskEntity(
        endpoint="/task2",
        data={"test": "data2"},
        status=TaskStatus.ASSIGNED,
        updated_at=current_time - timedelta(seconds=2),
    )
    task3 = TaskEntity(
        endpoint="/task3",
        data={"test": "data3"},
        status=TaskStatus.RUNNING,
        updated_at=current_time,
    )
    task4 = TaskEntity(
        endpoint="/task4",
        data={"test": "data4"},
        status=TaskStatus.COMPLETED,
        updated_at=current_time - timedelta(hours=2),
    )
    task5 = TaskEntity(
        endpoint="/task5",
        data={"test": "data5"},
        status=TaskStatus.FAILED,
        updated_at=current_time - timedelta(seconds=4),
    )

    db_session.add_all([task1, task2, task3, task4, task5])
    db_session.commit()

    # Fetch expired tasks
    expired_tasks = task_repo.update_expired_tasks(
        execution_timeout=execution_timeout, max_retries=3
    )

    # Assert that only tasks with RUNNING or ASSIGNED status and an updated_at older than the cutoff are returned
    expected_task_ids = {str(task1.id)}
    returned_task_ids = {str(task.id) for task in expired_tasks}

    assert returned_task_ids == expected_task_ids
