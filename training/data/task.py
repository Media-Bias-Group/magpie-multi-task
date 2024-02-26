"""This module contains the Task class."""


class Task:
    """Wrap subtasks."""

    def __init__(self, task_id, subtasks_list):
        """Initialize a Task."""
        self.task_id = task_id
        self.subtasks_list = subtasks_list

    def __repr__(self):
        """Represent a task."""
        return (
            f"Task {self.task_id} with {len(self.subtasks_list)} subtask{'s' if len(self.subtasks_list) > 1 else ''}"
        )

    def __str__(self) -> str:
        return str(self.task_id)
