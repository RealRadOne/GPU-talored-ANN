import random
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from taskScheduler.broker import ShardTask


class OrderingPolicy(Protocol):
    def order_tasks(self, tasks: list["ShardTask"]) -> list["ShardTask"]:
        ...

class DefaultOrderingPolicy:
    def order_tasks(self, tasks: list["ShardTask"]) -> list["ShardTask"]:
        return sorted(tasks, key=lambda task: (task.shard_id, task.task_id))


class RetryPrioritizedPolicy:
    def order_tasks(self, tasks: list["ShardTask"]) -> list["ShardTask"]:
        return sorted(tasks, key=lambda task: (-task.retry_count, task.shard_id, task.task_id))

class RandomOrderingPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.shuffler = random.Random(seed)  # noqa: S311  # nosec B311

    def order_tasks(self, tasks: list["ShardTask"]) -> list["ShardTask"]:
        shuffled_tasks = sorted(tasks, key=lambda task: (task.shard_id, task.task_id))
        self.shuffler.shuffle(shuffled_tasks)
        return shuffled_tasks

