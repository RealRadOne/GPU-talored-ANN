
from typing import List, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from scheduler.broker import ShardTask


class OrderingPolicy(Protocol):
    def order_tasks(self, tasks: List["ShardTask"]) -> List["ShardTask"]:
        """Given a list of READY ShardTasks, returns them sorted by dispatch priority."""
        ...


class DefaultOrderingPolicy:
    def order_tasks(self, tasks: List["ShardTask"]) -> List["ShardTask"]:
        return sorted(tasks, key=lambda t: t.task_id)