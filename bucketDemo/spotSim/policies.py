import random
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from spotSim.broker import ShardTask


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



class LanAssignmentPolicy:

    def __init__(self, ondemand_share: float = 0.25) -> None:
        self.ondemand_share = ondemand_share

    def assign(self, broker, worker: int, ondemand: bool, worker_available, work_s: float, n_spot: int,
               min_steal_s: float):
        """Returns (segment index, stolen) or None. worker_available(w) -> bool."""
        eps = broker.EPS
        od_cap = self.ondemand_share * work_s / max(1, n_spot)
        for i, s in enumerate(broker.segments):
            if s.reserved == worker and s.owner < 0 and s.cursor < s.end - eps:
                return i, False
        best = None
        for i, s in enumerate(broker.segments):
            if s.reserved >= 0 and s.reserved != worker:
                if worker_available(s.reserved):
                    continue
                s.reserved = -1   # its worker is not available: free for anyone
            if s.owner < 0 and s.cursor < s.end - eps and (best is None or s.cursor < broker.segments[best].cursor):
                best = i
        if best is not None:
            s = broker.segments[best]
            if ondemand and s.end - s.cursor > od_cap * 1.0001:
                broker.split(best, s.cursor + od_cap)
            return best, False
        victim, victim_left = None, 0.0
        for i, s in enumerate(broker.segments):
            if s.owner < 0 or s.owner == worker:
                continue
            left = s.left_after_issued()
            if left >= min_steal_s and left > victim_left:
                victim, victim_left = i, left
        if victim is None:
            return None
        take = min(od_cap, victim_left / 2) if ondemand else victim_left / 2
        return broker.split(victim, broker.segments[victim].end - take), True
