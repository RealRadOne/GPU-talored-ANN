from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scheduler.policies import OrderingPolicy

class PipelineStep(Enum):
    UNINITIALIZED = 0
    CLUSTERING   = 1
    ASSIGNMENT   = 2
    INDEXING     = 3
    SEARCH       = 4
    MERGE        = 5

class TaskStatus(Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ShardTask:
    def __init__(self,task_id: str, 
                 shard_id: int,
                 rows: int,
                 dim: int,
                 deg: int)->None:
        self.task_id = task_id
        self.shard_id = shard_id
        self.rows = rows
        self.dim = dim
        self.deg = deg

        self.status = TaskStatus.READY
        self.current_step = PipelineStep.UNINITIALIZED
        self.retry_count = 0
        self.gpu_id = -1
        
    def update_stage(self, step: PipelineStep):
        self.current_step = step

    def __repr__(self)->str:
        return (
            f"ShardTask(id='{self.task_id}', shard={self.shard_id}, rows={self.rows}, "
            f"step={self.current_step.name}, status={self.status.value}, retries={self.retry_count})"
        )

class TaskBroker:
    def __init__(self,ordering_policy: Optional["OrderingPolicy"]=None)->None:
        self.tasks: dict[str, ShardTask] = {}

        if ordering_policy is None:
            from taskScheduler.policies import DefaultOrderingPolicy
            self.ordering_policy = DefaultOrderingPolicy()
        else:
            self.ordering_policy = ordering_policy

    def add_task(
        self,
        task_id: str,
        shard_id: int,
        rows: int,
        dim: int,
        deg: int
        )-> None:
        tid = str(task_id)
        self.tasks[tid] = ShardTask(
            task_id=tid,
            shard_id=shard_id,
            rows=rows,
            dim=dim,
            deg=deg
        )

    def next_task(self)->str | None:
        ready_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.READY]
        if not ready_tasks:
            return None
        ordered_tasks = self.ordering_policy.order_tasks(ready_tasks)
        selected_task = ordered_tasks[0]
        selected_task.status = TaskStatus.RUNNING
        return selected_task.task_id

    def update_task_stage(self,task_id: str, step:PipelineStep)->None:
        tid = str(task_id)
        if tid in self.tasks:
            self.tasks[tid].update_stage(step)

    def complete(self,task_id:str)->None:
        tid = str(task_id)
        if tid in self.tasks:
            self.tasks[tid].status = TaskStatus.COMPLETED

    def fail(self,task_id:str)->None:
        tid = str(task_id)
        if tid in self.tasks:
            self.tasks[tid].status = TaskStatus.FAILED

    def interrupt(self,task_id:str,max_retries:Optional[int]=None)->None:
        task = self.tasks.get(str(task_id))
        if task is None or task.status != TaskStatus.RUNNING:
            return
        task.retry_count += 1
        has_retries_left = max_retries is None or task.retry_count < max_retries
        task.status = TaskStatus.READY if has_retries_left else TaskStatus.FAILED

    def get_task(self,task_id:str)->ShardTask | None:
        return self.tasks.get(str(task_id))

    def all_done(self)->bool:
        if not self.tasks:
            return False
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self.tasks.values()
        )

    