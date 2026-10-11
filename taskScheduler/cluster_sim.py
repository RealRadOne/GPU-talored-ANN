from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from scheduler.broker import TaskBroker, TaskStatus
from scheduler.policies import DefaultOrderingPolicy, OrderingPolicy


@dataclass
class AvailabilityEvent:
    """Represents a GPU availability state change in scenario time."""

    timestamp: float
    event_type: str  # "ADD" or "REMOVE"
    gpu_id: int


class TaskRunner(Protocol):
    """Protocol signature for task execution strategies (FakeRunner vs ShardRunner)."""

    def is_finished(self) -> bool:
        ...

    def terminate(self) -> None:
        ...


class FakeRunner:
    """Simulates task execution via scaled wall-clock duration without GPU dependencies."""

    def __init__(
        self, task_id: str, base_duration_sec: float, time_scale: float = 1.0
    ) -> None:
        self.task_id = task_id
        self.duration = base_duration_sec / time_scale
        self.start_time = time.monotonic()

    def is_finished(self) -> bool:
        return (time.monotonic() - self.start_time) >= self.duration

    def terminate(self) -> None:
        pass


class EventLogger:
    """Logs simulation timeline events (START, INTERRUPT, FINISH) to CSV."""

    def __init__(self, log_filepath: str) -> None:
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)
        self.file = open(log_filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "event_type", "task_id", "gpu_id", "npts"])

    def log_event(
        self,
        timestamp: float,
        event_type: str,
        task_id: str,
        gpu_id: int,
        npts: int,
    ) -> None:
        self.writer.writerow([round(timestamp, 4), event_type, task_id, gpu_id, npts])
        self.file.flush()

    def close(self) -> None:
        if self.file and not self.file.closed:
            self.file.close()


class ClusterSimulator:
    """Multi-worker controller driving TaskBroker against GPU availability timelines."""

    def __init__(
        self,
        broker: TaskBroker,
        events: List[AvailabilityEvent],
        log_filepath: str = "cluster_events.csv",
        time_scale: float = 1.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self.broker = broker
        # Sort events by ascending scenario timestamp
        self.events = sorted(events, key=lambda e: e.timestamp)
        self.time_scale = time_scale
        self.poll_seconds = poll_seconds
        self.logger = EventLogger(log_filepath)

        self.start_wall_time = time.monotonic()
        self.active_gpus: set[int] = set()
        self.worker_assignments: Dict[int, str] = {}  # gpu_id -> task_id
        self.active_runners: Dict[str, TaskRunner] = {}  # task_id -> TaskRunner

    def scenario_time(self) -> float:
        """Returns elapsed scenario time scaled by time_scale."""
        return (time.monotonic() - self.start_wall_time) * self.time_scale

    def process_due_events(self) -> None:
        """Applies GPU ADD/REMOVE availability events due at the current scenario timestamp."""
        now = self.scenario_time()
        while self.events and self.events[0].timestamp <= now:
            event = self.events.pop(0)
            if event.event_type == "ADD":
                self.active_gpus.add(event.gpu_id)
            elif event.event_type == "REMOVE":
                if event.gpu_id in self.active_gpus:
                    self.active_gpus.remove(event.gpu_id)

                # Preempt active task running on evicted GPU
                if event.gpu_id in self.worker_assignments:
                    task_id = self.worker_assignments.pop(event.gpu_id)
                    runner = self.active_runners.pop(task_id, None)
                    if runner:
                        runner.terminate()

                    task = self.broker.get_task(task_id)
                    npts = task.rows if task else 0
                    self.broker.interrupt(task_id)
                    self.logger.log_event(
                        self.scenario_time(), "INTERRUPT", task_id, event.gpu_id, npts
                    )

    def reap_completed_workers(self) -> None:
        """Checks for finished task runners and updates task completion state."""
        finished_tasks = []
        for task_id, runner in list(self.active_runners.items()):
            if runner.is_finished():
                finished_tasks.append(task_id)

        for task_id in finished_tasks:
            self.active_runners.pop(task_id)
            # Find associated GPU worker
            gpu_id = -1
            for g_id, t_id in list(self.worker_assignments.items()):
                if t_id == task_id:
                    gpu_id = g_id
                    del self.worker_assignments[g_id]
                    break

            task = self.broker.get_task(task_id)
            npts = task.rows if task else 0
            self.broker.complete(task_id)
            self.logger.log_event(
                self.scenario_time(), "FINISH", task_id, gpu_id, npts
            )

    def dispatch_idle_workers(self) -> None:
        """Assigns READY tasks from TaskBroker to idle active GPUs."""
        idle_gpus = [
            gpu_id
            for gpu_id in self.active_gpus
            if gpu_id not in self.worker_assignments
        ]
        for gpu_id in idle_gpus:
            task_id = self.broker.next_task()
            if task_id is None:
                break

            task = self.broker.get_task(task_id)
            npts = task.rows if task else 0

            # Base duration heuristic: 1 second per 100k vectors
            simulated_duration = max(0.5, (npts / 100_000.0))
            runner = FakeRunner(
                task_id,
                base_duration_sec=simulated_duration,
                time_scale=self.time_scale,
            )

            self.worker_assignments[gpu_id] = task_id
            self.active_runners[task_id] = runner
            self.logger.log_event(
                self.scenario_time(), "START", task_id, gpu_id, npts
            )

    def run(self) -> None:
        """Main simulation execution loop."""
        self.start_wall_time = time.monotonic()
        try:
            while not self.broker.all_done():
                self.process_due_events()
                self.reap_completed_workers()
                self.dispatch_idle_workers()
                time.sleep(self.poll_seconds)
        finally:
            self.logger.close()


if __name__ == "__main__":
    # Self-contained dry-run validation using FakeRunner
    broker = TaskBroker(ordering_policy=DefaultOrderingPolicy())

    # Register 5 test tasks
    for i in range(1, 6):
        broker.add_task(
            task_id=f"task_{i}", shard_id=i, rows=100_000 * i, dim=128, deg=32
        )

    # Simulated trace scenario: 2 GPUs joining, 1 getting evicted mid-run
    events = [
        AvailabilityEvent(timestamp=0.0, event_type="ADD", gpu_id=0),
        AvailabilityEvent(timestamp=0.0, event_type="ADD", gpu_id=1),
        AvailabilityEvent(timestamp=1.5, event_type="REMOVE", gpu_id=0),
        AvailabilityEvent(timestamp=3.0, event_type="ADD", gpu_id=0),
    ]

    sim = ClusterSimulator(
        broker=broker,
        events=events,
        log_filepath="/content/cluster_events.csv",
        time_scale=1.0,
    )
    print("=== Starting Cluster Simulator Dry Run ===")
    sim.run()
    print("=== Simulation Complete. Log written to /content/cluster_events.csv ===")