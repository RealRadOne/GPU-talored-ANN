from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from spotSim.broker import TaskBroker
from spotSim.policies import DefaultOrderingPolicy


@dataclass
class AvailabilityEvent:
    """Represents a GPU availability state change in scenario time."""

    timestamp: float
    event_type: str  # "ADD" or "REMOVE"
    gpu_id: int

def parse_gpu_number(gpu_label:str)->int:
    digits = "".join(character for character in gpu_label if character.isdigit())
    if not digits:
        raise ValueError(f"GPU label {gpu_label} does not contain a number")
    return int(digits)

def load_trace_events(
        trace_filepath: str,
        trace_time_scale: float = 1.0,
        keep_every_nth_gpu: int = 1
        )-> list[AvailabilityEvent]:
    events = []
    with open(trace_filepath,newline="") as trace_file:
        for row in csv.DictReader(trace_file):
            gpu_id = parse_gpu_number(row["gpu_id"])
            if gpu_id % keep_every_nth_gpu != 0:
                continue
            event_type = row["event_type"].strip().upper()
            if event_type not in ("ADD", "REMOVE"):
                raise ValueError(f"unknown event type {event_type!r} in {trace_filepath}")
            timestamp_seconds = float(row["timestamp"])/1000.0 * trace_time_scale
            events.append(AvailabilityEvent(timestamp=timestamp_seconds,event_type=event_type, gpu_id=gpu_id))
            events.sort(key=lambda event: event.timestamp)
    return events

class TaskRunner(Protocol):
    def is_finished(self) -> bool:
        ...

    def terminate(self) -> None:
        ...


class FakeRunner:
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
        slots_per_gpu: int = 1,
    ) -> None:
        self.broker = broker
        # Sort events by ascending scenario timestamp
        self.events = sorted(events, key=lambda e: e.timestamp)
        self.time_scale = time_scale
        self.poll_seconds = poll_seconds
        self.slots_per_gpu = slots_per_gpu  # max tasks running at once on one GPU (bucket2: 4)
        self.logger = EventLogger(log_filepath)

        self.start_wall_time = time.monotonic()
        self.active_gpus: set[int] = set()
        self.worker_assignments: Dict[int, List[str]] = {}  # gpu_id -> running task_ids
        self.task_gpu: Dict[str, int] = {}  # task_id -> gpu_id
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

                # Preempt every task running on the evicted GPU
                for task_id in self.worker_assignments.pop(event.gpu_id, []):
                    self.task_gpu.pop(task_id, None)
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
            gpu_id = self.task_gpu.pop(task_id, -1)
            if gpu_id in self.worker_assignments:
                self.worker_assignments[gpu_id].remove(task_id)
                if not self.worker_assignments[gpu_id]:
                    del self.worker_assignments[gpu_id]

            task = self.broker.get_task(task_id)
            npts = task.rows if task else 0
            self.broker.complete(task_id)
            self.logger.log_event(
                self.scenario_time(), "FINISH", task_id, gpu_id, npts
            )

    def dispatch_idle_workers(self) -> None:
        """Least-busy scheduler: each READY task goes to the active GPU with a free slot that has the
        fewest running tasks (ties: lowest GPU id)."""
        while True:
            free_gpus = [
                gpu_id for gpu_id in self.active_gpus
                if len(self.worker_assignments.get(gpu_id, [])) < self.slots_per_gpu
            ]
            if not free_gpus:
                return  # every GPU is full
            gpu_id = min(free_gpus, key=lambda g: (len(self.worker_assignments.get(g, [])), g))
            task_id = self.broker.next_task()
            if task_id is None:
                return  # no READY task left
            self.start_task(task_id, gpu_id)

    def start_task(self, task_id: str, gpu_id: int) -> None:
        """Starts task_id on gpu_id with its measured (or heuristic) duration."""
        task = self.broker.get_task(task_id)
        npts = task.rows if task else 0

        # Measured task time if the task has one, else heuristic: 1 second per 100k vectors
        if task is not None and task.duration_sec is not None:
            simulated_duration = task.duration_sec
        else:
            simulated_duration = max(0.5, (npts / 100_000.0))
        runner = FakeRunner(
            task_id,
            base_duration_sec=simulated_duration,
            time_scale=self.time_scale,
        )

        self.worker_assignments.setdefault(gpu_id, []).append(task_id)
        self.task_gpu[task_id] = gpu_id
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
    import argparse

    parser = argparse.ArgumentParser(description="Dry run of the cluster simulator with 5 fake tasks")
    parser.add_argument("--slots", type=int, default=1, help="max tasks running at once per GPU")
    arguments = parser.parse_args()

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

    dry_run_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                               f"dry_run_events_slots{arguments.slots}.csv")
    sim = ClusterSimulator(
        broker=broker,
        events=events,
        log_filepath=dry_run_log,
        time_scale=1.0,
        slots_per_gpu=arguments.slots,
    )
    print("=== Starting Cluster Simulator Dry Run ===")
    sim.run()
    print(f"=== Simulation Complete. Log written to {dry_run_log} ===")