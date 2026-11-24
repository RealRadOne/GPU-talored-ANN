from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from spotSim.broker import SegmentBroker, TaskBroker
from spotSim.policies import DefaultOrderingPolicy, LanAssignmentPolicy
from spotSim.tasks import LanBuild, StageSpec, load_lan_build

INF = float("inf")


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
    finish_time: float

    def is_finished(self, now: float) -> bool:
        ...

    def terminate(self) -> None:
        ...


class FakeRunner:
    """A task that runs for a known duration, starting at simulated time start_time."""

    def __init__(self, task_id: str, base_duration_sec: float, start_time: float) -> None:
        self.task_id = task_id
        self.duration = base_duration_sec
        self.start_time = start_time
        self.finish_time = start_time + base_duration_sec

    def is_finished(self, now: float) -> bool:
        return now >= self.finish_time

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
        self.writer.writerow([round(timestamp, 6), event_type, task_id, gpu_id, npts])
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
        slots_per_gpu: int = 1,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.broker = broker
        # Sort events by ascending scenario timestamp (storm speed-up is applied in load_trace_events)
        self.events = sorted(events, key=lambda e: e.timestamp)
        self.slots_per_gpu = slots_per_gpu  # max tasks running at once on one GPU (bucket2: 4)
        self.rng = rng  # tie-break between equally busy GPUs: random if given, else lowest GPU id
        self.logger = EventLogger(log_filepath)

        self.now = 0.0  # simulated time in seconds; run() moves it from event to event
        self.active_gpus: set[int] = set()
        self.worker_assignments: Dict[int, List[str]] = {}  # gpu_id -> running task_ids
        self.task_gpu: Dict[str, int] = {}  # task_id -> gpu_id
        self.active_runners: Dict[str, TaskRunner] = {}  # task_id -> TaskRunner

    def scenario_time(self) -> float:
        """Returns the current simulated time (seconds)."""
        return self.now

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
            if runner.is_finished(self.now):
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
        fewest running tasks (ties: random if rng is set, else lowest GPU id)."""
        while True:
            free_gpus = [
                gpu_id for gpu_id in self.active_gpus
                if len(self.worker_assignments.get(gpu_id, [])) < self.slots_per_gpu
            ]
            if not free_gpus:
                return  # every GPU is full
            fewest = min(len(self.worker_assignments.get(g, [])) for g in free_gpus)
            least_busy = sorted(g for g in free_gpus if len(self.worker_assignments.get(g, [])) == fewest)
            gpu_id = self.rng.choice(least_busy) if self.rng else least_busy[0]
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
            start_time=self.now,
        )

        self.worker_assignments.setdefault(gpu_id, []).append(task_id)
        self.task_gpu[task_id] = gpu_id
        self.active_runners[task_id] = runner
        self.logger.log_event(
            self.scenario_time(), "START", task_id, gpu_id, npts
        )

    def next_event_time(self) -> Optional[float]:
        """Earliest moment something happens: a running task finishes or a GPU is added/removed."""
        candidates = [runner.finish_time for runner in self.active_runners.values()]
        if self.events:
            candidates.append(self.events[0].timestamp)
        return min(candidates) if candidates else None

    def run(self) -> None:
        """Main simulation loop: handle everything due now, then jump straight to the next event
        (no waiting; the whole timeline is computed from the task times and the GPU trace)."""
        self.now = 0.0
        try:
            while not self.broker.all_done():
                self.process_due_events()
                self.reap_completed_workers()
                self.dispatch_idle_workers()
                if self.broker.all_done():
                    break
                next_time = self.next_event_time()
                if next_time is None:
                    print(f"[ClusterSimulator] stuck at t={self.now:.6f}: tasks left, "
                          "but no GPU and no more trace events")
                    break
                self.now = max(self.now, next_time)
        finally:
            self.logger.close()


# =========================================================================================================
# Lan's spot build (bucketDemo/SPOT_DESIGN.md, src/spot_sched.hpp) on top of the same pieces: the stages
# come from tasks.load_lan_build, the segments from broker.SegmentBroker, who gets what from
# policies.LanAssignmentPolicy. Differences from the task-level ClusterSimulator above:
# - GPU 0 is on-demand (never preempted; dedicated mode, G >= 2), GPUs 1..G-1 are spot instances 1..G-1.
#   G = 1 (shared mode): one spot instance 0 plus an on-demand rescue worker on the same GPU that only
#   works after every spot worker has been unavailable for rescue_after_s.
# - A worker commits after each batch; a preemption loses only the batch in flight, the rest of its
#   segment goes back to the pool. During a notice no batch starts that would end after
#   deadline - drain_margin_s. A worker pays setup_s on its first lease of an instance epoch, lease_s after.
# - Batches running at once share the stage's host rate cap (prune / Step 6 are host-bound).
# Event log (--log): npts holds work seconds (LEASE: segment size, COMMIT: batch, INTERRUPT: time lost).
# The availability trace is Lan's format (time_s,instance,event with notice|preempt|join; trace seconds,
# real = trace / scale) and starts when bucket2 starts.

LAN_EVENT_ORDER = {"notice": 0, "preempt": 1, "join": 2}


class InstanceTimeline:
    """Per-instance up / notice / down state over real time."""

    def __init__(self, rows: list[tuple[float, int, str]], scale: float) -> None:
        self.by_inst: Dict[int, List[tuple[float, str]]] = {}
        for t, inst, ev in sorted(rows, key=lambda r: (r[0], LAN_EVENT_ORDER[r[2]])):
            self.by_inst.setdefault(inst, []).append((t / scale, ev))
        self.times = sorted({t for evs in self.by_inst.values() for t, _ in evs})

    def state(self, inst: int, t: float) -> tuple[str, int, float]:
        """(state, epoch, deadline) at real time t; deadline = the coming preempt during a notice."""
        st, epoch = "up", 0
        evs = self.by_inst.get(inst, [])
        for te, ev in evs:
            if te > t:
                break
            if ev == "notice" and st == "up":
                st = "notice"
            elif ev == "preempt":
                st = "down"
            elif ev == "join":
                st, epoch = "up", epoch + 1
        deadline = INF
        if st == "notice":
            deadline = min((te for te, ev in evs if te > t and ev == "preempt"), default=INF)
        return st, epoch, deadline

    def next_time(self, t: float) -> float:
        for te in self.times:
            if te > t + 1e-12:
                return te
        return INF


def load_lan_trace(trace_filepath: str, scale: float = 60.0) -> InstanceTimeline:
    """Lan's spot_sim.py trace ('' = no events: every instance stays up)."""
    rows = []
    if trace_filepath:
        with open(trace_filepath, newline="") as trace_file:
            rows = [(float(r["time_s"]), int(r["instance"]), r["event"].strip()) for r in csv.DictReader(trace_file)]
    return InstanceTimeline(rows, scale)


class NullLogger:
    def log_event(self, *args) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass
class SpotWorker:
    idx: int
    inst: int                    # -1 = on-demand
    ondemand: bool
    setup_epoch: int = -1
    seg: int = -1                # leased segment, -1 = idle
    phase: str = ""              # "overhead" (setup / lease start) or "batch"
    left: float = 0.0            # overhead seconds or batch work left
    batch: Optional[tuple] = None  # (a, b) in flight
    phase_start: float = 0.0
    parked_epoch: int = -1       # drained in this epoch: wait for the next one


@dataclass
class StageResult:
    name: str
    wall: float = 0.0
    preempts: int = 0
    lost_s: float = 0.0          # wall seconds of setup / batches thrown away by preemptions
    setups: int = 0
    steals: int = 0
    drains: int = 0


class SpotStageSimulator:
    """One spot stage of Lan's build, event-driven like ClusterSimulator."""

    def __init__(self, spec: StageSpec, gpus: int, timeline: InstanceTimeline, start_time: float,
                 policy: Optional[LanAssignmentPolicy] = None, drain_margin_s: float = 0.25,
                 rescue_after_s: float = 2.0, logger=None) -> None:
        self.spec, self.timeline, self.now = spec, timeline, start_time
        self.policy = policy or LanAssignmentPolicy()
        self.drain_margin_s, self.rescue_after_s = drain_margin_s, rescue_after_s
        self.logger = logger or NullLogger()
        self.shared = gpus == 1
        if self.shared:
            self.workers = [SpotWorker(0, 0, False), SpotWorker(1, -1, True)]
        else:
            self.workers = [SpotWorker(0, -1, True)] + [SpotWorker(g, g, False) for g in range(1, gpus)]
        self.n_spot = sum(1 for w in self.workers if not w.ondemand)
        self.broker = SegmentBroker(spec.work_s, 1 if self.shared else gpus)
        self.result = StageResult(spec.name)
        self.spot_down_since: Optional[float] = None

    # -- instance state
    def state(self, w: SpotWorker) -> tuple[str, int, float]:
        return ("up", 0, INF) if w.ondemand else self.timeline.state(w.inst, self.now)

    def available(self, w: SpotWorker) -> bool:
        st, epoch, _ = self.state(w)
        return st == "up" and epoch != w.parked_epoch

    def any_spot_up(self) -> bool:
        return any(not w.ondemand and self.available(w) for w in self.workers)

    def log(self, event: str, w: SpotWorker, amount: float = 0.0) -> None:
        self.logger.log_event(self.now, event, f"{self.spec.name}:seg{w.seg}", w.idx, round(amount, 6))

    # -- rates: batches share the host cap, overhead runs at wall speed
    def batch_speed(self) -> float:
        n = sum(1 for w in self.workers if w.phase == "batch")
        return min(1.0, self.spec.host_rate / n) if n else 1.0

    def speed(self, w: SpotWorker, batch_speed: float) -> float:
        return batch_speed if w.phase == "batch" else 1.0

    # -- leases
    def free_lease(self, w: SpotWorker) -> None:
        if w.seg >= 0:
            self.broker.release(w.seg)
        w.seg, w.phase, w.batch, w.left = -1, "", None, 0.0

    def try_assign(self, w: SpotWorker) -> bool:
        got = self.policy.assign(self.broker, w.idx, w.ondemand, lambda i: self.available(self.workers[i]),
                                 self.spec.work_s, self.n_spot, self.spec.min_steal_s)
        if got is None:
            return False
        index, stolen = got
        self.result.steals += stolen
        self.broker.grant(index, w.idx)
        w.seg = index
        _, epoch, _ = self.state(w)
        cold = w.setup_epoch != epoch
        w.setup_epoch = epoch
        self.result.setups += cold
        w.phase, w.left, w.batch, w.phase_start = "overhead", self.spec.setup_s if cold else self.spec.lease_s, None, self.now
        self.log("STEAL" if stolen else "LEASE", w, self.broker.segments[index].left())
        return True

    def next_batch(self, w: SpotWorker) -> None:
        """After an overhead or a committed batch: start the next batch, or give the lease back."""
        s = self.broker.segments[w.seg]
        a, b = s.cursor, min(s.end, s.cursor + self.spec.batch_s)
        if b <= a + SegmentBroker.EPS:
            self.free_lease(w)
            return
        st, epoch, deadline = self.state(w)
        if st == "notice":
            estimate = (b - a) / max(self.batch_speed(), 1e-9)
            if self.now + estimate + self.drain_margin_s > deadline:
                self.result.drains += 1
                self.log("DRAIN", w)
                w.parked_epoch = epoch
                self.free_lease(w)
                return
        if self.shared and w.ondemand and self.any_spot_up():
            self.free_lease(w)   # rescue worker yields to the spot instance
            return
        self.broker.issue(w.seg, b)
        w.phase, w.left, w.batch, w.phase_start = "batch", b - a, (a, b), self.now

    # -- main loop
    def process_due(self) -> None:
        for w in self.workers:   # preemptions: drop what is in flight, segment back to the pool
            if not w.ondemand and w.seg >= 0 and self.state(w)[0] == "down":
                self.result.preempts += 1
                self.result.lost_s += self.now - w.phase_start
                self.log("INTERRUPT", w, self.now - w.phase_start)
                self.free_lease(w)
        for w in self.workers:   # finished overhead / batches
            if w.seg >= 0 and w.left <= SegmentBroker.EPS:
                if w.phase == "batch":
                    self.broker.commit(w.seg, w.batch[1])
                    self.log("COMMIT", w, w.batch[1] - w.batch[0])
                self.next_batch(w)
        if self.shared:          # rescue timer
            if self.any_spot_up() or any(not w.ondemand and w.seg >= 0 for w in self.workers):
                self.spot_down_since = None
            elif self.spot_down_since is None:
                self.spot_down_since = self.now
        waiting_spot = False     # idle workers: spot first; on-demand only if no spot worker is waiting
        for w in sorted(self.workers, key=lambda x: x.ondemand):
            if w.seg >= 0:
                continue
            if w.ondemand:
                if waiting_spot:
                    continue
                if self.shared:
                    if self.spot_down_since is None or self.now - self.spot_down_since < self.rescue_after_s - SegmentBroker.EPS:
                        continue
                    if any(x.seg >= 0 for x in self.workers):
                        continue
            elif not self.available(w):
                continue
            elif self.shared and self.workers[1].seg >= 0:
                continue   # same GPU: wait for the rescue worker to give it back
            if not self.try_assign(w) and not w.ondemand:
                waiting_spot = True

    def run(self) -> StageResult:
        start, steps = self.now, 0
        while True:
            self.process_due()
            if self.broker.all_done():
                break
            bs = self.batch_speed()
            candidates = [self.timeline.next_time(self.now)]
            candidates += [self.now + w.left / self.speed(w, bs) for w in self.workers if w.seg >= 0]
            if self.shared and self.spot_down_since is not None:
                candidates.append(self.spot_down_since + self.rescue_after_s)
            later = [c for c in candidates if c > self.now + SegmentBroker.EPS / 10]
            if not later or min(later) == INF:
                raise RuntimeError(f"{self.spec.name}: stuck at t={self.now:.2f} (no worker can run, no more events)")
            next_time = min(later)
            for w in self.workers:
                if w.seg >= 0:
                    w.left -= (next_time - self.now) * self.speed(w, bs)
            self.now = next_time
            steps += 1
            if steps > 5_000_000:
                raise RuntimeError("SpotStageSimulator: too many steps")
        self.result.wall = self.now - start
        return self.result


@dataclass
class BuildResult:
    gpus: int
    total_s: float
    stages: List[StageResult]

    @property
    def preempts(self) -> int:
        return sum(s.preempts for s in self.stages)

    @property
    def lost_s(self) -> float:
        return sum(s.lost_s for s in self.stages)


def simulate_lan_build(build: LanBuild, timeline: InstanceTimeline, log_filepath: Optional[str] = None,
                       **options) -> BuildResult:
    """Serial steps take their measured time; each spot stage runs on SpotStageSimulator."""
    logger = EventLogger(log_filepath) if log_filepath else NullLogger()
    now, stages = 0.0, []
    try:
        for kind, step in build.steps:
            if kind == "serial":
                now += step
            else:
                r = SpotStageSimulator(step, build.gpus, timeline, now, logger=logger, **options).run()
                stages.append(r)
                now += r.wall
    finally:
        logger.close()
    return BuildResult(build.gpus, now, stages)


def format_build_result(r: BuildResult) -> str:
    lines = [f"G={r.gpus} total {r.total_s:.1f} s, preempts {r.preempts}, lost {r.lost_s:.1f} s"]
    for s in r.stages:
        lines.append(f"  {s.name:9s} wall {s.wall:7.2f} s  preempts {s.preempts}  lost {s.lost_s:6.2f} s  "
                     f"setups {s.setups}  steals {s.steals}  drains {s.drains}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Lan's spot build under a trace (--gpus), or a dry run of the task-level simulator with 5 fake tasks")
    parser.add_argument("--gpus", type=int, default=None, help="simulate Lan's SIFT100M spot build on G GPUs")
    parser.add_argument("--lan-trace", default="", help="Lan-format trace (time_s,instance,event); '' = no events")
    parser.add_argument("--scale", type=float, default=60.0, help="trace seconds per real second (SPOT_SCALE)")
    parser.add_argument("--host-cap-scale", type=float, default=1.0, help="unmeasured G (8): scale the host caps")
    parser.add_argument("--log", default=None, help="write LEASE/STEAL/COMMIT/INTERRUPT/DRAIN events to this CSV")
    parser.add_argument("--slots", type=int, default=1, help="dry run: max tasks running at once per GPU")
    arguments = parser.parse_args()

    if arguments.gpus is not None:
        lan_build = load_lan_build(arguments.gpus, host_cap_scale=arguments.host_cap_scale)
        print(f"profile: {lan_build.source}")
        result = simulate_lan_build(lan_build, load_lan_trace(arguments.lan_trace, arguments.scale), arguments.log)
        print(format_build_result(result))
        raise SystemExit(0)

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
        slots_per_gpu=arguments.slots,
    )
    print("=== Starting Cluster Simulator Dry Run ===")
    sim.run()
    print(f"=== Simulation Complete. Log written to {dry_run_log} ===")