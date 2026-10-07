from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess  # nosec B404 - runs the local gpann_modular binary only
import sys
import time
from collections.abc import Callable
from enum import Enum

import numpy as np

DEFAULT_BINARY_FILEPATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "bucketDemo", "buildBucket", "build", "gpann_modular")

LINE_BUFFERED_PREFIX = ["stdbuf", "-oL", "-eL"]  # keeps the last lines before a kill
DEFAULT_STEP_DELAY_SECONDS = 0.5
DEFAULT_MAX_PREEMPTIONS = 3
MIN_UPTIME_SECONDS = 1.0
GPU_RELEASE_PAUSE_SECONDS = 1.0  # lets the driver free the killed process's GPU memory
KNN_HEADER_BYTES = 12  # int64 row count + int32 neighbor count, as in RunningKnnFile
COMPARE_CHUNK_ROWS = 100_000
BASELINE_INDEX = 0  # the uninterrupted run; spot attempts count from 1

STEP_DONE_PATTERN = re.compile(r"^\s*(Step [\d.]+) done", re.MULTILINE)
RESTART_LINE_PATTERN = re.compile(r"\[restart\] (previous run|result rows).*")
RECOVERY_LATENCY_PATTERN = re.compile(r"\[restart\] recovery latency:\s*([\d.]+)s")


class Outcome(str, Enum):
    COMPLETED = "completed"
    PREEMPTED = "preempted"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Everything the command line configures."""

    binary_filepath: str
    input_filepath: str
    output_root: str
    pipeline_args: list[str]
    schedule_specification: str
    seed: int
    should_run_baseline: bool
    max_attempts: int
    notice_seconds: float  # grace period between SIGTERM and SIGKILL; 0 means SIGKILL only
    provision_delay_seconds: float
    spot_discount: float
    poll_seconds: float


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    """What one attempt did before it exited or was preempted."""

    index: int
    trigger_description: str
    outcome: Outcome
    duration_seconds: float
    last_step_done: str
    restart_lines: list[str]
    recovery_latency_seconds: float | None


@dataclasses.dataclass(frozen=True)
class SimulationReport:
    """The summary written to spot_report.json."""

    attempts: list[AttemptResult]
    is_completed: bool
    total_wall_seconds: float
    wasted_compute_seconds: float  # every preempted attempt is wasted until resume exists
    baseline_seconds: float | None = None
    overhead_percent: float | None = None
    cost_ratio_vs_on_demand: float | None = None
    identical_rows_vs_baseline: float | None = None  # also None when Step 6 was skipped


def main() -> int:
    """Run the simulation, write spot_report.json and print a summary."""
    config = parse_config(sys.argv[1:])
    if not os.path.exists(config.binary_filepath):
        raise FileNotFoundError(f"binary not found: {config.binary_filepath}")
    baseline_seconds = run_baseline(config) if config.should_run_baseline else None
    report = build_report(run_until_complete(config), baseline_seconds, config)
    with open(os.path.join(config.output_root, "spot_report.json"), "w") as report_file:
        json.dump(dataclasses.asdict(report), report_file, indent=2)
    print_report(report)
    return 0 if report.is_completed else 1


def run_baseline(config: SimulationConfig) -> float:
    """Run the pipeline once without preemption and return its wall-clock seconds."""
    result = run_attempt(config, BASELINE_INDEX, None)
    if result.outcome is not Outcome.COMPLETED:
        raise RuntimeError("baseline run failed, see spot_logs/attempt_0.log")
    print(f"[spot_sim] baseline: {result.duration_seconds:.1f}s")
    return result.duration_seconds


def run_until_complete(config: SimulationConfig) -> list[AttemptResult]:
    """Restart the pipeline after every preemption until it completes or fails."""
    schedule = parse_schedule(config.schedule_specification, config.seed)
    attempts: list[AttemptResult] = []
    for index in range(1, config.max_attempts + 1):
        result = run_attempt(config, index, schedule(index - 1))
        attempts.append(result)
        print(f"[spot_sim] attempt {index}: {result.outcome.value} after "
              f"{result.duration_seconds:.1f}s")
        if result.outcome is not Outcome.PREEMPTED:
            break
        time.sleep(config.provision_delay_seconds + GPU_RELEASE_PAUSE_SECONDS)
    return attempts


def run_attempt(config: SimulationConfig, index: int,
                trigger: UptimeTrigger | StepTrigger | None) -> AttemptResult:
    """Run one attempt until it exits or its trigger preempts it.

    The baseline runs in output_root/baseline and spot attempts in output_root/spot. Each
    starts from an empty folder on its first attempt, so restart state from an earlier
    simulation cannot leak in; later attempts reuse the folder, like a real restart.
    """
    folder = "baseline" if index == BASELINE_INDEX else "spot"
    output_dir = os.path.join(config.output_root, folder)
    if index <= 1 and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    log_filepath = os.path.join(config.output_root, "spot_logs", f"attempt_{index}.log")
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    command = [config.binary_filepath, "-i", config.input_filepath, "-o", output_dir]
    attempt = RunningAttempt(command + config.pipeline_args, log_filepath)
    was_signalled = attempt.wait_or_preempt(trigger, config)
    trigger_description = trigger.describe() if trigger else "none"
    return attempt.build_result(index, trigger_description, was_signalled)


def classify_outcome(was_signalled: bool, return_code: int) -> Outcome:
    """Exit code 0 counts as completed even if signalled: it finished before the signal."""
    if return_code == 0:
        return Outcome.COMPLETED
    return Outcome.PREEMPTED if was_signalled else Outcome.FAILED


class RunningAttempt:
    """One gpann_modular process, in its own process group, logging to a file."""

    def __init__(self, command: list[str], log_filepath: str) -> None:
        self.log_filepath = log_filepath
        self.started_at = time.monotonic()
        with open(log_filepath, "w") as log_file:
            self.process = subprocess.Popen(  # nosec B603 - argument list, no shell
                LINE_BUFFERED_PREFIX + command, stdout=log_file,
                stderr=subprocess.STDOUT, start_new_session=True)

    def uptime_seconds(self) -> float:
        """Return seconds since the attempt started."""
        return time.monotonic() - self.started_at

    def read_log(self) -> str:
        """Return the log so far; bytes a kill cut mid-character are replaced."""
        with open(self.log_filepath, errors="replace") as log_file:
            return log_file.read()

    def build_result(self, index: int, trigger_description: str,
                     was_signalled: bool) -> AttemptResult:
        """Build the attempt's result from it****it code and log, once it ha****ited."""
        log_text = self.read_log()
        steps = STEP_DONE_PATTERN.findall(log_text)
        latency_match = RECOVERY_LATENCY_PATTERN.search(log_text)
        return AttemptResult(
            index=index,
            trigger_description=trigger_description,
            outcome=classify_outcome(was_signalled, self.process.returncode),
            duration_seconds=round(self.uptime_seconds(), 2),
            last_step_done=steps[-1] if steps else "(none)",
            restart_lines=[match.group(0) for match in RESTART_LINE_PATTERN.finditer(log_text)],
            recovery_latency_seconds=float(latency_match.group(1)) if latency_match else None,
        )

    def wait_or_preempt(self, trigger: UptimeTrigger | StepTrigger | None,
                        config: SimulationConfig) -> bool:
        """Wait for exit, preempting when the trigger is due; return True if signalled."""
        while self.process.poll() is None:
            if trigger is not None:
                trigger.observe(self)
                if trigger.is_due():
                    self.preempt(config.notice_seconds)
                    return True
            time.sleep(config.poll_seconds)
        return False

    def preempt(self, notice_seconds: float) -> None:
        """Send SIGTERM, wait up to notice_seconds, then SIGKILL, as a spot provider would."""
        if notice_seconds > 0:
            self.signal_process_group(signal.SIGTERM)
            # A timeout i****pected while the pipeline has no SIGTERM handler.
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=notice_seconds)
        if self.process.poll() is None:
            self.signal_process_group(signal.SIGKILL)
            self.process.wait()

    def signal_process_group(self, signal_number: signal.Signals) -> None:
        """Signal stdbuf and gpann_modular together."""
        # The process may exit between the trigger check and the signal.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal_number)


class UptimeTrigger:
    """Preempts an attempt after a fixed number of seconds."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.latest_uptime_seconds = 0.0

    def describe(self) -> str:
        """Return a short description for the summary."""
        return f"t+{self.seconds:.1f}s"

    def observe(self, attempt: RunningAttempt) -> None:
        """Record how long the attempt has run."""
        self.latest_uptime_seconds = attempt.uptime_seconds()

    def is_due(self) -> bool:
        """Return True once the attempt has run for the configured time."""
        return self.latest_uptime_seconds >= self.seconds


class StepTrigger:
    """Preempts delay_seconds after the occurrence-th "Step <step_number> done" line."""

    def __init__(self, step_number: str, occurrence: int, delay_seconds: float) -> None:
        self.step_label = f"Step {step_number}"
        self.occurrence = occurrence
        self.delay_seconds = delay_seconds
        self.step_seen_at: float | None = None

    def describe(self) -> str:
        """Return a short description for the summary."""
        return f"{self.delay_seconds:.1f}s after {self.step_label} done (#{self.occurrence})"

    def observe(self, attempt: RunningAttempt) -> None:
        """Record the moment the step first reaches the wanted occurrence."""
        step_count = STEP_DONE_PATTERN.findall(attempt.read_log()).count(self.step_label)
        if self.step_seen_at is None and step_count >= self.occurrence:
            self.step_seen_at = time.monotonic()

    def is_due(self) -> bool:
        """Return True once the delay has passed after the step finished."""
        return (self.step_seen_at is not None
                and time.monotonic() - self.step_seen_at >= self.delay_seconds)


Schedule = Callable[[int], UptimeTrigger | StepTrigger | None]


def parse_schedule(specification: str, seed: int) -> Schedule:
    """Turn "fixed:40,2*******p:MEAN[:MAX]" or "after-step:4+1,6@2+2" into a schedule.

    Raises:
        ValueError: If the schedule name is unknown.
    """
    name, _, argument = specification.partition(":")
    entries = [entry for entry in argument.split(",") if entry]
    if name == "fixed":
        uptimes = [float(entry) for entry in entries]
        return lambda index: UptimeTrigger(uptimes[index]) if index < len(uptimes) else None
    if name == "after-step":
        triggers = [parse_step_entry(entry) for entry in entries]
        return lambda index: StepTrigger(*triggers[index]) if index < len(triggers) else None
    if name == "exp":
        return exponential_schedule(argument, seed)
    raise ValueError(f"unknown schedule {specification!r}; use fixed, exp or after-step")


def parse_step_entry(entry: str) -> tuple[str, int, float]:
    """Parse STEP[@OCCURRENCE][+DELAY], for example "6@2+5", into its three parts."""
    step_text, _, delay_text = entry.partition("+")
    step_number, _, occurrence_text = step_text.partition("@")
    occurrence = int(occurrence_text) if occurrence_text else 1
    delay_seconds = float(delay_text) if delay_text else DEFAULT_STEP_DELAY_SECONDS
    return step_number, occurrence, delay_seconds


def exponential_schedule(argument: str, seed: int) -> Schedule:
    """Draw uptimes from an exponential distribution, as in a Poisson preemption model."""
    mean_text, _, limit_text = argument.partition(":")
    preemption_limit = int(limit_text) if limit_text else DEFAULT_MAX_PREEMPTIONS
    random_generator = random.Random(seed)  # nosec B311 - simulation timing, not security

    def schedule(index: int) -> UptimeTrigger | None:
        if index >= preemption_limit:
            return None
        uptime_seconds = random_generator.expovariate(1.0 / float(mean_text))
        return UptimeTrigger(max(MIN_UPTIME_SECONDS, uptime_seconds))

    return schedule


def build_report(attempts: list[AttemptResult], baseline_seconds: float | None,
                 config: SimulationConfig) -> SimulationReport:
    """Summarize the attempts and, if there was a baseline, compare against it."""
    attempt_seconds = sum(result.duration_seconds for result in attempts)
    preempted_seconds = sum(result.duration_seconds for result in attempts
                            if result.outcome is Outcome.PREEMPTED)
    downtime_seconds = (len(attempts) - 1) * config.provision_delay_seconds
    total_wall_seconds = attempt_seconds + downtime_seconds
    report = SimulationReport(
        attempts=attempts, is_completed=attempts[-1].outcome is Outcome.COMPLETED,
        total_wall_seconds=round(total_wall_seconds, 2),
        wasted_compute_seconds=round(preempted_seconds, 2))
    if baseline_seconds is None:
        return report
    return dataclasses.replace(
        report,
        baseline_seconds=baseline_seconds,
        overhead_percent=round(100.0 * (total_wall_seconds / baseline_seconds - 1), 1),
        cost_ratio_vs_on_demand=round(
            attempt_seconds * (1 - config.spot_discount) / baseline_seconds, 3),
        identical_rows_vs_baseline=compare_graphs(config.output_root))


def compare_graphs(output_root: str) -> float | None:
    """Return the share of identical rows in the spot and baseline vector_knn.bin files."""
    filepaths = [glob.glob(os.path.join(output_root, folder, "k*p*m*", "vector_knn.bin"))
                 for folder in ("spot", "baseline")]
    if not all(filepaths):
        return None  # Step 6 was skipped (--neighbors-m 0)
    spot_ids, baseline_ids = (load_neighbor_ids(matches[0]) for matches in filepaths)
    if spot_ids.shape != baseline_ids.shape:
        raise ValueError(f"graph shape {spot_ids.shape} differs from {baseline_ids.shape}")
    identical_row_count = 0
    for row_start in range(0, len(spot_ids), COMPARE_CHUNK_ROWS):
        rows = slice(row_start, row_start + COMPARE_CHUNK_ROWS)
        identical_row_count += int((spot_ids[rows] == baseline_ids[rows]).all(axis=1).sum())
    return round(identical_row_count / len(spot_ids), 6)


def load_neighbor_ids(knn_filepath: str) -> np.ndarray:
    """Map vector_knn.bin as an (N, M) int32 array without loading it into memory."""
    row_count = int(np.fromfile(knn_filepath, dtype=np.int64, count=1)[0])
    neighbor_count = int(np.fromfile(knn_filepath, dtype=np.int32, count=1, offset=8)[0])
    return np.memmap(knn_filepath, dtype=np.int32, mode="r", offset=KNN_HEADER_BYTES,
                     shape=(row_count, neighbor_count))


def print_report(report: SimulationReport) -> None:
    """Print one line per attempt with its restart lines, then the totals."""
    print("\n=== Spot simulation summary ===")
    for result in report.attempts:
        print(f"{result.index:>2} {result.outcome.value:<10} {result.duration_seconds:>7.1f}s"
              f"  last: {result.last_step_done}  (trigger: {result.trigger_description})")
        for line in result.restart_lines:
            print(f"     {line}")
        if result.recovery_latency_seconds is not None:
            print(f"     recovery latency: {result.recovery_latency_seconds:.1f}s")
    print(f"completed: {report.is_completed}   total wall: {report.total_wall_seconds:.1f}s"
          f"   wasted: {report.wasted_compute_seconds:.1f}s")
    if report.baseline_seconds is not None:
        print(f"baseline: {report.baseline_seconds:.1f}s   overhead: {report.overhead_percent}%"
              f"   cost vs on-demand: {report.cost_ratio_vs_on_demand}x"
              f"   identical rows vs baseline: {report.identical_rows_vs_baseline}")


def parse_config(command_line: list[str]) -> SimulationConfig:
    """Parse the command line into a configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", default=DEFAULT_BINARY_FILEPATH)
    parser.add_argument("--input", required=True, help="dataset, for example data.fbin")
    parser.add_argument("--output", required=True, help="root folder for runs and logs")
    parser.add_argument("--pipeline-args", default="--knn-k 32 --neighbors-m 32 --iterations 3")
    parser.add_argument("--schedule", default="exp:60:3",
                        help="fixed:T1,T2 | exp:MEAN[:MAX] | after-step:S[@N][+DELAY],...")
    parser.add_argument("--seed", type=int, default=0, help="seed for the exp schedule")
    parser.add_argument("--baseline", action="store_true", help="run once uninterrupted first")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--notice", type=float, default=0.0, help="SIGTERM to SIGKILL seconds")
    parser.add_argument("--provision-delay", type=float, default=0.0,
                        help="simulated seconds to get a new instance after a preemption")
    parser.add_argument("--spot-discount", type=float, default=0.7)
    parser.add_argument("--poll", type=float, default=0.1, help="seconds between checks")
    arguments = parser.parse_args(command_line)
    return SimulationConfig(
        binary_filepath=arguments.binary, input_filepath=arguments.input,
        output_root=arguments.output, pipeline_args=shlex.split(arguments.pipeline_args),
        schedule_specification=arguments.schedule, seed=arguments.seed,
        should_run_baseline=arguments.baseline, max_attempts=arguments.max_attempts,
        notice_seconds=arguments.notice, provision_delay_seconds=arguments.provision_delay,
        spot_discount=arguments.spot_discount, poll_seconds=arguments.poll)


if __name__ == "__main__":
    sys.exit(main())
