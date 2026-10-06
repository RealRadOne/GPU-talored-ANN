#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

LINE_BUFFERED_PREFIX = ["stdbuf", "-oL"]
DEFAULT_MAX_PREEMPTIONS = 3
DEFAULT_STEP_DELAY_SECONDS = 0.5
MIN_UPTIME_SECONDS = 1.0
GPU_RELEASE_PAUSE_SECONDS = 1.0

OUTCOME_COMPLETED = "completed"
OUTCOME_PREEMPTED = "preempted"
OUTCOME_FAILED = "failed"

STEP_DONE_PATTERN = re.compile(r"^\s*(Step [\d.]+) done", re.MULTILINE)
TOTAL_SECONDS_PATTERN = re.compile(r"^\s*Total:\s*([\d.]+)s", re.MULTILINE)
PREVIOUS_RUN_PATTERN = re.compile(r"\[restart\] previous run was interrupted.*")
ROWS_ON_DISK_PATTERN = re.compile(r"\[restart\] result rows on disk:.*")
RECOVERY_LATENCY_PATTERN = re.compile(r"\[restart\] recovery latency:\s*([\d.]+)s")


@dataclass
class AttemptResult:
    index: int
    trigger: str
    outcome: str
    duration_seconds: float
    last_step_done: str
    previous_run_line: str
    rows_on_disk_line: str
    recovery_latency_seconds: Optional[float]
    log_path: str


@dataclass
class SimulationReport:
    schedule: str
    seed: int
    completed: bool
    attempts: List[AttemptResult]
    preemptions: int
    total_wall_seconds: float
    wasted_compute_seconds: float
    baseline_seconds: Optional[float] = None
    overhead_percent: Optional[float] = None
    cost_ratio_vs_on_demand: Optional[float] = None


def read_text(path: str) -> str:
    try:
        with open(path, errors="replace") as text_file:
            return text_file.read()
    except OSError:
        return ""


def find_line(pattern: "re.Pattern", text: str) -> str:
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


def find_number(pattern: "re.Pattern", text: str) -> Optional[float]:
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def completed_steps(text: str) -> List[str]:
    return STEP_DONE_PATTERN.findall(text)


class RunningAttempt:
    def __init__(self, command: List[str], log_path: str):
        self.log_path = log_path
        self.started_at = time.time()
        with open(log_path, "w") as log_file:
            self.process = subprocess.Popen(LINE_BUFFERED_PREFIX + command, stdout=log_file,
                                            stderr=subprocess.STDOUT, start_new_session=True)

    def is_running(self) -> bool:
        return self.process.poll() is None

    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    def completed_steps(self) -> List[str]:
        return completed_steps(read_text(self.log_path))

    def terminate(self, notice_seconds: float) -> None:
        if notice_seconds > 0:
            self.send_signal(signal.SIGTERM)
            if self.exits_within(notice_seconds):
                return
        self.send_signal(signal.SIGKILL)
        self.process.wait()

    def send_signal(self, signal_number: int) -> None:
        try:
            os.killpg(self.process.pid, signal_number)
        except ProcessLookupError:
            pass

    def exits_within(self, seconds: float) -> bool:
        try:
            self.process.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            return False


class UptimeTrigger:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def describe(self) -> str:
        return f"t+{self.seconds:.1f}s"

    def is_due(self, attempt: RunningAttempt) -> bool:
        return attempt.uptime_seconds() >= self.seconds


class StepTrigger:
    def __init__(self, step: str, delay_seconds: float):
        self.label = f"Step {step}"
        self.delay_seconds = delay_seconds
        self.step_seen_at: Optional[float] = None

    def describe(self) -> str:
        return f"{self.delay_seconds:.1f}s after {self.label} done"

    def is_due(self, attempt: RunningAttempt) -> bool:
        if self.step_seen_at is None and self.label in attempt.completed_steps():
            self.step_seen_at = time.time()
        return (self.step_seen_at is not None
                and time.time() - self.step_seen_at >= self.delay_seconds)


Schedule = Callable[[int, random.Random], Optional[object]]


def fixed_uptimes(argument: str) -> Schedule:
    uptimes = [float(item) for item in argument.split(",") if item]
    return lambda index, rng: UptimeTrigger(uptimes[index]) if index < len(uptimes) else None


def exponential_uptimes(argument: str) -> Schedule:
    mean_text, _, limit_text = argument.partition(":")
    mean_seconds = float(mean_text)
    limit = int(limit_text) if limit_text else DEFAULT_MAX_PREEMPTIONS

    def schedule(index: int, rng: random.Random) -> Optional[UptimeTrigger]:
        if index >= limit:
            return None
        return UptimeTrigger(max(MIN_UPTIME_SECONDS, rng.expovariate(1.0 / mean_seconds)))

    return schedule


def after_steps(argument: str) -> Schedule:
    entries = []
    for item in argument.split(","):
        step, _, delay_text = item.partition("+")
        entries.append((step, float(delay_text) if delay_text else DEFAULT_STEP_DELAY_SECONDS))

    def schedule(index: int, rng: random.Random) -> Optional[StepTrigger]:
        return StepTrigger(*entries[index]) if index < len(entries) else None

    return schedule


SCHEDULE_BUILDERS: Dict[str, Callable[[str], Schedule]] = {
    "fixed": fixed_uptimes,
    "exp": exponential_uptimes,
    "after-step": after_steps,
}


def parse_schedule(spec: str) -> Schedule:
    name, _, argument = spec.partition(":")
    if name not in SCHEDULE_BUILDERS:
        raise SystemExit(f"unknown schedule: {spec}")
    return SCHEDULE_BUILDERS[name](argument)


def wait_until_exit_or_preempted(attempt: RunningAttempt, trigger: Optional[object],
                                 notice_seconds: float, poll_seconds: float) -> bool:
    while attempt.is_running():
        if trigger is not None and trigger.is_due(attempt):
            attempt.terminate(notice_seconds)
            return True
        time.sleep(poll_seconds)
    return False


def classify_outcome(was_preempted: bool, return_code: Optional[int]) -> str:
    if was_preempted:
        return OUTCOME_PREEMPTED
    return OUTCOME_COMPLETED if return_code == 0 else OUTCOME_FAILED


def run_attempt(index: int, command: List[str], log_path: str, trigger: Optional[object],
                notice_seconds: float, poll_seconds: float) -> AttemptResult:
    attempt = RunningAttempt(command, log_path)
    was_preempted = wait_until_exit_or_preempted(attempt, trigger, notice_seconds, poll_seconds)
    duration_seconds = attempt.uptime_seconds()
    text = read_text(log_path)
    steps = completed_steps(text)
    return AttemptResult(
        index=index,
        trigger=trigger.describe() if trigger else "none",
        outcome=classify_outcome(was_preempted, attempt.process.returncode),
        duration_seconds=duration_seconds,
        last_step_done=steps[-1] if steps else "(none)",
        previous_run_line=find_line(PREVIOUS_RUN_PATTERN, text),
        rows_on_disk_line=find_line(ROWS_ON_DISK_PATTERN, text),
        recovery_latency_seconds=find_number(RECOVERY_LATENCY_PATTERN, text),
        log_path=log_path,
    )


def build_command(args: argparse.Namespace, output_dir: str) -> List[str]:
    return [args.binary, "-i", args.input, "-o", output_dir] + shlex.split(args.pipeline_args)


def run_baseline(args: argparse.Namespace, log_dir: str) -> float:
    baseline_dir = os.path.join(args.output, "baseline")
    result = run_attempt(0, build_command(args, baseline_dir),
                         os.path.join(log_dir, "baseline.log"), None, 0.0, args.poll)
    if result.outcome != OUTCOME_COMPLETED:
        raise SystemExit(f"baseline failed, see {result.log_path}")
    pipeline_total = find_number(TOTAL_SECONDS_PATTERN, read_text(result.log_path))
    baseline_seconds = pipeline_total or result.duration_seconds
    print(f"[spot_sim] baseline: {baseline_seconds:.1f}s")
    return baseline_seconds


def run_until_complete(args: argparse.Namespace, log_dir: str) -> Tuple[List[AttemptResult], float]:
    schedule = parse_schedule(args.schedule)
    rng = random.Random(args.seed)
    command = build_command(args, os.path.join(args.output, "spot"))
    results: List[AttemptResult] = []
    downtime_seconds = 0.0
    for index in range(args.max_attempts):
        trigger = schedule(len(results), rng)
        log_path = os.path.join(log_dir, f"attempt_{index + 1}.log")
        result = run_attempt(index + 1, command, log_path, trigger, args.notice, args.poll)
        results.append(result)
        print(f"[spot_sim] attempt {result.index}: {result.outcome} after "
              f"{result.duration_seconds:.1f}s (trigger: {result.trigger})")
        if result.outcome != OUTCOME_PREEMPTED:
            break
        time.sleep(args.provision_delay + GPU_RELEASE_PAUSE_SECONDS)
        downtime_seconds += args.provision_delay
    return results, downtime_seconds


def build_report(args: argparse.Namespace, results: List[AttemptResult],
                 downtime_seconds: float, baseline_seconds: Optional[float]) -> SimulationReport:
    attempt_seconds = sum(result.duration_seconds for result in results)
    preempted = [result for result in results if result.outcome == OUTCOME_PREEMPTED]
    total_wall_seconds = attempt_seconds + downtime_seconds
    report = SimulationReport(
        schedule=args.schedule,
        seed=args.seed,
        completed=results[-1].outcome == OUTCOME_COMPLETED,
        attempts=results,
        preemptions=len(preempted),
        total_wall_seconds=round(total_wall_seconds, 2),
        wasted_compute_seconds=round(sum(r.duration_seconds for r in preempted), 2),
        baseline_seconds=baseline_seconds,
    )
    if baseline_seconds:
        report.overhead_percent = round(
            100.0 * (total_wall_seconds - baseline_seconds) / baseline_seconds, 1)
        report.cost_ratio_vs_on_demand = round(
            attempt_seconds * (1.0 - args.spot_discount) / baseline_seconds, 3)
    return report


def print_summary(report: SimulationReport, discount: float) -> None:
    print("\n=== Spot simulation summary ===")
    for result in report.attempts:
        print(f"{result.index:>2} {result.outcome:<10} {result.duration_seconds:>7.1f}s  "
              f"last: {result.last_step_done}")
        for line in (result.previous_run_line, result.rows_on_disk_line):
            if line:
                print(f"     {line}")
    wasted_percent = 100.0 * report.wasted_compute_seconds / max(report.total_wall_seconds, 1e-9)
    print(f"completed: {report.completed}   preemptions: {report.preemptions}")
    print(f"total wall: {report.total_wall_seconds:.1f}s   "
          f"wasted: {report.wasted_compute_seconds:.1f}s ({wasted_percent:.0f}%)")
    if report.baseline_seconds:
        print(f"baseline: {report.baseline_seconds:.1f}s   overhead: {report.overhead_percent}%   "
              f"cost vs on-demand @ {discount:.0%} discount: {report.cost_ratio_vs_on_demand}x")


def parse_arguments() -> argparse.Namespace:
    default_binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                  "bucketDemo", "buildBucket", "build", "gpann_modular")
    parser = argparse.ArgumentParser(description="Simulate spot-instance preemptions of gpann_modular")
    parser.add_argument("--binary", default=default_binary)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pipeline-args", default="--knn-k 32 --neighbors-m 32 --iterations 1")
    parser.add_argument("--schedule", default="exp:60:3",
                        help="fixed:T1,T2 | exp:MEAN[:MAX] | after-step:S[+DELAY],S2[+DELAY2]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--notice", type=float, default=0.0,
                        help="seconds between SIGTERM and SIGKILL; 0 sends SIGKILL immediately")
    parser.add_argument("--provision-delay", type=float, default=0.0)
    parser.add_argument("--spot-discount", type=float, default=0.7)
    parser.add_argument("--poll", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not os.path.exists(args.binary):
        raise SystemExit(f"binary not found: {args.binary}")
    log_dir = os.path.join(args.output, "spot_logs")
    os.makedirs(log_dir, exist_ok=True)

    baseline_seconds = run_baseline(args, log_dir) if args.baseline else None
    results, downtime_seconds = run_until_complete(args, log_dir)
    report = build_report(args, results, downtime_seconds, baseline_seconds)

    with open(os.path.join(args.output, "spot_report.json"), "w") as report_file:
        json.dump(asdict(report), report_file, indent=2)
    print_summary(report, args.spot_discount)
    return 0 if report.completed else 1


if __name__ == "__main__":
    sys.exit(main())