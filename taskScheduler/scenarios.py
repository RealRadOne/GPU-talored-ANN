from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# Compiled Regular Expressions matching C++ logging patterns
STEP_DONE_PATTERN = re.compile(
    r"===\s*Step\s*(\d+(?:\.\d+)?)\s*done\s*\[([\d\.]+)s\]", re.IGNORECASE
)

TOTAL_TIME_PATTERN = re.compile(
    r"Total:\s*([\d\.]+)s", re.IGNORECASE
)

RESTART_LINE_PATTERN = re.compile(
    r"\[SPOT METRICS\].*", re.IGNORECASE
)

RECOVERY_LATENCY_PATTERN = re.compile(
    r"Recovery Latency:\s*([\d\.]+)s", re.IGNORECASE
)

WORK_LOST_PATTERN = re.compile(
    r"Work Lost on Restart:\s*(\d+)\s*rows", re.IGNORECASE
)


@dataclass
class StepTiming:
    """Represents a single step completion metric parsed from the log."""
    step_id: str
    duration_seconds: float


@dataclass
class LogParseResult:
    """Container for all metrics extracted from a pipeline execution log."""
    completed_steps: List[StepTiming]
    total_elapsed_seconds: Optional[float]
    last_step_done: str
    restart_lines: List[str]
    recovery_latency_seconds: Optional[float]
    work_lost_rows: Optional[int]


def parse_completed_steps(log_text: str) -> List[StepTiming]:
    """Parses all '=== Step X done [Y.YYs]' lines from the log."""
    steps = []
    for match in STEP_DONE_PATTERN.finditer(log_text):
        step_id = match.group(1)
        duration = float(match.group(2))
        steps.append(StepTiming(step_id=step_id, duration_seconds=duration))
    return steps


def parse_total_elapsed_seconds(log_text: str) -> Optional[float]:
    """Extracts the final 'Total: X.XXXs' pipeline duration if completed."""
    match = TOTAL_TIME_PATTERN.search(log_text)
    if match:
        return float(match.group(1))
    return None


def parse_spot_metrics(log_text: str) -> tuple[List[str], Optional[float], Optional[int]]:
    """Extracts all [SPOT METRICS] log lines, recovery latency, and work lost."""
    restart_lines = [match.group(0) for match in RESTART_LINE_PATTERN.finditer(log_text)]
    
    latency_match = RECOVERY_LATENCY_PATTERN.search(log_text)
    recovery_latency = float(latency_match.group(1)) if latency_match else None

    work_lost_match = WORK_LOST_PATTERN.search(log_text)
    work_lost = int(work_lost_match.group(1)) if work_lost_match else None

    return restart_lines, recovery_latency, work_lost


def parse_pipeline_log(log_text: str) -> LogParseResult:
    """
    Pure function that parses raw log output text into a structured LogParseResult.
    """
    completed_steps = parse_completed_steps(log_text)
    total_time = parse_total_elapsed_seconds(log_text)
    restart_lines, recovery_latency, work_lost = parse_spot_metrics(log_text)

    last_step = completed_steps[-1].step_id if completed_steps else "(none)"

    return LogParseResult(
        completed_steps=completed_steps,
        total_elapsed_seconds=total_time,
        last_step_done=last_step,
        restart_lines=restart_lines,
        recovery_latency_seconds=recovery_latency,
        work_lost_rows=work_lost,
    )