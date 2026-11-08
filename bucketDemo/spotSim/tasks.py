"""Load measured task traces into TaskBroker tasks."""
from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass
class TraceTask:
    task_id: str        # "r{round}_b{bucket}"
    order: int          # global processing order: all of round 0 by pos, then round 1, ...
    rows: int           # bucket_size
    duration_sec: float  # measured (data_ms + compute_ms) / 1000


def load_bucket2_tasks(csv_path: str) -> list[TraceTask]:
    """One task per row of a bucket2 task trace (BUCKET2_TASK_TRACE csv), in Lan's order."""
    with open(csv_path, newline="") as trace_file:
        rows = list(csv.DictReader(trace_file))
    rows.sort(key=lambda row: (int(row["call"]), int(row["pos"])))
    return [
        TraceTask(
            task_id=f"r{row['call']}_b{row['bucket']}",
            order=order,
            rows=int(row["bucket_size"]),
            duration_sec=(float(row["data_ms"]) + float(row["compute_ms"])) / 1000.0,
        )
        for order, row in enumerate(rows)
    ]
