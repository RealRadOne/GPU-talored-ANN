"""What the simulator runs: measured bucket2 Step 6 tasks (task trace) and Lan's spot build as stages."""
from __future__ import annotations

import csv
import json
import os
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


# ---------------------------------------------------------------------------------------------------------
# Lan's spot build (bucketDemo/SPOT_DESIGN.md): only Step 4 (two rounds), Step 6 (two rounds) and prune run
# on spot GPUs; everything else is serial on the on-demand GPU. The build is a list of ("serial", seconds)
# and ("stage", StageSpec) steps, measured from her no-event runs in state/best_spot.json.

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEST_SPOT = os.path.join(REPO, ".claude/skills/optimize-bucket2-spot/state/best_spot.json")
INF = float("inf")


@dataclass
class StageSpec:
    name: str            # step4_r1, step6_r1, step4_r2, step6_r2, prune
    work_s: float        # total work, in seconds of one GPU working alone (from the G=1 run)
    host_rate: float     # cap on the stage's total rate (work-s per s): the host side (scatter, writes, I/O)
    batch_s: float       # work per commit (the in-flight work a preemption throws away)
    setup_s: float       # first lease of an instance epoch (cold start)
    lease_s: float       # every later lease
    min_steal_s: float   # steal only if the victim has at least this much work left


@dataclass
class LanBuild:
    gpus: int
    steps: list          # ("serial", seconds) | ("stage", StageSpec)
    source: str


# Per-stage settings not in best_spot.json. Measured from Lan's [spot] worker lines (runs/merge1 + spot5 on
# gwhiz1): Step 4 setup 0.16 s per instance epoch; Step 4 commits every ~4.2 M rows in round 1 (of 99.8 M);
# round 2 commits ~8.4 M rows but a preemption drops ~22 M, so round 2 is modelled by that loss; prune
# commits every ~682 buckets (of 195 k); Step 6 every bucket (modelled as 0.05 % of the stage to keep the
# event count low). Not measured (her logs print 0 for Step 6 / prune setup): Step 6 / prune setup and
# per-lease cost, from her code comments (prune: "each lease ~2-3 s fixed cost"; Step 6 rebuilds
# future_access per lease). Steal thresholds = her defaults (bucket.cu, prune_windowed.cpp).
STAGE_KNOBS = {
    #            batch_frac  setup_s lease_s min_steal_s
    "step4_r1": (0.042,      0.16,   0.05,   0.5),
    "step4_r2": (0.22,       0.16,   0.05,   0.5),
    "step6_r1": (0.0005,     1.0,    1.0,    4.0),
    "step6_r2": (0.0005,     1.0,    1.0,    4.0),
    "prune":    (0.0035,     2.5,    2.5,    4.0),
}
STAGE_ORDER = ["step4_r1", "step6_r1", "step4_r2", "step6_r2", "prune"]
MEASURED_G = (1, 2, 4)


def _stage_walls(run: dict) -> dict[str, float]:
    rep = run["spot_reps"][0]["stages"]
    assert [s["stage"] for s in rep] == ["step4", "step6", "step4", "step6", "prune"], rep
    return {name: float(str(s["wall"]).rstrip("s")) for name, s in zip(STAGE_ORDER, rep)}


def load_lan_build(gpus: int, best_spot: str = BEST_SPOT, dataset: str = "sift100M",
                   host_cap_scale: float = 1.0) -> LanBuild:
    """Serial times from the no-event run of the nearest measured G <= gpus; per stage, the work from the
    G=1 run and the host rate cap from that G's run (rate = work / (wall - setup), a cap only if < G).
    For an unmeasured G (8) the cap is multiplied by host_cap_scale: 1 = the host stays the limit
    (conservative), gpus / measured G = the cap grows with the GPUs (optimistic)."""
    runs = json.load(open(best_spot))["runs"]
    key = lambda g: f"{dataset}_auto_g{g}_spotnone"
    g_ref = max(g for g in MEASURED_G if g <= gpus)
    run = runs[key(g_ref)]
    walls, walls_g1 = _stage_walls(run), _stage_walls(runs[key(1)])
    step_times = run["bucket2_steps_s"]

    def step(prefix: str) -> float:   # "Step 4 (" -> Step 4, "Step 4.5", "Wait" -> final merge wait
        return sum(v for k, v in step_times.items() if k.startswith(prefix))
    s4_extra = step("Step 4 (") - walls["step4_r1"] - walls["step4_r2"]
    s6_extra = step("Step 6 (") - walls["step6_r1"] - walls["step6_r2"]
    prune_extra = run["stages_s"]["prune"] - walls["prune"]
    rest = run["build_total_s"] - sum(run["stages_s"].values())   # run_pipeline overhead (can be < 0: medians)
    bucket2_unlisted = run["stages_s"]["bucket2"] - sum(step_times.values())   # bucket2 time outside its steps

    def stage(name: str) -> StageSpec:
        frac, setup, lease, min_steal = STAGE_KNOBS[name]
        work = max(1e-9, walls_g1[name] - setup)
        rate = work / max(1e-9, walls[name] - setup)
        host_rate = (rate if rate < g_ref else INF) * (host_cap_scale if gpus > g_ref else 1.0)
        return StageSpec(name, work, host_rate, frac * work, setup, lease, min_steal)

    steps = [
        ("serial", step("Step 1 (") + step("Step 2 (") + step("Step 3 (") + s4_extra / 2),
        ("stage", stage("step4_r1")),
        ("serial", step("Step 4.5") + step("Step 5 (") / 2),
        ("stage", stage("step6_r1")),
        ("serial", s6_extra / 2 + s4_extra / 2),
        ("stage", stage("step4_r2")),
        ("serial", step("Step 5 (") / 2),
        ("stage", stage("step6_r2")),
        ("serial", s6_extra / 2 + step("Wait") + prune_extra),
        ("stage", stage("prune")),
        ("serial", run["stages_s"]["optimize_chunked"] + rest + bucket2_unlisted),
    ]
    return LanBuild(gpus, steps, f"{key(g_ref)} (G={g_ref}), work from {key(1)}")
