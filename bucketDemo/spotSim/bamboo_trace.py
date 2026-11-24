"""Convert a Bamboo GPU trace into Lan's spot_sim.py trace format.

Bamboo trace (e.g. ~/scaleGANN/SchedulerArtifacts/GPU_Traces/p3-trace.csv):
    timestamp,event_type,gpu_id        timestamp in ms, event_type add|remove, gpu_id nodeN
    A node never comes back after its remove; a replacement shows up as a new node's add.

Lan's format (.claude/skills/optimize-bucket2-spot/scripts/spot_sim.py, bucketDemo/SPOT_DESIGN.md):
    time_s,instance,event              time in trace seconds, event notice|preempt|join
    Instances not mentioned stay up; a join brings a new instance up in the same slot (cold start).

Mapping: we rent one node per spot slot (instances 1..G-1 in dedicated mode, instance 0 for G=1).
At the window start each slot gets a node that is alive then (picked with --seed). When that node is
removed the slot is preempted; the next node added to the cluster while the slot is empty joins it.
Nodes added while every slot is full are not ours. The window must lie inside the trace (11.4 h for p3).

Run from bucketDemo/:
    python3 -m spotSim.bamboo_trace --gpus 4 -o spotSim/out/traces/bamboo_g4.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass

DEFAULT_BAMBOO = os.path.expanduser("~/scaleGANN/SchedulerArtifacts/GPU_Traces/p3-trace.csv")
EVENT_ORDER = {"notice": 0, "preempt": 1, "join": 2}


@dataclass
class BambooEvent:
    time_s: float
    kind: str   # "add" or "remove"
    node: str


def parse_node_number(node: str) -> int:
    digits = "".join(c for c in node if c.isdigit())
    return int(digits) if digits else -1


def load_bamboo(path: str) -> list[BambooEvent]:
    with open(path, newline="") as f:
        events = [BambooEvent(float(r["timestamp"]) / 1000.0, r["event_type"].strip().lower(), r["gpu_id"].strip())
                  for r in csv.DictReader(f)]
    events.sort(key=lambda e: (e.time_s, e.kind != "remove"))
    return events


def convert(events: list[BambooEvent], instances: list[int], offset_s: float, duration_s: float,
            notice_s: float, seed: int | None) -> list[tuple[float, int, str]]:
    """Lan-format events (time relative to offset_s) for the given spot instances.
    seed None: the slots start on the lowest-numbered alive nodes; else a seeded random pick."""
    rng = random.Random(seed)
    if offset_s + duration_s > events[-1].time_s:
        print(f"warning: window ends at {offset_s + duration_s:.0f} s, after the trace's last event "
              f"({events[-1].time_s:.0f} s); slots keep their last state", file=sys.stderr)
    alive: set[str] = set()
    i = 0
    while i < len(events) and events[i].time_s <= offset_s:
        (alive.add if events[i].kind == "add" else alive.discard)(events[i].node)
        i += 1
    slot_node: dict[int, str | None] = {}
    by_number = sorted(alive, key=lambda n: (parse_node_number(n), n))
    k_start = min(len(instances), len(alive))
    start_nodes = by_number[:k_start] if seed is None else rng.sample(by_number, k_start)
    out: list[tuple[float, int, str]] = []
    for k, inst in enumerate(instances):
        if k < len(start_nodes):
            slot_node[inst] = start_nodes[k]
        else:
            slot_node[inst] = None
            out.append((0.0, inst, "preempt"))   # no node for this slot at the start
    for e in events[i:]:
        t = e.time_s - offset_s
        if t > duration_s:
            break
        if e.kind == "remove":
            for inst, node in slot_node.items():
                if node == e.node:
                    if notice_s > 0:
                        out.append((max(0.0, t - notice_s), inst, "notice"))
                    out.append((t, inst, "preempt"))
                    slot_node[inst] = None
        elif e.kind == "add":
            empty = [inst for inst, node in slot_node.items() if node is None]
            if empty:
                slot_node[empty[0]] = e.node
                out.append((t, empty[0], "join"))
    out.sort(key=lambda x: (x[0], EVENT_ORDER[x[2]], x[1]))
    return out


def spot_instances(gpus: int) -> list[int]:
    """Spot instance ids for G GPUs: dedicated mode (G >= 2) = 1..G-1, shared mode (G = 1) = 0."""
    return [0] if gpus == 1 else list(range(1, gpus))


def write_trace(path: str, rows: list[tuple[float, int, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "instance", "event"])
        for t, inst, ev in rows:
            w.writerow([f"{t:.3f}", inst, ev])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bamboo", default=DEFAULT_BAMBOO)
    p.add_argument("--gpus", type=int, required=True, help="G of the run (MGPU); spot instances follow Lan's mode")
    p.add_argument("--offset-s", type=float, default=0.0, help="window start in the Bamboo trace (trace seconds)")
    p.add_argument("--duration-s", type=float, default=None, help="window length (trace seconds; default: to the trace end)")
    p.add_argument("--notice-s", type=float, default=0.0, help="notice before each preempt (0 = hard kill; AWS 120)")
    p.add_argument("--seed", type=int, default=1, help="which alive nodes the slots start on")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    events = load_bamboo(a.bamboo)
    duration_s = a.duration_s if a.duration_s is not None else events[-1].time_s - a.offset_s
    rows = convert(events, spot_instances(a.gpus), a.offset_s, duration_s, a.notice_s, a.seed)
    write_trace(a.output, rows)
    n_pre = sum(1 for r in rows if r[2] == "preempt")
    print(f"{a.output}: {len(rows)} events ({n_pre} preempt) for instances {spot_instances(a.gpus)}", file=sys.stderr)


if __name__ == "__main__":
    main()
