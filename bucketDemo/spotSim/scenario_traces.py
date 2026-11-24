"""Spot traces for Lan's bucket2 from our trace generator's scenarios (A / B / C), for 4 and 8 GPUs.

Inputs (spotSim/scenario_inputs/): the scenario traces made by generate_synthetic_trace (SchedulerArtifacts/utils.py, run from
run_pipeline.py on feature/Sinha-expt): Weibull node lifetimes, exponential wave arrivals and geometric
batch sizes fitted to Bamboo p3, then modified per scenario:
    A high volatility     lifetimes x0.2, arrivals x5
    B burst arrivals      arrivals x0.5, batches ~5x larger
    C sustained capacity  lifetimes x3, arrivals x0.5
Each is converted with bamboo_trace.convert (start at time 0, slots on the lowest-numbered alive nodes, no
notice) into Lan's format, and simulated with cluster_sim.simulate_lan_build (SIFT100M, her calibrated build).

Run from bucketDemo/:   python3 -m spotSim.scenario_traces
Writes spotSim/experimental_traces/g<G>_<scenario>.csv and predictions.csv.
"""
from __future__ import annotations

import argparse
import csv
import os

from spotSim.bamboo_trace import convert, load_bamboo, spot_instances, write_trace
from spotSim.cluster_sim import InstanceTimeline, simulate_lan_build
from spotSim.tasks import load_lan_build

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_inputs")   # copies of ~/scaleGANN/SchedulerArtifacts/GPU_Traces/scenario_*.csv
SCENARIOS = {
    "A_high_volatility": "scenario_a_high_volatility.csv",
    "B_burst_arrivals": "scenario_b_burst_arrivals.csv",
    "C_sustained_capacity": "scenario_c_sustained_capacity.csv",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experimental_traces")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gpus", type=int, nargs="+", default=[4, 8])
    p.add_argument("--trace-dir", default=TRACE_DIR)
    p.add_argument("--scale", type=float, default=60.0, help="trace s per real s (Lan's SPOT_SCALE)")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    rows = []
    for g in a.gpus:
        build = load_lan_build(g)
        base = simulate_lan_build(build, InstanceTimeline([], a.scale)).total_s
        for name, fname in SCENARIOS.items():
            src = load_bamboo(os.path.join(a.trace_dir, fname))
            events = convert(src, spot_instances(g), 0.0, src[-1].time_s, 0.0, None)
            path = os.path.join(a.out, f"g{g}_{name}.csv")
            write_trace(path, events)
            r = simulate_lan_build(build, InstanceTimeline(events, a.scale))
            in_run = sum(1 for t, _, ev in events if ev == "preempt" and t / a.scale <= r.total_s)
            rows.append(dict(trace=os.path.basename(path), gpus=g, scenario=name, scale=a.scale,
                             covers_real_s=round(events[-1][0] / a.scale, 1) if events else 0.0,
                             no_event_s=round(base, 1), predicted_s=round(r.total_s, 1),
                             slowdown=round(r.total_s / base, 3), preempts_in_run=in_run,
                             lost_s=round(r.lost_s, 1)))
            print(f"G={g} {name:22s} -> {r.total_s:6.1f} s (x{r.total_s / base:.3f}), "
                  f"{in_run} preempts during the build, trace covers {rows[-1]['covers_real_s']} s")
    with open(os.path.join(a.out, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
