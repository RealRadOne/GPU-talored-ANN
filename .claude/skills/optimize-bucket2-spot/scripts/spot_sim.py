#!/usr/bin/env python3
"""spot_sim.py — spot instance interruption simulator for the bucket2 build (local simulation).

Subcommands
  gen     write a synthetic trace CSV (time_s,instance,event with event in notice|preempt|join)
  run     replay a trace in (compressed) real time: write the mmap state file <dir>/state.bin that
          bucket2 / prune_windowed read (src/spot_sched.hpp), and send SIGUSR1 to every process that
          registered itself in <dir>/pids/ ("the GPU worker receives the interruption signal")
  show    print the current state file

State file layout (little endian, must match spot_sched.hpp):
  header 64 B: magic u32 'SPOT', version u32 1, n_inst u32, pad u32, seq u64 (seqlock: odd while
               writing), t0_ns i64 (CLOCK_REALTIME at start), scale f64, 24 B reserved
  n_inst x 64 B: state u32 (0 up, 1 notice, 2 down), ondemand u32, epoch u64 (+1 per join),
               deadline_ns i64 (notice: preemption time; down: time it went down), changed_ns i64,
               n_notice u32, n_preempt u32, n_join u32, pad u32, 16 B reserved

Trace semantics: a `notice` for instance i announces the next `preempt` of i (deadline = that preempt's
time). A `preempt` without a preceding notice is a hard kill. `join` brings a new instance up in the same
slot (epoch + 1, cold start). Instances not mentioned stay up. The real (Bamboo) traces will get a
converter to this format later.
"""
import argparse
import csv
import heapq
import mmap
import os
import random
import signal
import struct
import sys
import time

MAGIC, VERSION = 0x544F5053, 1
HDR = struct.Struct("<IIIIQqd24x")
INST = struct.Struct("<IIQqqIIII16x")
assert HDR.size == 64 and INST.size == 64
UP, NOTICE, DOWN = 0, 1, 2
NAMES = {UP: "up", NOTICE: "notice", DOWN: "down"}


def now_ns():
    return time.clock_gettime_ns(time.CLOCK_REALTIME)


# ----------------------------------------------------------------------------- gen
def cmd_gen(a):
    """Burst model: a global Poisson process of preemption events with rate n/mtti; each event takes
    1 instance, or with probability --correlated a random group of 2..n currently-up instances (Bamboo
    observed correlated reclaims). Down time ~ Exp(mean --down-s), then join. A fraction --hard-frac of
    preemptions has no notice; the rest get a --notice-s notice (shorter if the instance joined recently)."""
    rng = random.Random(a.seed)
    insts = [int(x) for x in a.instances.split(",") if x.strip() != ""]
    if not insts:
        sys.exit("gen: --instances is empty")
    n = len(insts)
    up_since = {i: 0.0 for i in insts}
    down_until = {}
    ev = []
    t = a.start_s
    rate = n / a.mtti_s
    joins = []   # heap of (time, inst)
    while True:
        t += rng.expovariate(rate)
        # process joins that happen before t
        while joins and joins[0][0] <= t:
            tj, i = heapq.heappop(joins)
            if tj < a.duration_s:
                ev.append((tj, i, "join"))
            up_since[i] = tj
            down_until.pop(i, None)
        if t >= a.duration_s:
            break
        upl = [i for i in insts if i not in down_until]
        if not upl:
            continue
        k = 1
        if len(upl) > 1 and rng.random() < a.correlated:
            k = rng.randint(2, len(upl))
        for i in rng.sample(upl, k):
            if rng.random() >= a.hard_frac:
                tn = max(up_since[i], t - a.notice_s)
                if tn < t:
                    ev.append((tn, i, "notice"))
            ev.append((t, i, "preempt"))
            d = rng.expovariate(1.0 / a.down_s) if a.down_s > 0 else 0.0
            down_until[i] = t + max(d, a.min_down_s)
            heapq.heappush(joins, (down_until[i], i))
    ev.sort(key=lambda e: (e[0], {"notice": 0, "preempt": 1, "join": 2}[e[2]], e[1]))
    out = open(a.output, "w", newline="") if a.output != "-" else sys.stdout
    w = csv.writer(out)
    w.writerow(["time_s", "instance", "event"])
    for tt, i, e in ev:
        w.writerow([f"{tt:.3f}", i, e])
    if out is not sys.stdout:
        out.close()
    np_ = sum(1 for e in ev if e[2] == "preempt")
    nn = sum(1 for e in ev if e[2] == "notice")
    print(f"gen: {len(ev)} events ({np_} preempt, {nn} with notice) over {a.duration_s:.0f} s, instances {insts}",
          file=sys.stderr)


def read_trace(path):
    ev = []
    with open(path) as f:
        for row in csv.DictReader(f):
            e = row["event"].strip()
            if e not in ("notice", "preempt", "join"):
                raise SystemExit(f"bad event {e!r} in {path}")
            ev.append((float(row["time_s"]), int(row["instance"]), e))
    ev.sort(key=lambda x: (x[0], {"notice": 0, "preempt": 1, "join": 2}[x[2]], x[1]))
    return ev


# ----------------------------------------------------------------------------- run
class State:
    def __init__(self, path, n_inst, ondemand, scale):
        self.n = n_inst
        size = HDR.size + INST.size * n_inst
        fd = os.open(path + ".tmp", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, size)
        self.mm = mmap.mmap(fd, size)
        os.close(fd)
        self.seq = 0
        self.t0 = now_ns()
        HDR.pack_into(self.mm, 0, MAGIC, VERSION, n_inst, 0, 0, self.t0, float(scale))
        self.inst = []
        for i in range(n_inst):
            rec = dict(state=UP, ondemand=1 if i in ondemand else 0, epoch=0, deadline=0, changed=self.t0,
                       n_notice=0, n_preempt=0, n_join=0)
            self.inst.append(rec)
            self._write(i)
        self.mm.flush()
        os.rename(path + ".tmp", path)   # readers only ever see a complete file

    def _write(self, i):
        r = self.inst[i]
        INST.pack_into(self.mm, HDR.size + INST.size * i, r["state"], r["ondemand"], r["epoch"], r["deadline"],
                       r["changed"], r["n_notice"], r["n_preempt"], r["n_join"], 0)

    def update(self, i, **kw):
        self.seq += 1
        struct.pack_into("<Q", self.mm, 16, self.seq)   # odd: writing
        self.inst[i].update(kw)
        self.inst[i]["changed"] = now_ns()
        self._write(i)
        self.seq += 1
        struct.pack_into("<Q", self.mm, 16, self.seq)   # even: consistent


def signal_pids(d):
    pd = os.path.join(d, "pids")
    sent = []
    try:
        names = os.listdir(pd)
    except FileNotFoundError:
        return sent
    for nm in names:
        try:
            pid = int(nm)
        except ValueError:
            continue
        try:
            os.kill(pid, signal.SIGUSR1)
            sent.append(pid)
        except ProcessLookupError:
            try:
                os.unlink(os.path.join(pd, nm))
            except FileNotFoundError:
                pass
        except PermissionError:
            pass
    return sent


def cmd_run(a):
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(os.path.join(a.dir, "pids"), exist_ok=True)
    ondemand = set(int(x) for x in a.ondemand.split(",") if x.strip() != "")
    st = State(os.path.join(a.dir, "state.bin"), a.n_inst, ondemand, a.scale)
    ev = read_trace(a.trace) if a.trace else []
    # drop events of instances outside [0, n_inst) or on-demand ones; apply offset
    ev = [(t - a.offset_s, i, e) for (t, i, e) in ev if 0 <= i < a.n_inst and i not in ondemand and t >= a.offset_s]
    # real-time schedule: trace seconds / scale; notice length optionally overridden in real seconds
    sched = []
    pending_notice = {}
    last_join_real = {i: 0.0 for i in range(a.n_inst)}
    for t, i, e in ev:
        tr = t / a.scale
        if e == "notice":
            pending_notice[i] = tr
            continue
        if e == "preempt":
            if i in pending_notice:
                tn = pending_notice.pop(i)
                if a.notice_real_s is not None:
                    tn = tr - a.notice_real_s
                tn = max(tn, last_join_real[i], 0.0)
                if tn < tr:
                    sched.append((tn, i, "notice", tr))
            sched.append((tr, i, "preempt", tr))
        elif e == "join":
            sched.append((tr, i, "join", tr))
            last_join_real[i] = tr
    sched.sort(key=lambda x: (x[0], {"notice": 0, "preempt": 1, "join": 2}[x[2]], x[1]))
    log = open(a.log, "a") if a.log else None

    def emit(msg):
        line = f"{(now_ns() - st.t0) / 1e9:10.3f} {msg}"
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    emit(f"start n_inst={a.n_inst} ondemand={sorted(ondemand)} scale={a.scale} events={len(sched)} "
         f"notice_real_s={a.notice_real_s} trace={a.trace}")
    stop = {"flag": False}

    def on_term(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    with open(os.path.join(a.dir, "sim.pid"), "w") as f:
        f.write(f"{os.getpid()}\n")
    k = 0
    while not stop["flag"]:
        el = (now_ns() - st.t0) / 1e9
        while k < len(sched) and sched[k][0] <= el:
            tr, i, e, tp = sched[k]
            k += 1
            r = st.inst[i]
            if e == "notice" and r["state"] == UP:
                st.update(i, state=NOTICE, deadline=st.t0 + int(tp * 1e9), n_notice=r["n_notice"] + 1)
            elif e == "preempt" and r["state"] != DOWN:
                st.update(i, state=DOWN, deadline=st.t0 + int(tr * 1e9), n_preempt=r["n_preempt"] + 1)
            elif e == "join" and r["state"] == DOWN:
                st.update(i, state=UP, deadline=0, epoch=r["epoch"] + 1, n_join=r["n_join"] + 1)
            else:
                emit(f"skip inst={i} {e} (state {NAMES[r['state']]})")
                continue
            pids = signal_pids(a.dir)
            extra = f" deadline_in={(tp - tr):.3f}s" if e == "notice" else ""
            emit(f"inst={i} {e} -> {NAMES[st.inst[i]['state']]} epoch={st.inst[i]['epoch']}{extra} signalled={pids}")
        if a.until_file and os.path.exists(a.until_file):
            break
        if k >= len(sched) and a.exit_when_done:
            break
        nxt = sched[k][0] if k < len(sched) else el + 0.2
        time.sleep(max(0.0, min(0.05, nxt - (now_ns() - st.t0) / 1e9)))
    tot = {key: sum(r[key] for r in st.inst) for key in ("n_notice", "n_preempt", "n_join")}
    emit(f"stop after {(now_ns() - st.t0) / 1e9:.1f}s: notices={tot['n_notice']} preempts={tot['n_preempt']} "
         f"joins={tot['n_join']} replayed={k}/{len(sched)}")


def cmd_show(a):
    with open(os.path.join(a.dir, "state.bin"), "rb") as f:
        b = f.read()
    magic, ver, n, _, seq, t0, scale = HDR.unpack_from(b, 0)
    print(f"magic={magic:#x} version={ver} n_inst={n} seq={seq} scale={scale} age={(now_ns() - t0) / 1e9:.1f}s")
    for i in range(n):
        s, od, ep, dl, ch, nn, np_, nj, _ = INST.unpack_from(b, HDR.size + INST.size * i)
        print(f"  inst {i}: {NAMES.get(s, s)} ondemand={od} epoch={ep} deadline_in={(dl - now_ns()) / 1e9 if dl else 0:.2f}s "
              f"notices={nn} preempts={np_} joins={nj}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--instances", required=True, help="comma list of spot instance ids (device ordinals)")
    g.add_argument("--duration-s", type=float, default=7200)
    g.add_argument("--start-s", type=float, default=0.0, help="no events before this trace time")
    g.add_argument("--mtti-s", type=float, default=1800, help="mean time to interruption per instance")
    g.add_argument("--down-s", type=float, default=300, help="mean time until a replacement joins")
    g.add_argument("--min-down-s", type=float, default=0.0)
    g.add_argument("--notice-s", type=float, default=120, help="AWS 120, GCP / Azure 30")
    g.add_argument("--hard-frac", type=float, default=0.2, help="fraction of preemptions without notice")
    g.add_argument("--correlated", type=float, default=0.3, help="P(an event reclaims several instances)")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("-o", "--output", default="-")
    g.set_defaults(fn=cmd_gen)
    r = sp.add_parser("run")
    r.add_argument("--trace", default="", help="trace CSV ('' = no events: all instances stay up)")
    r.add_argument("--dir", required=True)
    r.add_argument("--n-inst", type=int, required=True)
    r.add_argument("--ondemand", default="", help="instances never preempted (trace lines ignored)")
    r.add_argument("--scale", type=float, default=60.0, help="trace seconds per real second")
    r.add_argument("--offset-s", type=float, default=0.0, help="start replay at this trace time")
    r.add_argument("--notice-real-s", type=float, default=None, help="override notice length (real seconds)")
    r.add_argument("--log", default="")
    r.add_argument("--until-file", default="", help="exit when this file exists")
    r.add_argument("--exit-when-done", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sp.add_parser("show")
    s.add_argument("--dir", required=True)
    s.set_defaults(fn=cmd_show)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
