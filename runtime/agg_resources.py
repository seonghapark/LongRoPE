#!/usr/bin/env python3
"""Aggregate the resource monitor's per-host output across ALL nodes.

WHY THIS EXISTS
    The in-job report prints a single host's *.summary.json (`ls ... | head -1`).
    A 4-node run writes four of them, so three nodes' worth of data never
    reaches the report. This reads every host file and aggregates properly.

WHY IT IS A SEPARATE SCRIPT
    PBS spools the job script at SUBMIT time, so editing job_agpt2b_post16384.sh
    after submission cannot change what the running job does. Aggregating from
    the on-disk output afterwards gets the same answer without re-queueing.

Usage:  python3 -I agg_resources.py <resource_metrics_dir>
"""
import glob
import json
import os
import sys


def gib(n):
    return n / (1024.0 ** 3)


def main(d):
    summaries = sorted(glob.glob(os.path.join(d, "*.summary.json")))
    csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not summaries:
        print("no *.summary.json under %s" % d)
        return 1

    print("=" * 78)
    print("RESOURCE CONSUMPTION  (sampled every 15 s, all %d hosts)" % len(summaries))
    print("=" * 78)

    # ---- per-host CPU / host-memory -------------------------------------
    hdr = "%-16s %7s %9s %9s %12s %12s" % (
        "host", "samples", "cpu_avg%", "cpu_max%", "mem_avg_GiB", "mem_max_GiB")
    print(hdr)
    print("-" * len(hdr))
    tot_s = 0
    cpu_avgs, cpu_maxs, mem_avgs, mem_maxs, elapsed = [], [], [], [], []
    for p in summaries:
        with open(p) as fh:
            s = json.load(fh)
        a, m = s.get("averages", {}), s.get("maxima", {})
        n = s.get("samples", 0)
        tot_s += n
        cpu_avgs.append(a.get("cpu_percent", 0.0))
        cpu_maxs.append(m.get("cpu_percent", 0.0))
        mem_avgs.append(a.get("memory_used_bytes", 0.0))
        mem_maxs.append(m.get("memory_used_bytes", 0.0))
        elapsed.append(s.get("elapsed_seconds", 0.0))
        print("%-16s %7d %9.3f %9.3f %12.2f %12.2f" % (
            s.get("hostname", "?"), n,
            a.get("cpu_percent", 0.0), m.get("cpu_percent", 0.0),
            gib(a.get("memory_used_bytes", 0.0)), gib(m.get("memory_used_bytes", 0.0))))
    print("-" * len(hdr))
    nh = len(summaries)
    print("%-16s %7d %9.3f %9.3f %12.2f %12.2f" % (
        "ALL (mean/max)", tot_s,
        sum(cpu_avgs) / nh, max(cpu_maxs),
        gib(sum(mem_avgs) / nh), gib(max(mem_maxs))))
    if elapsed:
        print("\nmonitored window : %.1f s (max across hosts)" % max(elapsed))
        print("total samples    : %d across %d hosts" % (tot_s, nh))

    # ---- per-XPU accelerator telemetry ----------------------------------
    # Columns are only populated when xpu-smi was findable on the compute node.
    util, amem, pwr, temp, errs = [], [], [], [], set()
    devices = set()
    for p in csvs:
        with open(p) as fh:
            head = fh.readline().rstrip("\n").split(",")
            try:
                i_dev = head.index("device_id")
                i_u = head.index("accelerator_utilization_percent")
                i_m = head.index("accelerator_memory_used_mib")
                i_p = head.index("accelerator_power_watts")
                i_t = head.index("accelerator_temperature_celsius")
                i_e = head.index("accelerator_error")
                i_h = head.index("hostname")
            except ValueError:
                continue
            for line in fh:
                f = line.rstrip("\n").split(",")
                if len(f) <= i_e:
                    continue
                if f[i_e].strip():
                    errs.add(f[i_e].strip())
                if not f[i_dev].strip():
                    continue
                devices.add((f[i_h], f[i_dev]))
                for idx, acc in ((i_u, util), (i_m, amem), (i_p, pwr), (i_t, temp)):
                    v = f[idx].strip()
                    if v:
                        try:
                            acc.append(float(v))
                        except ValueError:
                            pass

    print()
    print("-" * 78)
    print("ACCELERATOR (XPU) TELEMETRY")
    print("-" * 78)
    if devices:
        print("distinct devices sampled : %d  (%d hosts x %d tiles)" % (
            len(devices), nh, len(devices) // max(nh, 1)))

        def stat(name, vals, unit, scale=1.0):
            if not vals:
                print("  %-26s (not reported)" % name)
                return
            print("  %-26s avg %8.2f %s   max %8.2f %s" % (
                name, sum(vals) / len(vals) * scale, unit, max(vals) * scale, unit))

        stat("utilization", util, "%")
        stat("memory used", amem, "GiB", 1.0 / 1024.0)
        stat("power", pwr, "W")
        stat("temperature", temp, "C")
    else:
        print("  no per-device rows -- xpu-smi did not report on the compute nodes")
    if errs:
        print("  accelerator_error(s): %s" % "; ".join(sorted(errs)[:3]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
