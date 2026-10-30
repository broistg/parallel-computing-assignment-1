#!/usr/bin/env python3
"""Ex3 helpers: capture | aggregate | plot | merge."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_DEVICES = 3
SKIP_JSON = {"machine.json", "machines.json"}
SKIP_DIRS = {"report", "plots"}
FIELDS = [
    "hostname",
    "cpu_model",
    "os",
    "logical_cpus",
    "memory_bytes",
    "threads",
    "increments_per_thread",
    "impl",
    "seconds",
    "total_increments",
    "source",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def _memory_bytes() -> int | None:
    system = platform.system()
    if system == "Darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        return int(raw) if raw.isdigit() else None
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None


def _os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        ver = _run(["sw_vers", "-productVersion"])
        return f"macOS {ver}" if ver else "macOS"
    if system == "Linux":
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return f"Linux {platform.release()}"
    return f"{system} {platform.release()}"


def cmd_capture(out: Path) -> int:
    mem = _memory_bytes()
    info = {
        "device_folder": os.environ.get("EX3_DEVICE_FOLDER") or out.parent.name,
        "hostname": platform.node() or "unknown",
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count() or 0,
        "os": _os_name(),
        "memory_bytes": mem,
        "memory_gib": round(mem / (1024**3), 2) if mem else None,
        "arch": platform.machine(),
        "platform": platform.platform(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(out)
    print(f"EX3_HOSTNAME={info['hostname']}")
    print(f"EX3_CPU_MODEL={info['cpu_model']}")
    print(f"EX3_LOGICAL_CPUS={info['logical_cpus']}")
    print(f"EX3_OS={info['os']}")
    if mem is not None:
        print(f"EX3_MEMORY_BYTES={mem}")
    return 0


def _resolve_dir(arg: str | None) -> Path:
    results = Path(arg) if arg else ROOT / "results"
    if not results.is_absolute():
        results = (Path.cwd() / results).resolve()
    return results


def cmd_aggregate(results_arg: str | None) -> int:
    results = _resolve_dir(results_arg)
    files = sorted(
        p
        for p in results.glob("**/*.json")
        if p.name not in SKIP_JSON and "plots" not in p.parts
    )
    if not files:
        print(f"no JSON files in {results}", file=sys.stderr)
        return 1
    rows = []
    for path in files:
        data = json.loads(path.read_text())
        row = {key: data.get(key, "") for key in FIELDS if key != "source"}
        try:
            row["source"] = str(path.relative_to(results))
        except ValueError:
            row["source"] = str(path)
        rows.append(row)
    rows.sort(
        key=lambda r: (
            str(r["hostname"]),
            str(r["impl"]),
            int(r["threads"] or 0),
            int(float(r["increments_per_thread"] or 0)),
        )
    )
    out = results / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def cmd_plot(results_arg: str | None) -> int:
    results = _resolve_dir(results_arg)
    summary = results / "summary.csv"
    if not summary.is_file():
        print(f"missing {summary}", file=sys.stderr)
        return 1
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        return 1

    with summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("summary.csv is empty", file=sys.stderr)
        return 1

    by_iters: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(
        list
    )
    by_threads: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
    host_meta: dict[str, str] = {}
    host_short: dict[str, str] = {}

    for row in rows:
        if not row.get("impl") or not row.get("threads"):
            continue
        host = row.get("hostname") or "unknown"
        impl = row["impl"]
        threads = int(row["threads"])
        iters = int(float(row.get("increments_per_thread") or 0))
        seconds = float(row["seconds"])
        total = float(row.get("total_increments") or 0)
        by_iters[(host, impl, iters)].append((threads, seconds, total))
        by_threads[(host, impl, threads)].append((iters, seconds))
        if host not in host_meta:
            folder = row.get("device_folder") or ""
            short = folder or host.split(".")[0]
            host_short[host] = short
            bits = [short]
            if row.get("cpu_model"):
                bits.append(row["cpu_model"])
            if row.get("os"):
                bits.append(row["os"])
            if row.get("logical_cpus"):
                bits.append(f"{row['logical_cpus']} CPUs")
            host_meta[host] = " | ".join(bits)

    plots_dir = results / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    hosts = sorted(host_short)
    all_iters = sorted({k[2] for k in by_iters})
    ref_iters = all_iters[len(all_iters) // 2] if all_iters else 0
    all_thread_counts = sorted({k[2] for k in by_threads})
    ref_threads = (
        min(all_thread_counts, key=lambda t: abs(t - 8)) if all_thread_counts else 8
    )

    for host in hosts:
        tag = _safe_name(host_short.get(host, host))
        subtitle = host_meta.get(host, host)
        impls = sorted({impl for h, impl, _ in by_iters if h == host})

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl in impls:
            pts = sorted(by_iters.get((host, impl, ref_iters), []), key=lambda p: p[0])
            if not pts:
                for it in reversed(all_iters):
                    pts = sorted(by_iters.get((host, impl, it), []), key=lambda p: p[0])
                    if pts:
                        break
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl)
        ax.set_xlabel("threads")
        ax.set_ylabel("execution time (seconds)")
        ax.set_title(
            f"Scalability vs threads — {host_short.get(host, host)} "
            f"(iters/thread={ref_iters})"
        )
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("png", "pdf"):
            out = plots_dir / f"time_vs_threads_{tag}.{ext}"
            fig.savefig(out, dpi=150)
            print(out)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl in impls:
            pts = sorted(
                by_threads.get((host, impl, ref_threads), []), key=lambda p: p[0]
            )
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl)
        ax.set_xlabel("increments per thread")
        ax.set_ylabel("execution time (seconds)")
        ax.set_title(
            f"Time vs workload — {host_short.get(host, host)} (threads={ref_threads})"
        )
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("png", "pdf"):
            out = plots_dir / f"time_vs_iters_{tag}.{ext}"
            fig.savefig(out, dpi=150)
            print(out)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl in impls:
            pts = sorted(by_iters.get((host, impl, ref_iters), []), key=lambda p: p[0])
            xs, ys = [], []
            for threads, seconds, total in pts:
                if seconds > 0:
                    xs.append(threads)
                    ys.append(total / seconds)
            if xs:
                ax.plot(xs, ys, marker="o", label=impl)
        ax.set_xlabel("threads")
        ax.set_ylabel("throughput (increments / second)")
        ax.set_title(
            f"Throughput — {host_short.get(host, host)} (iters/thread={ref_iters})"
        )
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("png", "pdf"):
            out = plots_dir / f"throughput_vs_threads_{tag}.{ext}"
            fig.savefig(out, dpi=150)
            print(out)
        plt.close(fig)

    if len(hosts) >= 2:
        impls_all = sorted({impl for _, impl, _ in by_iters})
        for impl in impls_all:
            fig, ax = plt.subplots(figsize=(8, 5))
            for host in hosts:
                pts = sorted(
                    by_iters.get((host, impl, ref_iters), []), key=lambda p: p[0]
                )
                if not pts:
                    continue
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    label=host_short.get(host, host),
                )
            ax.set_xlabel("threads")
            ax.set_ylabel("execution time (seconds)")
            ax.set_title(f"{impl}: across devices (iters/thread={ref_iters})")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(title="device")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                out = plots_dir / f"compare_devices_{impl}.{ext}"
                fig.savefig(out, dpi=150)
                print(out)
            plt.close(fig)

        n = len(hosts)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, host in zip(axes, hosts):
            impls = sorted({impl for h, impl, _ in by_iters if h == host})
            for impl in impls:
                pts = sorted(
                    by_iters.get((host, impl, ref_iters), []), key=lambda p: p[0]
                )
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="o",
                        label=impl,
                    )
            ax.set_title(host_short.get(host, host))
            ax.set_xlabel("threads")
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("execution time (seconds)")
        fig.suptitle(f"Scalability on all platforms (iters/thread={ref_iters})")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = plots_dir / f"compare_all_devices.{ext}"
            fig.savefig(out, dpi=150)
            print(out)
        plt.close(fig)

    print(f"plots in {plots_dir} ({len(hosts)} device(s))")
    return 0


def cmd_merge(args: list[str]) -> int:
    if args:
        device_dirs = []
        for arg in args:
            p = Path(arg)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if not p.is_dir():
                print(f"not a directory: {p}", file=sys.stderr)
                return 1
            device_dirs.append(p)
    else:
        results = ROOT / "results"
        device_dirs = sorted(
            p
            for p in results.iterdir()
            if p.is_dir()
            and p.name not in SKIP_DIRS
            and (p / "machine.json").is_file()
        )

    if len(device_dirs) < MIN_DEVICES:
        print(
            f"need ≥{MIN_DEVICES} device folders with machine.json, "
            f"found {len(device_dirs)}: {[p.name for p in device_dirs]}",
            file=sys.stderr,
        )
        print("On each machine: ./run.sh <name>  then copy results/ here and merge.", file=sys.stderr)
        return 1

    machines = []
    for d in device_dirs:
        data = json.loads((d / "machine.json").read_text())
        data["device_folder"] = d.name
        machines.append(data)
    hosts = [m.get("hostname") or m["device_folder"] for m in machines]
    if len(set(hosts)) < MIN_DEVICES:
        print(f"need {MIN_DEVICES} distinct hostnames, got: {hosts}", file=sys.stderr)
        return 1

    out = ROOT / "results" / "report"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "devices").mkdir()
    for d in device_dirs:
        shutil.copytree(d, out / "devices" / d.name, ignore=shutil.ignore_patterns("plots"))

    (out / "machines.json").write_text(
        json.dumps({"device_count": len(machines), "devices": machines}, indent=2) + "\n"
    )
    lines = [
        "# Experiment platforms",
        "",
        f"Collected **{len(machines)}** shared-memory platforms.",
        "",
        "| # | Folder | Hostname | CPU | OS | Logical CPUs | RAM (GiB) |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for i, m in enumerate(machines, 1):
        ram = m.get("memory_gib")
        if ram is None and m.get("memory_bytes"):
            ram = round(m["memory_bytes"] / (1024**3), 2)
        lines.append(
            f"| {i} | `{m['device_folder']}` | {m.get('hostname','')} | "
            f"{m.get('cpu_model','')} | {m.get('os','')} | "
            f"{m.get('logical_cpus','')} | {ram if ram is not None else ''} |"
        )
    lines.append("")
    (out / "machines.md").write_text("\n".join(lines))

    if cmd_aggregate(str(out)) != 0:
        return 1

    summary = out / "summary.csv"
    rows = list(csv.DictReader(summary.open()))
    fieldnames = list(rows[0].keys()) if rows else []
    if "device_folder" not in fieldnames:
        fieldnames.insert(0, "device_folder")
    for row in rows:
        parts = Path(row.get("source", "")).parts
        row["device_folder"] = (
            parts[1] if len(parts) >= 2 and parts[0] == "devices" else ""
        )
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if cmd_plot(str(out)) != 0:
        print("warning: plots failed", file=sys.stderr)

    print(f"merged {len(device_dirs)} devices -> {out}")
    print(f"platforms: {', '.join(hosts)}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: report.py capture <machine.json>")
        print("       report.py aggregate [results_dir]")
        print("       report.py plot [results_dir]")
        print("       report.py merge [device_dir ...]")
        return 0
    cmd = sys.argv[1]
    if cmd == "capture":
        return cmd_capture(Path(sys.argv[2]) if len(sys.argv) > 2 else Path("machine.json"))
    if cmd == "aggregate":
        return cmd_aggregate(sys.argv[2] if len(sys.argv) > 2 else None)
    if cmd == "plot":
        return cmd_plot(sys.argv[2] if len(sys.argv) > 2 else None)
    if cmd == "merge":
        return cmd_merge(sys.argv[2:])
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
