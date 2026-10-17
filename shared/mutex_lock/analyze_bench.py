#!/usr/bin/env python3
"""Summarise and plot bench_mutex CSV results (one or several platforms).

    python3 analyze_bench.py bench_results.csv [more.csv ...] [--out plots] [--no-plots]

Writes   <out>/bench_summary.csv   median / min / sd per configuration, speed-up,
                                   time relative to std-mutex, finish-time spread
         <out>/*.png               figures per platform and critical-section size N

Only status == ok rows are used (timeouts / invalid runs are counted and reported).
Requires pandas; matplotlib for the figures.
"""
import argparse
import os
import sys

import pandas as pd

KEYS = ["platform", "series", "mode", "lock", "threads", "outside_work", "N"]

# Lock family -> categorical slot (fixed order, validated palette); variants within a
# family are told apart by line style, so identity never relies on colour alone.
FAMILY_ORDER = ["tas", "ttas", "ticket", "mcs", "std-mutex", "pthread-mutex", "pthread-spin", "omp"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def family(lock):
    if lock.startswith("omp"):
        return "omp"
    for f in ("ttas", "tas", "ticket", "mcs"):
        if lock == f or lock.startswith(f + "-"):
            return f
    return lock


def line_style(lock):
    if lock.endswith("-sleep"):
        return ":"
    if lock.endswith("-yield"):
        return "-."
    if lock.endswith("-pause") or lock.endswith("-prop") or lock == "omp-critical":
        return "--"
    return "-"


def summarise(df):
    ok = df[df["status"] == "ok"].copy()
    ok["spread"] = (ok["finish_max_ms"] - ok["finish_min_ms"]) / ok["finish_max_ms"].where(ok["finish_max_ms"] > 0)
    g = ok.groupby(KEYS)
    s = g.agg(reps_ok=("time_ms", "size"),
              median_ms=("time_ms", "median"),
              min_ms=("time_ms", "min"),
              sd_ms=("time_ms", "std"),
              median_mops=("throughput_mops", "median"),
              finish_spread=("spread", "median"),
              cs_ns=("cs_ns", "median")).reset_index()
    # relative to std-mutex in the same configuration
    ref = s[s["lock"] == "std-mutex"][[k for k in KEYS if k != "lock"] + ["median_ms"]]
    ref = ref.rename(columns={"median_ms": "ref_ms"})
    s = s.merge(ref, on=[k for k in KEYS if k != "lock"], how="left")
    s["rel_std_mutex"] = s["median_ms"] / s["ref_ms"]
    # strong scaling speed-up: the same lock on 1 thread
    one = s[s["threads"] == 1][["platform", "mode", "lock", "outside_work", "N", "median_ms"]]
    one = one.rename(columns={"median_ms": "t1_ms"})
    s = s.merge(one, on=["platform", "mode", "lock", "outside_work", "N"], how="left")
    s["speedup"] = s["t1_ms"] / s["median_ms"]
    return s.drop(columns=["ref_ms", "t1_ms"])


def plots(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "text.color": "#0b0b0b", "axes.grid": True,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9,
    })

    def style(lock):
        f = family(lock)
        i = FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 0
        return dict(color=COLORS[i], marker=MARKERS[i], linestyle=line_style(lock), linewidth=2, markersize=6)

    def legend(fig, locks):
        handles = [Line2D([0], [0], label=l, **style(l)) for l in sorted(locks, key=lambda x: (
            FAMILY_ORDER.index(family(x)) if family(x) in FAMILY_ORDER else 99, x))]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)

    written = []

    def panel_figure(d, ycol, ylabel, fname, title, ideal=None, hline=None, logy=False):
        outs = sorted(d["outside_work"].unique())
        if d.empty or not outs:
            return
        cols = min(3, len(outs))
        rows = -(-len(outs) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols + 1.8, 3.0 * rows), squeeze=False, sharex=True)
        for ax, o in zip(axes.flat, outs):
            sub = d[d["outside_work"] == o]
            for lock, ld in sub.groupby("lock"):
                ld = ld.sort_values("threads")
                ax.plot(ld["threads"], ld[ycol], **style(lock))
            if ideal == "linear":
                th = sorted(sub["threads"].unique())
                ax.plot(th, th, color="#898781", linewidth=1, linestyle=(0, (1, 2)))
            if hline is not None:
                ax.axhline(hline, color="#898781", linewidth=1)
            ax.set_xscale("log", base=2)
            ths = sorted(sub["threads"].unique())
            ax.set_xticks(ths)
            ax.set_xticklabels([str(t) for t in ths])
            ax.minorticks_off()
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"OUTSIDE_WORK = {o}", fontsize=9, color="#52514e", loc="left")
            ax.set_xlabel("threads")
            ax.set_ylabel(ylabel)
        for ax in list(axes.flat)[len(outs):]:
            ax.axis("off")
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
        legend(fig, d["lock"].unique())
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = os.path.join(out, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    for (plat, N), g in s.groupby(["platform", "N"]):
        cs = g["cs_ns"].median()
        tag = f"{plat}_N{N}".replace(" ", "_")
        core = g[g["series"] == "core"]
        head = f"{plat}  |  N={N} (critical section about {cs:,.0f} ns)"
        fx = core[core["mode"] == "fixed"]
        st = core[core["mode"] == "strong"]
        panel_figure(fx, "median_mops", "critical sections / us", f"{tag}_fixed_throughput.png",
                     head + "  |  fixed work per thread: throughput (higher is better)")
        panel_figure(fx, "rel_std_mutex", "time / std-mutex", f"{tag}_fixed_rel.png",
                     head + "  |  fixed work per thread: whole-program time relative to std-mutex (lower is better)",
                     hline=1.0, logy=True)
        panel_figure(st, "speedup", "speed-up vs 1 thread", f"{tag}_strong_speedup.png",
                     head + "  |  strong scaling: speed-up of the same lock (dotted = ideal)", ideal="linear")
        panel_figure(st, "rel_std_mutex", "time / std-mutex", f"{tag}_strong_rel.png",
                     head + "  |  strong scaling: whole-program time relative to std-mutex (lower is better)",
                     hline=1.0, logy=True)

        # secondary series: oversubscription (2x usable CPUs), x = OUTSIDE_WORK
        ov = g[(g["series"] == "oversub")]
        for mode, md in ov.groupby("mode"):
            fig, ax = plt.subplots(figsize=(6.6, 3.4))
            for lock, ld in md.groupby("lock"):
                ld = ld.sort_values("outside_work")
                ax.plot(ld["outside_work"], ld["rel_std_mutex"], **style(lock))
            ax.axhline(1.0, color="#898781", linewidth=1)
            ax.set_yscale("log")
            ow = sorted(md["outside_work"].unique())
            ax.set_xticks(ow)
            ax.set_xlabel("OUTSIDE_WORK")
            ax.set_ylabel("time / std-mutex")
            ax.set_title(f"{head}  |  {mode}, oversubscribed ({int(md['threads'].max())} threads)",
                         fontsize=9, loc="left")
            legend(fig, md["lock"].unique())
            path = os.path.join(out, f"{tag}_oversub_{mode}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    bad = df[df["status"] != "ok"]
    print(f"{len(df)} runs from {len(a.csv)} file(s); platforms: {', '.join(sorted(df['platform'].unique()))}")
    if len(bad):
        print(f"WARNING: {len(bad)} run(s) not ok (excluded): "
              + ", ".join(f"{k}={v}" for k, v in bad["status"].value_counts().items()))
        print(bad.groupby(["platform", "lock", "status"]).size().to_string())

    os.makedirs(a.out, exist_ok=True)
    s = summarise(df)
    path = os.path.join(a.out, "bench_summary.csv")
    s.round(4).to_csv(path, index=False)
    print(f"summary -> {path} ({len(s)} configurations)")

    if not a.no_plots:
        try:
            for p in plots(s, a.out):
                print("figure  ->", p)
        except ImportError:
            print("matplotlib not installed: skipped figures (pip install matplotlib)")


if __name__ == "__main__":
    sys.exit(main())