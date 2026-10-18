#!/usr/bin/env python3
"""Summarise and plot the fairness experiment (bench_mutex --mode=fairness).

    python3 analyze_fairness.py fair_<platform>.csv [more.csv ...] [--out plots] [--no-plots]

Figures: Jain's index, max/min ratio and p99 wait versus threads, and a per-thread acquisition-count plot
(one dot per thread and repetition, relative to the mean of the run).
Per (platform, N, OUTSIDE_WORK, threads, lock): medians over the repetitions of
  acquisitions per thread (min / median / max), max/min ratio, Jain's index,
  acquire() latency p50 / p99 / max (sampled), plus the IQR of Jain's index.
Writes <out>/fairness_summary.csv and figures. The numbers describe the behaviour observed in
this environment; they are not a guarantee (FIFO order is not equal throughput, and a finite
experiment cannot prove that starvation is impossible).
Requires pandas; matplotlib for figures.
"""
import argparse
import os

import pandas as pd

import config_check

KEYS = ["platform", "series", "N", "outside_work", "threads", "lock"]
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


def style(lock):
    f = family(lock)
    i = FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 0
    ls = "--" if lock.endswith(("-pause", "-prop")) else (":" if lock.endswith("-sleep") else "-")
    return dict(color=COLORS[i], marker=MARKERS[i], linestyle=ls, linewidth=2, markersize=6)


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def summarise(df):
    ok = df[df["status"] == "ok"]
    g = ok.groupby(KEYS)
    s = g.agg(reps_ok=("jain", "size"), min_acq=("min_acq", "median"), median_acq=("median_acq", "median"),
              max_acq=("max_acq", "median"), max_min_ratio=("max_min_ratio", "median"),
              jain=("jain", "median"), jain_q25=("jain", q25), jain_q75=("jain", q75),
              wait_p50_ns=("wait_p50_ns", "median"), wait_p99_ns=("wait_p99_ns", "median"),
              wait_max_ns=("wait_max_ns", "median"), cs_ns=("cs_ns", "median")).reset_index()
    s["jain_iqr"] = s["jain_q75"] - s["jain_q25"]
    return s


def plots(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    written = []
    metrics = [("jain", "Jain's fairness index (1 = equal)", False, ("jain_q25", "jain_q75")),
               ("max_min_ratio", "max / min acquisitions per thread", True, None),
               ("wait_p99_ns", "acquire() latency p99 (ns)", True, None)]
    for (plat, N), g in s.groupby(["platform", "N"]):
        g = g[g["series"] == "core"] if (g["series"] == "core").any() else g
        outs = sorted(g["outside_work"].unique())
        for col, label, logy, band in metrics:
            fig, axes = plt.subplots(1, len(outs), figsize=(4.2 * len(outs) + 1.8, 3.2), squeeze=False, sharex=True)
            for ax, o in zip(axes.flat, outs):
                for lock, ld in g[g["outside_work"] == o].groupby("lock"):
                    ld = ld.sort_values("threads")
                    ax.plot(ld["threads"], ld[col], **style(lock))
                    if band:
                        ax.fill_between(ld["threads"], ld[band[0]], ld[band[1]], color=style(lock)["color"], alpha=0.15, linewidth=0)
                ths = sorted(g["threads"].unique())
                ax.set_xscale("log", base=2)
                ax.set_xticks(ths)
                ax.set_xticklabels([str(t) for t in ths])
                ax.minorticks_off()
                if logy:
                    ax.set_yscale("log")
                ax.set_title(f"OUTSIDE_WORK = {o}", fontsize=9, loc="left")
                ax.set_xlabel("threads")
                ax.set_ylabel(label)
            handles = [Line2D([0], [0], label=l, **style(l)) for l in sorted(g["lock"].unique())]
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
            fig.suptitle(f"{plat} | N={N} | fairness: {label}", x=0.01, ha="left", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            path = os.path.join(out, f"{plat}_N{N}_fair_{col}.png".replace(" ", "_"))
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


def per_thread_plot(df, out):
    """Per-thread acquisition counts (relative to the mean of the run): one dot per thread and repetition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok = df[df["status"] == "ok"]
    written = []
    for (plat, N), g in ok.groupby(["platform", "N"]):
        g = g[g["series"] == "core"] if (g["series"] == "core").any() else g
        ow = g["outside_work"].min()
        g = g[g["outside_work"] == ow]
        ths = sorted(g["threads"].unique())
        locks = sorted(g["lock"].unique(), key=lambda x: (FAMILY_ORDER.index(family(x)) if family(x) in FAMILY_ORDER else 99, x))
        fig, axes = plt.subplots(1, len(ths), figsize=(max(4.0, 0.7 * len(locks)) * len(ths) + 1.0, 3.6), squeeze=False)
        rng = np.random.default_rng(1)
        for ax, t in zip(axes.flat, ths):
            for i, lock in enumerate(locks):
                rows = g[(g["threads"] == t) & (g["lock"] == lock)]
                ys = []
                for v in rows["per_thread_acq"]:
                    c = np.array([float(x) for x in str(v).split(";")])
                    if c.mean() > 0:
                        ys.extend(c / c.mean())
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(ys)), ys, s=14, color=style(lock)["color"], alpha=0.7, linewidths=0)
            ax.axhline(1.0, color="#898781", linewidth=1)
            ax.set_xticks(range(len(locks)))
            ax.set_xticklabels(locks, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{t} threads", fontsize=9, loc="left")
            ax.set_ylabel("acquisitions / mean of the run")
            ax.grid(alpha=0.3)
        fig.suptitle(f"{plat} | N={N} | OUTSIDE_WORK={ow} | acquisitions per thread (1 = equal share)", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = os.path.join(out, f"{plat}_N{N}_fair_per_thread.png".replace(" ", "_"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--config", default="config/reference.cfg", help="canonical configuration the rows are checked against")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    df = config_check.validate_fairness(df, config_check.load_cfg(a.config))
    bad = df[df["status"] != "ok"]
    print(f"{len(df)} runs from {len(a.csv)} file(s)")
    if len(bad):
        print(f"WARNING: {len(bad)} run(s) not ok (excluded)")
    os.makedirs(a.out, exist_ok=True)
    s = summarise(df)
    path = os.path.join(a.out, "fairness_summary.csv")
    s.round(4).to_csv(path, index=False)
    print(f"summary -> {path} ({len(s)} configurations)")
    cols = ["lock", "threads", "outside_work", "min_acq", "median_acq", "max_acq", "max_min_ratio", "jain",
            "wait_p50_ns", "wait_p99_ns"]
    for (plat, N), g in s.groupby(["platform", "N"]):
        print(f"\n=== {plat}, N={N} ===")
        print(g[cols].sort_values(["outside_work", "threads", "lock"]).round(3).to_string(index=False))
    if not a.no_plots:
        try:
            for p in plots(s, a.out) + per_thread_plot(df, a.out):
                print("figure  ->", p)
        except ImportError:
            print("matplotlib not installed; skipping figures")


if __name__ == "__main__":
    main()