#!/usr/bin/env python3
"""Summarise bench_mpi_mutex CSV output: median throughput per (lock, ranks, out_work).

  python3 analyze_mpi.py results_mpi.csv [--out plots] [--no-plots]
Needs pandas; matplotlib only for plots.
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("csv")
ap.add_argument("--out", default="plots")
ap.add_argument("--no-plots", action="store_true")
a = ap.parse_args()

df = pd.read_csv(a.csv)
bad = df[df["valid"] != 1]
if len(bad):
    print(f"WARNING: {len(bad)} invalid rows (counter mismatch) excluded")
df = df[df["valid"] == 1]
g = df.groupby(["out_work", "lock", "ranks"])["ops_per_sec"].agg(["median", "min", "max"]).reset_index()
for ow, sub in g.groupby("out_work"):
    print(f"\n=== out_work={ow}: median ops/s (all ranks combined) ===")
    print(sub.pivot(index="lock", columns="ranks", values="median").round(0).to_string())
if not a.no_plots:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots"); raise SystemExit
    os.makedirs(a.out, exist_ok=True)
    for ow, sub in g.groupby("out_work"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lock, ld in sub.groupby("lock"):
            ld = ld.sort_values("ranks")
            ax.plot(ld["ranks"], ld["median"], marker="o", label=lock)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("MPI ranks"); ax.set_ylabel("lock rounds / s (total)")
        ax.set_title(f"MPI RMA mutex locks, out_work={ow}")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"mpi_throughput_ow{ow}.png"), dpi=140); plt.close(fig)
    print(f"plots in {a.out}/")
