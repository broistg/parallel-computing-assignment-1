#!/usr/bin/env python3
"""Summarise bench_mpi_mutex CSV output (timing or fairness, detected from the columns).

  python3 analyze_mpi.py results_mpi.csv [more.csv ...] [--out plots] [--no-plots]

Timing CSV (series scale, contention, cs, strong, oversub; see README):
  writes <out>/mpi_timing_summary.csv (median, IQR, speed-up, time relative to winlock) and
  <out>/mpi_oversub_table.csv (series 5: throughput and whole-run time at the logical-CPU rank
  count and at twice that count), plus one figure set per series.
Fairness CSV: median min/median/max acquisitions per rank, max/min ratio, Jain's index (with IQR),
  acquire() latency p50/p99, figures including a per-rank acquisition-count plot.
Rows with valid != 1 are excluded. Lines are medians, shaded bands the interquartile range.
Fairness numbers describe the observed behaviour, they are not a guarantee.
Needs pandas; matplotlib only for plots.
"""
import argparse
import os

import pandas as pd


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


FAMILY = ["tas", "ttas", "ticket", "mcs", "winlock"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "X"]


def family(lock):
    return next((x for x in ("ttas", "tas", "ticket", "mcs", "winlock") if lock == x or lock.startswith(x + "-")), lock)


def style(lock):
    f = family(lock)
    i = FAMILY.index(f) if f in FAMILY else 0
    ls = "--" if lock.endswith(("-pause", "-prop")) else (":" if lock.endswith("-sleep") else "-")
    return dict(color=COLORS[i], marker=MARKERS[i], linestyle=ls, linewidth=2, markersize=6)


def lock_order(locks):
    return sorted(locks, key=lambda x: (FAMILY.index(family(x)) if family(x) in FAMILY else 99, x))


def has_series(s, tag):
    return s[s["experiment"].astype(str).apply(lambda e: tag in e.split("+"))]


def panels(d, xcol, panelcol, ycol, ylabel, xlabel, path, title, panel_label, band=None, logy=False, logx=None,
           hline=None, ideal=False, symlog_x=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if d.empty:
        return None
    pv = sorted(d[panelcol].unique())
    cols = min(3, len(pv))
    rows = -(-len(pv) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols + 1.8, 3.0 * rows), squeeze=False, sharex=True)
    for ax, v in zip(axes.flat, pv):
        sub = d[d[panelcol] == v]
        for lock, ld in sub.groupby("lock"):
            ld = ld.sort_values(xcol)
            ax.plot(ld[xcol], ld[ycol], **style(lock))
            if band:
                ax.fill_between(ld[xcol], ld[band[0]], ld[band[1]], color=style(lock)["color"], alpha=0.15, linewidth=0)
        xs = sorted(sub[xcol].unique())
        if ideal:
            ax.plot(xs, xs, color="#898781", linewidth=1, linestyle=(0, (1, 2)))
        if hline is not None:
            ax.axhline(hline, color="#898781", linewidth=1)
        if symlog_x:
            ax.set_xscale("symlog", linthresh=10)
        elif logx:
            ax.set_xscale("log", base=2)
        if not symlog_x:
            ax.set_xticks(xs)
            ax.set_xticklabels([str(int(x)) for x in xs])
        ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{panel_label} = {v:g}", fontsize=9, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(pv):]:
        ax.axis("off")
    handles = [Line2D([0], [0], label=l, **style(l)) for l in lock_order(d["lock"].unique())]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def timing(df, out, plots):
    keys = ["lock", "ranks", "mode", "cs_work", "out_work", "outside_k"]
    s = df.groupby(keys).agg(trials=("ops_per_sec", "size"), median_ops_s=("ops_per_sec", "median"),
                             ops_q25=("ops_per_sec", q25), ops_q75=("ops_per_sec", q75),
                             median_s=("seconds", "median"), s_q25=("seconds", q25), s_q75=("seconds", q75),
                             experiment=("experiment", "first")).reset_index()
    s["iqr_s"] = s["s_q75"] - s["s_q25"]
    # time relative to the library lock (MPI_Win_lock), same configuration
    if (s["lock"] == "winlock").any():
        ref = s[s["lock"] == "winlock"][[k for k in keys if k != "lock"] + ["median_s"]].rename(columns={"median_s": "ref_s"})
        s = s.merge(ref, on=[k for k in keys if k != "lock"], how="left")
        s["rel_winlock"] = s["median_s"] / s["ref_s"]
        s["rel_lo"] = s["s_q25"] / s["ref_s"]
        s["rel_hi"] = s["s_q75"] / s["ref_s"]
    # strong scaling speed-up: the same lock on one rank
    one = s[s["ranks"] == 1][["lock", "mode", "cs_work", "out_work", "median_s"]].rename(columns={"median_s": "t1_s"})
    s = s.merge(one, on=["lock", "mode", "cs_work", "out_work"], how="left")
    s["speedup"] = s["t1_s"] / s["median_s"]
    s["speedup_lo"] = s["t1_s"] / s["s_q75"]
    s["speedup_hi"] = s["t1_s"] / s["s_q25"]
    s.round(6).to_csv(os.path.join(out, "mpi_timing_summary.csv"), index=False)

    # series 5 table
    d = has_series(s, "oversub")
    rows = []
    for (ow, k, lock), g in d.groupby(["out_work", "outside_k", "lock"]):
        g = g.sort_values("ranks")
        if len(g) >= 2:
            a, b = g.iloc[0], g.iloc[-1]
            rows.append(dict(outside_k=k, lock=lock, ranks_lo=int(a["ranks"]), ranks_hi=int(b["ranks"]),
                             ops_s_lo=a["median_ops_s"], ops_s_hi=b["median_ops_s"], ops_ratio=b["median_ops_s"] / a["median_ops_s"],
                             seconds_lo=a["median_s"], seconds_hi=b["median_s"], time_ratio=b["median_s"] / a["median_s"]))
    ot = pd.DataFrame(rows)
    if len(ot):
        ot.round(4).to_csv(os.path.join(out, "mpi_oversub_table.csv"), index=False)
        print("\nSeries 5 (oversubscription), fixed work per rank: throughput = lock rounds / s (all ranks), time = whole run")
        print("time ratio ideal = 2 (twice the work), throughput ratio ideal = 1")
        print(ot.round(3).to_string(index=False))

    for name in ("scale", "contention", "cs", "strong"):
        dd = has_series(s, name)
        if dd.empty:
            continue
        print(f"\n=== series {name}: median ops/s (all ranks combined) ===")
        for (cs, k), sub in dd.groupby(["cs_work", "outside_k"]):
            print(f"-- cs_work={cs} outside_k={k}")
            print(sub.pivot(index="lock", columns="ranks", values="median_ops_s").round(0).to_string())
    if not plots:
        return
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    written = []
    rel = "rel_winlock" in s.columns
    d = has_series(s, "scale")
    d = d[d["mode"] == "fixed"]
    written += [panels(d, "ranks", "outside_k", "median_ops_s", "lock rounds / s (all ranks)", "MPI ranks",
                       os.path.join(out, "mpi_s1_scale_throughput.png"), "series 1 scale: throughput (band = IQR)",
                       "OUTSIDE_WORK k", band=("ops_q25", "ops_q75"), logx=True, logy=True)]
    if rel:
        written += [panels(d, "ranks", "outside_k", "rel_winlock", "time / winlock", "MPI ranks",
                           os.path.join(out, "mpi_s1_scale_rel.png"), "series 2 scale: whole-run time relative to winlock (lower is better)",
                           "OUTSIDE_WORK k", band=("rel_lo", "rel_hi"), logx=True, logy=True, hline=1.0)]
    d = has_series(s, "contention")
    written += [panels(d, "outside_k", "ranks", "median_ops_s", "lock rounds / s (all ranks)", "OUTSIDE_WORK k",
                       os.path.join(out, "mpi_s2_contention_throughput.png"), "series 2 contention: throughput (band = IQR)",
                       "ranks", band=("ops_q25", "ops_q75"), logy=True)]
    d = has_series(s, "cs")
    for k, dd in d.groupby("outside_k"):
        written += [panels(dd, "cs_work", "ranks", "median_ops_s", "lock rounds / s (all ranks)", "critical-section work",
                           os.path.join(out, f"mpi_s3_cs_k{int(k)}_throughput.png"),
                           f"series 3 critical-section length, OUTSIDE_WORK k={int(k)}: throughput (band = IQR)", "ranks",
                           band=("ops_q25", "ops_q75"), logy=True, symlog_x=True)]
    d = has_series(s, "strong")
    d = d[d["mode"] == "strong"]
    written += [panels(d, "ranks", "outside_k", "speedup", "speed-up vs 1 rank", "MPI ranks",
                       os.path.join(out, "mpi_s4_strong_speedup.png"), "series 4 strong scaling: speed-up (dotted = ideal, band = IQR)",
                       "OUTSIDE_WORK k", band=("speedup_lo", "speedup_hi"), logx=True, ideal=True)]
    if len(ot):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import numpy as np
        d = has_series(s, "oversub")
        ks = sorted(d["outside_k"].unique())
        locks = lock_order(d["lock"].unique())
        for metric, ycol, lo, hi, ylabel, fn in (("throughput", "median_ops_s", "ops_q25", "ops_q75", "lock rounds / s", "throughput"),
                                                 ("whole-run time", "median_s", "s_q25", "s_q75", "seconds", "time")):
            fig, axes = plt.subplots(1, len(ks), figsize=(max(4.2, 0.7 * len(locks)) * len(ks) + 1.5, 3.6), squeeze=False)
            lo_r, hi_r = int(ot["ranks_lo"].iloc[0]), int(ot["ranks_hi"].iloc[0])
            for ax, k in zip(axes.flat, ks):
                x = np.arange(len(locks))
                for j, (r, hatch) in enumerate(((lo_r, None), (hi_r, "///"))):
                    vals, le, he, cols = [], [], [], []
                    for l in locks:
                        rr = d[(d["lock"] == l) & (d["ranks"] == r) & (d["outside_k"] == k)]
                        if rr.empty:
                            vals.append(np.nan); le.append(0); he.append(0)
                        else:
                            v = float(rr[ycol].iloc[0])
                            vals.append(v); le.append(max(0.0, v - float(rr[lo].iloc[0]))); he.append(max(0.0, float(rr[hi].iloc[0]) - v))
                        cols.append(style(l)["color"])
                    ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=cols, hatch=hatch, edgecolor="#fcfcfb",
                           yerr=[le, he], error_kw=dict(ecolor="#52514e", lw=0.8, capsize=2))
                ax.set_xticks(x)
                ax.set_xticklabels(locks, rotation=45, ha="right", fontsize=8)
                ax.set_yscale("log")
                ax.set_title(f"OUTSIDE_WORK k = {k:g}", fontsize=9, loc="left")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
            fig.legend(handles=[Patch(facecolor="#898781", label=f"{lo_r} ranks (logical CPUs)"),
                                Patch(facecolor="#898781", hatch="///", edgecolor="#fcfcfb", label=f"{hi_r} ranks (2x logical)")],
                       loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
            fig.suptitle(f"series 5 oversubscription: {metric} (fixed work per rank, error bars = IQR)", x=0.01, ha="left", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            p = os.path.join(out, f"mpi_s5_oversub_{fn}.png")
            fig.savefig(p, dpi=140, bbox_inches="tight")
            plt.close(fig)
            written.append(p)
    for p in written:
        if p:
            print("figure  ->", p)


def fairness(df, out, plots):
    keys = ["out_work", "lock", "ranks"]
    s = df.groupby(keys).agg(trials=("jain", "size"), min_acq=("min_acq", "median"), median_acq=("median_acq", "median"),
                             max_acq=("max_acq", "median"), max_min_ratio=("max_min_ratio", "median"),
                             jain=("jain", "median"), jain_q25=("jain", q25), jain_q75=("jain", q75),
                             wait_p50_ns=("wait_p50_ns", "median"), wait_p99_ns=("wait_p99_ns", "median"),
                             wait_max_ns=("wait_max_ns", "median")).reset_index()
    s["jain_iqr"] = s["jain_q75"] - s["jain_q25"]
    s.round(4).to_csv(os.path.join(out, "mpi_fairness_summary.csv"), index=False)
    cols = ["lock", "ranks", "out_work", "min_acq", "median_acq", "max_acq", "max_min_ratio", "jain", "jain_iqr",
            "wait_p50_ns", "wait_p99_ns"]
    print(s[cols].sort_values(["out_work", "ranks", "lock"]).round(3).to_string(index=False))
    if not plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    s["outside_k"] = s["out_work"]
    for col, label, title, fn, logy, band in (("jain", "Jain's fairness index (1 = equal)", "fairness (band = IQR)", "mpi_fair_jain.png", False, ("jain_q25", "jain_q75")),
                                               ("wait_p99_ns", "acquire() latency p99 (ns)", "tail latency", "mpi_fair_p99.png", True, None)):
        p = panels(s, "ranks", "out_work", col, label, "MPI ranks", os.path.join(out, fn), "MPI RMA mutex locks: " + title,
                   "out_work", band=band, logx=True, logy=logy)
        if p:
            print("figure  ->", p)
    # per-rank acquisition counts
    ow = df["out_work"].min()
    g = df[df["out_work"] == ow]
    rk = sorted(g["ranks"].unique())
    locks = lock_order(g["lock"].unique())
    fig, axes = plt.subplots(1, len(rk), figsize=(max(4.0, 0.7 * len(locks)) * len(rk) + 1.0, 3.6), squeeze=False)
    rng = np.random.default_rng(1)
    for ax, r in zip(axes.flat, rk):
        for i, lock in enumerate(locks):
            ys = []
            for v in g[(g["ranks"] == r) & (g["lock"] == lock)]["per_rank_acq"]:
                c = np.array([float(x) for x in str(v).split(";")])
                if c.mean() > 0:
                    ys.extend(c / c.mean())
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(ys)), ys, s=14, color=style(lock)["color"], alpha=0.7, linewidths=0)
        ax.axhline(1.0, color="#898781", linewidth=1)
        ax.set_xticks(range(len(locks)))
        ax.set_xticklabels(locks, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{r} ranks", fontsize=9, loc="left")
        ax.set_ylabel("acquisitions / mean of the run")
        ax.grid(alpha=0.3)
    fig.suptitle(f"MPI RMA mutex locks: acquisitions per rank (out_work={ow}, 1 = equal share)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "mpi_fair_per_rank.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("figure  ->", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    if "config_id" not in df.columns:
        raise SystemExit("These CSV files lack the config_id column (written by an older version); rerun the benchmark.")
    ids = sorted(df["config_id"].unique())
    if len(ids) > 1:
        print("WARNING: more than one config_id (different rounds, busy work, trials, warm-up, duration or lock parameters):")
        print(df.groupby("config_id")["lock"].agg(lambda x: f"{len(x)} rows").to_string())
    else:
        print(f"config_id {ids[0]} on all rows")
    bad = df[df["valid"] != 1]
    if len(bad):
        print(f"WARNING: {len(bad)} invalid rows (counter mismatch) excluded")
    df = df[df["valid"] == 1]
    os.makedirs(a.out, exist_ok=True)
    if "jain" in df.columns:
        fairness(df, a.out, not a.no_plots)
    else:
        timing(df, a.out, not a.no_plots)


if __name__ == "__main__":
    main()