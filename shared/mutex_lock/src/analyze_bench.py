#!/usr/bin/env python3
"""Summarise and plot bench_mutex CSV results (one or several platforms).

    python3 analyze_bench.py bench_results.csv [more.csv ...] [--out plots] [--no-plots]

Writes   <out>/bench_summary.csv   median / IQR / min / sd per configuration (whole-program time and the
                                   start-gate-to-join time region_*), speed-up, time relative to std-mutex,
                                   finish-time spread
         <out>/oversub_table.csv   series 5: throughput and whole-program time of every lock at the logical
                                   CPU count and at twice that count
Figures per experiment series (the series column of the benchmark: scale, contention, cs, strong, oversub):
  1 scale       throughput / relative time versus threads, one panel per OUTSIDE_WORK
  2 contention  throughput / relative time versus OUTSIDE_WORK, one panel per thread count
  3 cs          relative time / throughput versus critical-section length, one figure per OUTSIDE_WORK
  4 strong      speed-up and relative time versus threads
  5 oversub     throughput and whole-program time at the logical-CPU count and at twice that count (bars)
Lines in the figures are medians; the shaded band is the interquartile range (25th to 75th percentile)
over the repetitions (meaningful from about 5 repetitions).
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


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def summarise(df):
    ok = df[df["status"] == "ok"].copy()
    ok["spread"] = (ok["finish_max_ms"] - ok["finish_min_ms"]) / ok["finish_max_ms"].where(ok["finish_max_ms"] > 0)
    g = ok.groupby(KEYS)
    s = g.agg(reps_ok=("time_ms", "size"),
              median_ms=("time_ms", "median"),
              min_ms=("time_ms", "min"),
              sd_ms=("time_ms", "std"),
              q25_ms=("time_ms", q25),
              q75_ms=("time_ms", q75),
              mops_q25=("throughput_mops", q25),
              mops_q75=("throughput_mops", q75),
              median_mops=("throughput_mops", "median"),
              finish_spread=("spread", "median"),
              cs_ns=("cs_ns", "median"),
              region_median_ms=("region_ms", "median"),
              region_q25_ms=("region_ms", q25),
              region_q75_ms=("region_ms", q75),
              experiment=("experiment", "first")).reset_index()
    # relative to std-mutex in the same configuration
    ref = s[s["lock"] == "std-mutex"][[k for k in KEYS if k != "lock"] + ["median_ms"]]
    ref = ref.rename(columns={"median_ms": "ref_ms"})
    s = s.merge(ref, on=[k for k in KEYS if k != "lock"], how="left")
    s["rel_std_mutex"] = s["median_ms"] / s["ref_ms"]
    s["rel_lo"] = s["q25_ms"] / s["ref_ms"]
    s["rel_hi"] = s["q75_ms"] / s["ref_ms"]
    s["iqr_ms"] = s["q75_ms"] - s["q25_ms"]
    # strong scaling speed-up: the same lock on 1 thread
    one = s[s["threads"] == 1][["platform", "mode", "lock", "outside_work", "N", "median_ms"]]
    one = one.rename(columns={"median_ms": "t1_ms"})
    s = s.merge(one, on=["platform", "mode", "lock", "outside_work", "N"], how="left")
    s["speedup"] = s["t1_ms"] / s["median_ms"]
    s["speedup_lo"] = s["t1_ms"] / s["q75_ms"]
    s["speedup_hi"] = s["t1_ms"] / s["q25_ms"]
    return s.drop(columns=["ref_ms", "t1_ms"])


def has_series(s, tag):
    return s[s["experiment"].astype(str).apply(lambda e: tag in e.split("+"))]


def plots(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    plt.rcParams.update({
        "font.family": "sans-serif", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "text.color": "#0b0b0b", "axes.grid": True,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9,
    })

    def lock_order(locks):
        return sorted(locks, key=lambda x: (FAMILY_ORDER.index(family(x)) if family(x) in FAMILY_ORDER else 99, x))

    def style(lock):
        f = family(lock)
        i = FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 0
        return dict(color=COLORS[i], marker=MARKERS[i], linestyle=line_style(lock), linewidth=2, markersize=6)

    def legend(fig, locks):
        handles = [Line2D([0], [0], label=l, **style(l)) for l in lock_order(locks)]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)

    written = []

    def save(fig, fname):
        path = os.path.join(out, fname.replace(" ", "_"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    def panels(d, xcol, panelcol, ycol, ylabel, xlabel, fname, title, panel_label, band=None, logy=False,
               logx=False, hline=None, ideal=False, int_ticks=True):
        if d.empty:
            return
        pv = sorted(d[panelcol].unique())
        cols = min(3, len(pv))
        rows = -(-len(pv) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols + 1.8, 3.0 * rows), squeeze=False, sharex=True)
        for ax, v in zip(axes.flat, pv):
            sub = d[d[panelcol] == v]
            for lock, ld in sub.groupby("lock"):
                ld = ld.sort_values(xcol)
                ax.plot(ld[xcol], ld[ycol], **style(lock))
                if band is not None:
                    ax.fill_between(ld[xcol], ld[band[0]], ld[band[1]], color=style(lock)["color"], alpha=0.15, linewidth=0)
            xs = sorted(sub[xcol].unique())
            if ideal:
                ax.plot(xs, xs, color="#898781", linewidth=1, linestyle=(0, (1, 2)))
            if hline is not None:
                ax.axhline(hline, color="#898781", linewidth=1)
            if logx:
                ax.set_xscale("log", base=2 if xcol == "threads" else 10)
            if int_ticks and xcol in ("threads", "outside_work"):
                ax.set_xticks(xs)
                ax.set_xticklabels([str(int(t)) for t in xs])
                ax.minorticks_off()
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{panel_label} = {v:g}", fontsize=9, color="#52514e", loc="left")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        for ax in list(axes.flat)[len(pv):]:
            ax.axis("off")
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
        legend(fig, d["lock"].unique())
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, fname)

    for plat, g in s.groupby("platform"):
        tag = str(plat).replace(" ", "_")

        # ---- series 1: core scalability -------------------------------------------------
        d = has_series(g, "scale")
        d = d[d["mode"] == "fixed"]
        if not d.empty:
            cs = d["cs_ns"].median()
            head = f"{plat}  |  series 1 scale  |  N={int(d['N'].iloc[0])} (about {cs:,.0f} ns)"
            panels(d, "threads", "outside_work", "median_mops", "critical sections / us", "threads",
                   f"{tag}_s1_scale_throughput.png", head + "  |  fixed work per thread: throughput (higher is better, band = IQR)",
                   "OUTSIDE_WORK", band=("mops_q25", "mops_q75"), logx=True)
            panels(d, "threads", "outside_work", "rel_std_mutex", "time / std-mutex", "threads",
                   f"{tag}_s1_scale_rel.png", head + "  |  whole-program time relative to std-mutex (lower is better)",
                   "OUTSIDE_WORK", band=("rel_lo", "rel_hi"), logx=True, logy=True, hline=1.0)

        # ---- series 2: contention sensitivity ------------------------------------------
        d = has_series(g, "contention")
        if not d.empty:
            cs = d["cs_ns"].median()
            head = f"{plat}  |  series 2 contention  |  N={int(d['N'].iloc[0])} (about {cs:,.0f} ns)"
            panels(d, "outside_work", "threads", "median_mops", "critical sections / us", "OUTSIDE_WORK",
                   f"{tag}_s2_contention_throughput.png", head + "  |  throughput versus work outside the lock (band = IQR)",
                   "threads", band=("mops_q25", "mops_q75"))
            panels(d, "outside_work", "threads", "rel_std_mutex", "time / std-mutex", "OUTSIDE_WORK",
                   f"{tag}_s2_contention_rel.png", head + "  |  whole-program time relative to std-mutex (lower is better)",
                   "threads", band=("rel_lo", "rel_hi"), logy=True, hline=1.0)

        # ---- series 3: critical-section duration -----------------------------------------
        d = has_series(g, "cs")
        for ow, dd in d.groupby("outside_work"):
            head = f"{plat}  |  series 3 critical-section length  |  OUTSIDE_WORK={ow}"
            panels(dd, "cs_ns", "threads", "rel_std_mutex", "time / std-mutex", "critical section (ns)",
                   f"{tag}_s3_cs_ow{ow}_rel.png", head + "  |  whole-program time relative to std-mutex (lower is better)",
                   "threads", band=("rel_lo", "rel_hi"), logx=True, logy=True, hline=1.0, int_ticks=False)
            panels(dd, "cs_ns", "threads", "median_mops", "critical sections / us", "critical section (ns)",
                   f"{tag}_s3_cs_ow{ow}_throughput.png", head + "  |  throughput (band = IQR)",
                   "threads", band=("mops_q25", "mops_q75"), logx=True, logy=True, int_ticks=False)

        # ---- series 4: strong scaling -------------------------------------------------------
        d = has_series(g, "strong")
        d = d[d["mode"] == "strong"]
        if not d.empty:
            cs = d["cs_ns"].median()
            head = f"{plat}  |  series 4 strong scaling  |  N={int(d['N'].iloc[0])} (about {cs:,.0f} ns)"
            panels(d, "threads", "outside_work", "speedup", "speed-up vs 1 thread", "threads",
                   f"{tag}_s4_strong_speedup.png", head + "  |  speed-up of the same lock (dotted = ideal, band = IQR)",
                   "OUTSIDE_WORK", band=("speedup_lo", "speedup_hi"), logx=True, ideal=True)
            panels(d, "threads", "outside_work", "rel_std_mutex", "time / std-mutex", "threads",
                   f"{tag}_s4_strong_rel.png", head + "  |  whole-program time relative to std-mutex (lower is better)",
                   "OUTSIDE_WORK", band=("rel_lo", "rel_hi"), logx=True, logy=True, hline=1.0)

        # ---- series 5: oversubscription (bars: logical CPUs vs twice that) -----------------
        d = has_series(g, "oversub")
        d = d[d["mode"] == "fixed"]
        if not d.empty:
            ths = sorted(d["threads"].unique())
            lo_t, hi_t = ths[0], ths[-1]
            ows = sorted(d["outside_work"].unique())
            locks = lock_order(d["lock"].unique())
            for metric, ycol, qlo, qhi, ylabel, fn in (
                    ("throughput", "median_mops", "mops_q25", "mops_q75", "critical sections / us", "throughput"),
                    ("whole-program time", "median_ms", "q25_ms", "q75_ms", "ms", "time")):
                fig, axes = plt.subplots(1, len(ows), figsize=(max(4.2, 0.55 * len(locks)) * len(ows) + 1.5, 3.6), squeeze=False)
                for ax, ow in zip(axes.flat, ows):
                    x = np.arange(len(locks))
                    for k, (t, hatch) in enumerate(((lo_t, None), (hi_t, "///"))):
                        vals, lo_e, hi_e, cols = [], [], [], []
                        for l in locks:
                            r = d[(d["lock"] == l) & (d["threads"] == t) & (d["outside_work"] == ow)]
                            if r.empty:
                                vals.append(np.nan); lo_e.append(0); hi_e.append(0)
                            else:
                                v = float(r[ycol].iloc[0])
                                vals.append(v); lo_e.append(max(0.0, v - float(r[qlo].iloc[0]))); hi_e.append(max(0.0, float(r[qhi].iloc[0]) - v))
                            cols.append(style(l)["color"])
                        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=cols, hatch=hatch, edgecolor="#fcfcfb",
                               yerr=[lo_e, hi_e], error_kw=dict(ecolor="#52514e", lw=0.8, capsize=2))
                    ax.set_xticks(x)
                    ax.set_xticklabels(locks, rotation=45, ha="right", fontsize=8)
                    ax.set_yscale("log")
                    ax.set_title(f"OUTSIDE_WORK = {ow}", fontsize=9, color="#52514e", loc="left")
                    ax.set_ylabel(ylabel)
                handles = [Patch(facecolor="#898781", label=f"{lo_t} threads (logical CPUs)"),
                           Patch(facecolor="#898781", hatch="///", edgecolor="#fcfcfb", label=f"{hi_t} threads (2x logical)")]
                fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
                cs = d["cs_ns"].median()
                fig.suptitle(f"{plat}  |  series 5 oversubscription  |  N={int(d['N'].iloc[0])} (about {cs:,.0f} ns)  |  "
                             f"fixed work per thread: {metric} (error bars = IQR)", x=0.01, ha="left", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, 0.93))
                save(fig, f"{tag}_s5_oversub_{fn}.png")
    return written


def oversub_table(s):
    """Series 5: throughput and whole-program time of every lock at the logical-CPU count and at twice that."""
    d = has_series(s, "oversub")
    d = d[d["mode"] == "fixed"]
    rows = []
    for (plat, ow, lock), g in d.groupby(["platform", "outside_work", "lock"]):
        g = g.sort_values("threads")
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[-1]
        rows.append(dict(platform=plat, outside_work=ow, lock=lock, threads_lo=int(a["threads"]), threads_hi=int(b["threads"]),
                         mops_lo=a["median_mops"], mops_hi=b["median_mops"], mops_ratio=b["median_mops"] / a["median_mops"],
                         time_ms_lo=a["median_ms"], time_ms_hi=b["median_ms"], time_ratio=b["median_ms"] / a["median_ms"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results/plots")
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
    ot = oversub_table(s)
    if len(ot):
        ot.round(4).to_csv(os.path.join(a.out, "oversub_table.csv"), index=False)
        print("\nSeries 5 (oversubscription), fixed work per thread: throughput = critical sections / us, time = whole program")
        print("time ratio ideal = 2 (twice the work), throughput ratio ideal = 1")
        print(ot.drop(columns=["platform"]).round(3).to_string(index=False))

    if not a.no_plots:
        try:
            for p in plots(s, a.out):
                print("figure  ->", p)
        except ImportError:
            print("matplotlib not installed: skipped figures (pip install matplotlib)")


if __name__ == "__main__":
    sys.exit(main())