"""Comparability checks shared by analyze_bench.py and analyze_fairness.py.

The benchmark runs one frozen workload (config/reference.cfg) on every platform. Before anything is
summarised, the rows are checked against that file:

  * smoke rows (iters_scale != 1) are dropped,
  * stale rows, whose iteration counts differ from the canonical config (an older config version, or a
    configuration that was revised and not yet rerun on this platform), are dropped and listed,
  * differing config_id values (fingerprint of config, repetitions, lock parameters) are reported,
  * timed-out configurations are listed as candidates for a count revision,
  * missing baselines and differing flags / compiler / pinning are reported.
"""
import pandas as pd


def load_cfg(path):
    """Parse the canonical config: returns dict(version, N=[s,m,l], cs_ns=[..], iters={(N,ow):n}, strong={(N,ow):n})."""
    c = {"version": 0, "N": [0, 0, 0], "cs_ns": [0.0, 0.0, 0.0], "iters": {}, "strong": {}, "reference": ""}
    with open(path) as f:
        for line in f:
            if line.startswith("reference "):
                c["reference"] = line[len("reference "):].split("#")[0].strip()
            t = line.split("#")[0].split()
            if not t:
                continue
            if t[0] == "version":
                c["version"] = int(t[1])
            elif t[0] == "n":
                i = ["small", "medium", "large"].index(t[1])
                c["N"][i], c["cs_ns"][i] = int(t[2]), float(t[3])
            elif t[0] in ("iters", "strong"):
                c[t[0]][(int(t[1]), int(t[2]))] = int(t[3])
    return c


def _expected(row, cfg):
    key = (int(row["N"]), int(row["outside_work"]))
    if row["mode"] == "strong":
        v = cfg["strong"].get(key)
        return None if v is None else max(v, int(row["threads"]))
    return cfg["iters"].get(key)


def validate_timing(df, cfg, log=print):
    """Returns the rows that belong to the canonical comparison."""
    if "config_id" not in df.columns or "iters_scale" not in df.columns:
        raise SystemExit("These CSV files lack the config_id / iters_scale columns (written by an older benchmark).\n"
                         "Rerun the benchmark with the current version; the old files are not comparable.")
    n0 = len(df)
    smoke = df["iters_scale"] != 1.0
    if smoke.any():
        log(f"smoke rows excluded: {int(smoke.sum())} (iters_scale != 1)")
    df = df[~smoke].copy()
    exp = df.apply(lambda r: _expected(r, cfg), axis=1)
    got = df.apply(lambda r: r["total_iters"] if r["mode"] == "strong" else r["iters_per_thread"], axis=1)
    stale = exp.isna() | (exp != got)
    if stale.any():
        log(f"WARNING: {int(stale.sum())} stale row(s) excluded: counts differ from config version {cfg['version']}; "
            "rerun these configurations (--filter) on that platform:")
        s = df[stale].groupby(["platform", "mode", "N", "outside_work"]).size().rename("rows")
        log(s.to_string())
    df = df[~stale]
    ids = df.groupby("platform")["config_id"].agg(lambda x: sorted(set(x)))
    allids = sorted({i for v in ids for i in v})
    if len(allids) > 1:
        log("WARNING: more than one config_id among the kept rows (different repetitions, lock parameters or config text):")
        for p, v in ids.items():
            log(f"  {p}: {', '.join(v)}")
    elif allids:
        log(f"config_id {allids[0]} on all platforms (config version {cfg['version']})")
    log(f"{len(df)} of {n0} rows used")
    return df


def timeouts(df, log=print):
    t = df[df["status"] == "timeout"]
    if len(t):
        log(f"{len(t)} timed-out run(s): revision candidates (edit the count in the canonical config, bump the version, "
            "rerun that configuration on ALL platforms):")
        log(t.groupby(["platform", "series", "mode", "N", "outside_work", "threads", "lock"]).size().rename("runs").to_string())


def platform_table(df, cfg, log=print):
    rows = []
    for p, g in df.groupby("platform"):
        r0 = g.iloc[0]
        med = g[g["N"] == cfg["N"][1]]["cs_ns"].median()
        locks = sorted(g["lock"].unique())
        rows.append(dict(platform=p, cpu_model=r0["cpu_model"], os=r0["os"], compiler=r0["compiler"],
                         usable_cpus=int(r0["usable_cpus"]), physical_cores=int(r0["physical_cores"]),
                         pinned=int(g["pinned"].min()), flags=r0["flags"],
                         cs_ns_medium=round(float(med), 1), cs_vs_reference=round(float(med) / cfg["cs_ns"][1], 2),
                         config_id=",".join(sorted(set(g["config_id"]))), rows=len(g),
                         timeouts=int((g["status"] == "timeout").sum()), locks=" ".join(locks)))
    t = pd.DataFrame(rows)
    if len(t) == 0:
        return t
    for col, what in (("flags", "compiler flags"), ("compiler", "compilers"), ("pinned", "pinning")):
        if t[col].nunique() > 1:
            log(f"NOTE: {what} differ between platforms: " + "; ".join(f"{a}={b}" for a, b in zip(t["platform"], t[col])))
    for p, g in df.groupby("platform"):
        missing = [b for b in ("std-mutex", "pthread-mutex", "pthread-spin") if b not in set(g["lock"])]
        if missing:
            log(f"NOTE: {p}: baseline(s) not in the data: {', '.join(missing)}")
    if (t["pinned"] == 0).any():
        log("NOTE: some platform was run without pinning: " + ", ".join(t[t["pinned"] == 0]["platform"]))
    return t


def validate_fairness(df, cfg, log=print):
    if "config_id" not in df.columns:
        raise SystemExit("These CSV files lack the config_id column (written by an older benchmark); rerun the fairness experiment.")
    ids = df.groupby("platform")["config_id"].agg(lambda x: sorted(set(x)))
    allids = sorted({i for v in ids for i in v})
    if len(allids) > 1:
        log("WARNING: more than one config_id among the fairness rows (different repetitions, duration, lock parameters or config):")
        for p, v in ids.items():
            log(f"  {p}: {', '.join(v)}")
    else:
        log(f"config_id {allids[0]} on all platforms")
    wrong = df[(df["N"] != cfg["N"][1])]
    if len(wrong):
        log(f"WARNING: {len(wrong)} fairness row(s) with N != canonical medium N {cfg['N'][1]} excluded")
        df = df[df["N"] == cfg["N"][1]]
    return df