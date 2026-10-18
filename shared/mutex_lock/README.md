# Section 1 – Mutex Locks

Spinlock implementations (TAS, TTAS, Ticket, MCS) and a correctness/stress harness for the
Parallel Computing assignment. Header-only locks, C++17, no external dependencies.

| Path | Purpose |
|---|---|
| `src/lock.h` | All lock implementations, backoff policies, portability layer |
| `src/test_correctness.cpp` | Correctness and stress test harness |
| `src/bench_mutex.cpp` | Performance and scalability benchmark (assignment workload, CSV output) |
| `src/analyze_bench.py` | Summaries and figures from the timing CSV files (median, IQR; pandas, matplotlib) |
| `src/config_check.py` | Checks shared by both analyzers: smoke and stale rows, config ids, timeouts, platform table |
| `config/reference.cfg` | Frozen benchmark workload (N, `OUTSIDE_WORK` sets, iteration counts); read at run time, identical on every platform |
| `src/analyze_fairness.py` | Summaries and figures from the fairness CSV files (Jain's index, wait times) |
| `Makefile` | Build/run shortcuts (tests: `test`, `test-quick`, `test-control`, `test-tsan`, `test-mutate`; benchmark: `bench-check`, `bench-smoke`, `bench-quick`, `bench`, `bench-fair`, `bench-analyze`, `bench-pilot`, `OMP=1`) |
| `report_mutex.tex` | LaTeX report (design, rationale, test method, benchmark methodology) |
| `build/` | Compiler output (binaries), created by make; `make clean` removes it |
| `results/` | Everything the programs write: CSV files and `plots/`; kept by `make clean` (`make distclean` removes it) |

## Implementations

All locks provide `acquire()` / `release()` (the assignment interface), `lock()` / `unlock()` /
`try_lock()` (so `std::lock_guard` and `std::unique_lock` work) and `static constexpr bool is_fifo`.
They are non-copyable, and hot atomics are padded to separate cache lines.

| Name | Algorithm | Notes |
|---|---|---|
| `lock_tas` | Test-and-set | Textbook, no backoff |
| `lock_tas_pause` / `_yield` / `_sleep` | TAS + exponential backoff | Pause hints with jitter / `yield()` / `sleep_for(ns)` (the spec's version) |
| `lock_ttas` | Test-and-test-and-set | Spins on a load, exchanges only when the lock looks free |
| `lock_ttas_pause` / `_yield` / `_sleep` | TTAS + backoff | Backoff only after a *failed exchange* |
| `lock_ticket` | Ticket lock (FIFO) | `next_` and `serving_` on separate cache lines |
| `lock_ticket_prop` | Ticket + proportional backoff | Waiters pause in proportion to their distance from the head |
| `lock_mcs` | MCS queue lock (FIFO) | Local spinning; per-thread node pool, so locks can be nested and released in any order |

Backoff policies are template parameters (`NoBackoff`, `PauseBackoff<Min,Max>`,
`YieldBackoff<Min,Max>`, `SleepBackoff<MinNs,MaxNs>`), so custom variants are one line:

```cpp
using my_lock = basic_lock_ttas<PauseBackoff<8, 4096>>;
```

Waiting loops use a hybrid strategy: `cpu_relax()` for `LOCK_SPIN_LIMIT` iterations, then
`yield()`. This keeps FIFO locks usable when there are more threads than cores.

Usage matches the assignment skeleton:

```cpp
#include "lock.h"
lock_mcs lock;
lock.acquire();  /* critical section */  lock.release();
```

## Build and run

Requires a C++17 compiler and pthreads (or the Windows equivalent).

```bash
make              # builds build/test_correctness and build/bench_mutex
make test         # full suite (about 1 minute on a 2-core machine; longer with more cores)
make test-quick   # fast subset
make test-control # negative control: harness must detect a lock that does nothing
make test-tsan    # ThreadSanitizer build and run
make test-mutate  # mutation test (failures / races are EXPECTED here)
```

Without make:

```bash
mkdir -p build && g++ -std=c++17 -O2 -pthread src/test_correctness.cpp -o build/test_correctness
build/test_correctness --help
```

Useful options:

```
--lock=tas,mcs      run only these variants (see --list)
--threads=1,2,4     thread counts (default: powers of two up to hardware threads, plus 2x)
--trials=N          trials per configuration (default 3)
--scale=X           scale all iteration counts (use 0.1 under TSan)
--quick             --trials=1 --scale=0.25
--stall-ms=N        deadlock watchdog: fail if NO thread progresses for N ms (default 5000)
--control           negative control only
```

The exit code is 0 only if every selected suite passed. A stalled lock terminates the process (stuck
spinning threads cannot be cancelled); re-run with `--lock=` to continue with the others.

## What the tests check

1. Mutual exclusion (overlap detector with a seq_cst canary)
2. Protected non-atomic counter (lost updates; races are visible to TSan)
3. Multi-field invariant (memory ordering between consecutive holders)
4. Fairness: acquisition distribution; for FIFO locks a hard check on the overlap window (only when threads ≤ hardware threads) and a soft overtaking indicator
5. Four stress profiles, including the lock holder yielding while holding the lock
6. API: `try_lock`, `std::lock_guard`, `std::unique_lock`
7. Several lock objects, nesting and hand-over-hand (non-LIFO) release
8. Backoff policy unit checks (delay sequences, saturation, overflow edge cases)

Variants with degenerate backoff parameters (`tas-pause-1-1`, `ttas-pause-1-1`, `tas-sleep-1-1`) run a
reduced suite. Sleep-based variants run with fewer iterations because they are much slower.

## Platform notes

**Linux** – nothing special. For benchmarking later, consider pinning threads and setting the CPU
governor to `performance`.

**macOS (Intel and Apple Silicon)** – works with Apple clang (`clang++ -std=c++17 -O2 -pthread`).
Apple Silicon is ARM (weak memory model), so run the suite there: it exercises acquire/release
orderings that x86 hides. There is no thread affinity API, and `pthread_spin_lock` does not exist.
The cache-line padding is 128 bytes.

**Windows** – MSVC: `cl /std:c++17 /O2 /EHsc src\test_correctness.cpp`. MinGW-w64 and WSL2 also work
(WSL2 behaves like Linux). On MSVC the pause hint uses `_mm_pause()` / `__yield()`. `sleep_for`
resolution is coarse (about 1–15 ms by default), so the `*_sleep` variants are very slow there.

**ARM Linux (Raspberry Pi, servers)** – works with GCC/Clang; the `yield` hint is a no-op on some
cores. Run the suite here as well.

**Hybrid CPUs (Intel P/E cores)** – results can vary between runs depending on which cores the
threads land on; note this in the report and pin threads when benchmarking.

**ThreadSanitizer** - run it with few threads (`make test-tsan` uses 1, 2, 4). With many threads, plain `tas`
(no backoff) can stop making progress under TSan: TSan serialises atomic operations on the same address, so
the waiters' constant `test_and_set` starves the holder's release. `tas-pause`, `ttas` and the other variants
are unaffected, and the same lock passes in a normal build. Races are detected regardless of the thread count.

**Oversubscription** – with more threads than cores, spinlocks are slow by nature. The default
thread counts include 2x the hardware thread count on purpose. The watchdog measures progress,
not wall time, so slow is not reported as a deadlock.

**Compile-time knobs** – `-DLOCK_CACHE_LINE=N`, `-DLOCK_SPIN_LIMIT=N`,
`-DLOCK_MUTATE_RELAXED_RELEASE` (tests only).

## Benchmark

`bench_mutex.cpp` runs the assignment's kernel (`acquire(); is_prime(N); release();` followed by
`OUTSIDE_WORK` calls of `is_prime(N)`) for every selected lock and appends one CSV row per run.

```bash
make bench-pilot PLATFORM=ref                # one-time, reference machine only (see Fixed workload)
make bench-quick PLATFORM=mylaptop     # shorter run: all five series, 5 reps -> results/quick_mylaptop.csv
make bench PLATFORM=mylaptop           # the run for the report: all five series, 10 reps -> results/bench_mylaptop.csv
make bench-fair PLATFORM=mylaptop      # fairness experiment -> results/fair_mylaptop.csv
make bench OMP=1 PLATFORM=mylaptop     # optional: any of the three targets with the OpenMP baselines added
make bench-analyze                           # summaries and figures from results/bench_*.csv and fair_*.csv -> results/plots/
```

Run `bench-quick` first to check the setup and get a first look (`python3 src/analyze_bench.py results/quick_*.csv --out results/plots_quick`),
then `bench` for the data of the report. The two write different files, so they are never pooled. Extra benchmark options can be
passed with `BENCHFLAGS`, e.g. `make bench BENCHFLAGS="--series=oversub --locks=tas,mcs"`.
`OMP=1` builds `bench_mutex_omp` and adds `omp-lock` and `omp-critical` to the series; in the fairness experiment only
`omp-lock` takes part (`omp-critical` is a block, not a lock object). Use the same setting on every platform of a comparison. CSV files written by an earlier version have fewer columns; the program
refuses to append to them, so move them away.

Without make: `mkdir -p build && g++ -std=c++17 -O2 -pthread src/bench_mutex.cpp -o build/bench_mutex && build/bench_mutex --help`.
Run it from the project root (it reads `config/reference.cfg`; `--config=FILE` selects another). `build/bench_mutex --dry-run` prints the series, their configurations and the number of runs without executing anything.
`--series=scale,contention,cs,strong,oversub` runs a subset of the series (default: all five). `--filter=n=medium,outside=0|8,threads=2` runs only the matching configurations (keys `n` (number or small|medium|large), `outside`, `threads`, `series`, `mode`).

**Contenders** (`--list`): own `tas`, `tas-pause`, `ttas`, `ttas-pause`, `ticket`, `ticket-prop`, `mcs`;
baselines `std-mutex`, `pthread-mutex`, `pthread-spin` (Linux only; macOS has no `pthread_spin_lock`),
and, when built with `-fopenmp -DUSE_OPENMP`, `omp-lock` and `omp-critical`. The slow variants
(`*-sleep`, `*-yield`) are opt-in: `--locks=core,tas-sleep,ttas-sleep` or `--locks=all`.
Everything is behind the same `acquire()`/`release()` interface (`omp-critical` has its own runner,
because `#pragma omp critical` is a block, not a lock object).

**Metric.** The primary time is the whole-program execution time: from just before the first thread
is created until the last join. It includes thread creation and the start-up skew, as a timed
program would. Also recorded per run: `region_ms` (start gate to join, creation excluded) and the
first/last per-thread finish times (their spread indicates unfairness).

**Two scaling modes** (set by the series: `strong` uses strong scaling, all others fixed work per thread):

| Mode | Work | Ideal behaviour | Reported |
|---|---|---|---|
| `fixed` | every thread runs `iters` iterations (the skeleton's `NUM_OF_ITERS`); total work grows with the threads | time stays flat | time, throughput (critical sections per microsecond) |
| `strong` | a fixed total, split across the threads | time falls as 1/threads | time, speed-up versus the same lock on 1 thread |

**Fixed workload.** The primary benchmark runs the same computation on every platform: the same N, the same
`OUTSIDE_WORK` sets and the same iteration counts. They are frozen in `config/reference.cfg`, which the program reads at
run time and never changes. The file holds
two explicit tables, kept apart on purpose: `iters N OUTSIDE_WORK count` is the number of iterations **per thread**
(fixed-work series) and `strong N OUTSIDE_WORK total` is the **total** number of iterations split across the threads
(strong-scaling series, where each thread runs `total / threads`). Small, medium and large critical sections are
N = 79, 739 and 7393 (about 0.1, 1 and 10 us on the reference machine: AMD Ryzen 5 6600H, gcc 15.2.0, `-O2`, WSL2); one
uncontended thread of each configuration runs about 50 ms there. A faster or slower CPU therefore has a shorter or longer
critical section for the same work. The duration of `is_prime(N)` measured on the running machine is stored in every row
(`cs_ns`) and quoted in the discussion; it never changes N or a count.

Every CSV row carries `config_id`, a fingerprint of the normalised config text, the repetitions, the warm-up, the iteration
scale and the parameters of the lock variants. All platforms of one comparison must show the same id; the analysis
reports differences.

| Quantity | Fixed (same on every platform) | Measured per platform (stored, never used to change the workload) |
|---|---|---|
| N (critical-section size), `OUTSIDE_WORK` sets, thread-count rules | yes | |
| Iterations per thread, strong-scaling totals | yes (`config/reference.cfg`) | |
| Repetitions, warm-up, lock parameters, flags | yes | |
| Thread counts (powers of two up to the logical CPUs, physical and logical counts) | rule is fixed | the resulting values depend on the CPU |
| Duration of `is_prime(N)`, run times, hardware, compiler, kernel | | yes (`cs_ns`, `results/platform_<platform>.txt`, `platform_table.csv`) |

**Workflow.**

```bash
make bench-check PLATFORM=mylaptop      # platform report + sanity checks (warnings only) -> results/platform_mylaptop.txt
make bench-smoke PLATFORM=mylaptop      # tiny end-to-end run (1% of the iterations): rows are marked and never analysed
make bench PLATFORM=mylaptop      # the real run; sanity checks run first and their warnings are printed
make bench-analyze                      # checks the rows against config/reference.cfg, then summaries and figures
```

`check` warns when: the cost of `is_prime` does not grow with N like on the reference machine; doubling a batch of
`is_prime(N)` calls does not double the time (the compiler may have removed or hoisted the calls); the single-thread
benchmark loop does not take 1 : 3 : 9 times as long for `OUTSIDE_WORK` 0 : 2 : 8, or its summed results are wrong; the
fastest configuration runs shorter than 20 ms; or a baseline is unavailable. Warnings never change anything.

**One-time pilot (reference machine only).** `make bench-pilot` measures the real single-thread benchmark loop (lock acquire,
critical section, release, `OUTSIDE_WORK` outside calls) for every (N, `OUTSIDE_WORK`) pair of the series in batches of at
least 20 ms, checks that all locks agree on the loop time, projects the worst case at twice the logical CPUs, runs the sanity
checks above and writes a **candidate** `config/pilot_<platform>.cfg` (options `--target-ms`, `--pilot-ns`,
`--pilot-cpus`). It warns only and never edits `config/reference.cfg`; review the candidate and copy it by hand.
`--iters-scale=K` multiplies all counts for smoke tests only: those rows carry `iters_scale != 1` and the analysis
excludes them.

**Revision policy.** Keep the counts as they are unless a configuration is impractical, for example it hits the run cap
(`status=timeout`; the analysis lists these as revision candidates). Then edit that count in `config/reference.cfg`, bump
`version`, and rerun that configuration on **all** platforms, e.g.
`make bench BENCHFLAGS="--filter=n=medium,outside=16"`. The analysis excludes rows whose counts differ from the current
file (stale rows) and says which platform has to rerun which configuration.

**Experiment series.** The default run is five series. Thread counts: `t_scale` = powers of two up to the logical CPUs plus the logical-CPU count and the exact
physical-core count; `t3` = {1, physical cores, logical CPUs}. A configuration that belongs to several series
runs once and carries all its tags in the `experiment` column.

| Series | Question | Mode | Threads | OUTSIDE_WORK | N |
|---|---|---|---|---|---|
| `scale` (1) | core scalability | fixed work per thread | `t_scale` | 0, 2, 8 | medium |
| `contention` (2) | sensitivity to the contention level | fixed | `t3` | 0, 1, 2, 4, 8, 16 | medium |
| `cs` (3) | effect of critical-section length | fixed | `t3` | 0, 8 | small, medium, large |
| `strong` (4) | strong scaling | fixed **total** work | `t_scale` | 2 | medium |
| `oversub` (5) | oversubscription | fixed work per thread | logical CPUs and 2x logical | 0, 8 | medium |

Series 5 reports throughput **and** whole-program time for every lock. With fixed work per thread the ideal
ratio 2x-threads / 1x-threads is 2 for time and 1 for throughput; `analyze_bench.py` prints this table and writes
`results/plots/oversub_table.csv`. Oversubscribed runs are marked `oversub` and mostly measure the OS scheduler.
Iteration counts are the same for all locks of a configuration and, with the fixed workload, on every platform.

**Protocol.** One warm-up pass, then at least 5 measured repetitions, 10 for the full `make bench`
(`--reps`, `--warmup`). Reported: median and interquartile range (IQR); minimum and standard deviation are
supplementary. Locks are run in block-randomised
order: each pass runs every lock once in a fresh random order (`--seed`), so drift in frequency or
temperature is spread over the locks. `is_prime` reads N through a volatile reference so the compiler
cannot fold or hoist the work. Every run is validated: completed iterations must equal the expected
count and the summed `is_prime` results must equal `completed * (1 + OUTSIDE_WORK) * is_prime(N)`,
otherwise the run is `invalid`. A run that exceeds `--max-run-ms` (20 s) is stopped through a flag
checked once per iteration and recorded as `timeout`; both statuses are excluded from the statistics
and reported at the end.

**Pinning.** `--pin` pins thread i to a CPU, physical cores first and SMT siblings last (Linux via
`sched_getaffinity`/`pthread_setaffinity_np`, Windows via `SetThreadAffinityMask`; macOS has no
affinity API, so the flag is ignored there with a warning). On hybrid P/E-core CPUs, note which cores
were used. For `omp-critical`, set `OMP_PROC_BIND` / `OMP_PLACES` yourself.

**Counting without distortion.** Every thread counts its own iterations in a private, cache-line padded
slot and the counts are summed after the join; there is no shared counter in the timed loop (the only shared
variable is the stop flag, which is read-only unless a run times out).

**Compiler settings.** Every variant is built with the same compiler and flags (`-O2`, recorded in each CSV
row through `-DBENCH_FLAGS`). The benchmark warns if it was built without optimisation or with
ThreadSanitizer; sanitizer builds (`make test-tsan`, `make test-mutate`) are correctness diagnostics only.

**Lock parameters.** The header and every CSV row (`lock_params`) state each variant's parameters, taken
from the lock types: `tas-pause`/`ttas-pause` use pause backoff 4..1024 `cpu_relax` (doubling, jitter in
[d/2, d], yield once saturated); `tas-sleep`/`ttas-sleep` use `sleep_for(nanoseconds(d))` with d from 1000 to
128000 ns (doubling); `*-yield` uses 1..64 `yield()` calls; `ticket-prop` pauses 64 `cpu_relax` per position
in line (cap 16384); the waiting loops of TTAS, ticket and MCS use SpinWait (2048 `cpu_relax`, then `yield`).

**Output.** `results/bench_<platform>.csv` has one row per run (platform, CPU model, OS, arch, compiler,
usable CPUs, pinning, series, mode, lock, threads, OUTSIDE_WORK, N, cs_ns, status, time_ms,
region_ms, finish_min/max_ms, completed, throughput) plus metadata: physical cores, kernel version, compile
flags, the CPU list used for pinning and the lock parameters. Files from several machines can be passed to
`analyze_bench.py` together; it writes `results/plots/bench_summary.csv` (median, IQR, min, sd, speed-up, ratio to
`std-mutex`, finish spread) and one figure set per series: scale (throughput and time relative to `std-mutex`
vs threads, per OUTSIDE_WORK), contention (vs OUTSIDE_WORK, panel per thread count), cs (vs critical-section length),
strong (speed-up) and oversub (grouped bars, logical vs 2x logical, throughput and time). Lines are medians, shaded bands the IQR. Figure colour encodes the lock
family and line style the variant (solid = plain, dashed = pause/proportional backoff or
`omp-critical`, dotted = sleep, dash-dot = yield).

**Fairness experiment** (`--mode=fairness`, `make bench-fair`). Giving every thread the same number of
iterations says little about fairness, because all threads are told to do the same amount of work. Here all
threads compete for a fixed time (`--duration-ms`, default 1000) and each keeps a private acquisition counter.
Every 32nd acquisition additionally times `acquire()` into a private pre-allocated log (two clock reads, only on
sampled iterations); counters and logs are merged after the join, so nothing shared is touched while the lock
is under test. Defaults: at least 2 threads, sustained contention (`OUTSIDE_WORK` 0, medium N), 5
repetitions, locks in block-randomised order, written to a separate file `results/fair_<platform>.csv`. Reported per run:
acquisitions per thread (min, median, max), max/min ratio, Jain's fairness index
`(sum x)^2 / (n * sum x^2)` (1 = equal shares), and `acquire()` latency p50, p99 and max plus the spread of the
per-thread p99. `analyze_fairness.py` (called by `make bench-analyze`) reports medians and the IQR of Jain's index.
Interpret with care: FIFO order (ticket, MCS) is not the same as equal throughput, a finite experiment
characterises the observed behaviour and does not prove that starvation is impossible, and oversubscribed runs
(marked `oversub`) mostly measure the OS scheduler. The correctness harness's fairness test is separate and
uses a shared log, which is acceptable there because it checks ordering, not performance.

**Platform notes for benchmarking.** Close other applications, plug in power, disable battery saver,
and let a laptop cool down between full sweeps. On Linux, `performance` governor
(`cpupower frequency-set -g performance`) reduces noise. On Windows, use the High performance power
plan; `sleep_for` has a 1-15 ms resolution there, so keep `*-sleep` out of the sweep or accept its
cost. Run the same `make bench` on every machine; the number of platforms must be at least the group
size.

## Limitations

- A lock must be released by the thread that acquired it (mutex semantics), as with `std::mutex`.
- MCS allocates one node per thread per concurrently held lock on first use (freed at thread exit).
- The mutation test shows the harness catches weakened release orderings under TSan and on ARM;
  on x86 without TSan such bugs may not show up natively.

> Distributed-memory (MPI-3 RMA) lock versions: see `distributed-memory/README.md`.