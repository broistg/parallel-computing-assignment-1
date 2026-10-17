# Section 1 – Mutex Locks

Spinlock implementations (TAS, TTAS, Ticket, MCS) and a correctness/stress harness for the Parallel Computing assignment. Header-only locks, C++17, no external dependencies.

| File | Purpose |
|---|---|
| `lock.h` | All lock implementations, backoff policies, portability layer |
| `test_correctness.cpp` | Correctness and stress test harness |
| `bench_mutex.cpp` | Performance and scalability benchmark (assignment workload, CSV output) |
| `analyze_bench.py` | Summaries and figures from the benchmark CSV files (pandas, matplotlib) |
| `Makefile` | Build/run shortcuts (tests: `test`, `quick`, `control`, `tsan`, `mutate`; benchmark: `bench-quick`, `bench`, `bench-omp`, `analyze`) |
| `report_mutex.tex` | LaTeX report (design, rationale, test method, benchmark methodology) |

## Implementations

All locks provide `acquire()` / `release()` (the assignment interface), `lock()` / `unlock()` / `try_lock()` (so `std::lock_guard` and `std::unique_lock` work) and `static constexpr bool is_fifo`. They are non-copyable, and hot atomics are padded to separate cache lines.

| Name | Algorithm | Notes |
|---|---|---|
| `lock_tas` | Test-and-set | Textbook, no backoff |
| `lock_tas_pause` / `_yield` / `_sleep` | TAS + exponential backoff | Pause hints with jitter / `yield()` / `sleep_for(ns)` (the spec's version) |
| `lock_ttas` | Test-and-test-and-set | Spins on a load, exchanges only when the lock looks free |
| `lock_ttas_pause` / `_yield` / `_sleep` | TTAS + backoff | Backoff only after a *failed exchange* |
| `lock_ticket` | Ticket lock (FIFO) | `next_` and `serving_` on separate cache lines |
| `lock_ticket_prop` | Ticket + proportional backoff | Waiters pause in proportion to their distance from the head |
| `lock_mcs` | MCS queue lock (FIFO) | Local spinning; per-thread node pool, so locks can be nested and released in any order |

Backoff policies are template parameters (`NoBackoff`, `PauseBackoff<Min,Max>`, `YieldBackoff<Min,Max>`, `SleepBackoff<MinNs,MaxNs>`), so custom variants are one line:

```cpp
using my_lock = basic_lock_ttas<PauseBackoff<8, 4096>>;
```

Waiting loops use a hybrid strategy: `cpu_relax()` for `LOCK_SPIN_LIMIT` iterations, then `yield()`. This keeps FIFO locks usable when there are more threads than cores.

Usage matches the assignment skeleton:

```cpp
#include "lock.h"
lock_mcs lock;
lock.acquire();  /* critical section */  lock.release();
```

## Build and run

Requires a C++17 compiler and pthreads (or the Windows equivalent).

```bash
make              # builds test_correctness
make test         # full suite (about 1 minute on a 2-core machine; longer with more cores)
make quick        # smoke test
make control      # negative control: harness must detect a lock that does nothing
make tsan         # ThreadSanitizer build and run
make mutate       # mutation test (failures / races are EXPECTED here)
```

Without make:

```bash
g++ -std=c++17 -O2 -pthread test_correctness.cpp -o test_correctness
./test_correctness --help
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

The exit code is 0 only if every selected suite passed. A stalled lock terminates the process (stuck spinning threads cannot be cancelled); re-run with `--lock=` to continue with the others.

## What the tests check

1. Mutual exclusion (overlap detector with a seq_cst canary)
2. Protected non-atomic counter (lost updates; races are visible to TSan)
3. Multi-field invariant (memory ordering between consecutive holders)
4. Fairness: acquisition distribution; for FIFO locks a hard check on the overlap window (only when threads ≤ hardware threads) and a soft overtaking indicator
5. Four stress profiles, including the lock holder yielding while holding the lock
6. API: `try_lock`, `std::lock_guard`, `std::unique_lock`
7. Several lock objects, nesting and hand-over-hand (non-LIFO) release
8. Backoff policy unit checks (delay sequences, saturation, overflow edge cases)

Variants with degenerate backoff parameters (`tas-pause-1-1`, `ttas-pause-1-1`, `tas-sleep-1-1`) run a reduced suite. Sleep-based variants run with fewer iterations because they are much slower.

## Platform notes

**Linux** – nothing special. For benchmarking later, consider pinning threads and setting the CPU governor to `performance`.

**macOS (Intel and Apple Silicon)** – works with Apple clang (`clang++ -std=c++17 -O2 -pthread`). Apple Silicon is ARM (weak memory model), so run the suite there: it exercises acquire/release orderings that x86 hides. There is no thread affinity API, and `pthread_spin_lock` does not exist. The cache-line padding is 128 bytes.

**Windows** – MSVC: `cl /std:c++17 /O2 /EHsc test_correctness.cpp`. MinGW-w64 and WSL2 also work (WSL2 behaves like Linux). On MSVC the pause hint uses `_mm_pause()` / `__yield()`. `sleep_for` resolution is coarse (about 1–15 ms by default), so the `*_sleep` variants are very slow there.

**ARM Linux (Raspberry Pi, servers)** – works with GCC/Clang; the `yield` hint is a no-op on some cores. Run the suite here as well.

**Hybrid CPUs (Intel P/E cores)** – results can vary between runs depending on which cores the threads land on; note this in the report and pin threads when benchmarking.

**ThreadSanitizer** - run it with few threads (`make tsan` uses 1, 2, 4). With many threads, plain `tas` (no backoff) can stop making progress under TSan: TSan serialises atomic operations on the same address, so the waiters' constant `test_and_set` starves the holder's release. `tas-pause`, `ttas` and the other variants are unaffected, and the same lock passes in a normal build. Races are detected regardless of the thread count.

**Oversubscription** – with more threads than cores, spinlocks are slow by nature. The default thread counts include 2x the hardware thread count on purpose. The watchdog measures progress, not wall time, so slow is not reported as a deadlock.

**Compile-time knobs** – `-DLOCK_CACHE_LINE=N`, `-DLOCK_SPIN_LIMIT=N`, `-DLOCK_MUTATE_RELAXED_RELEASE` (tests only).

## Benchmark

`bench_mutex.cpp` runs the assignment's kernel (`acquire(); is_prime(N); release();` followed by `OUTSIDE_WORK` calls of `is_prime(N)`) for every selected lock and appends one CSV row per run.

```bash
make bench-quick PLATFORM=mylaptop     # a few minutes: sanity check of the setup
make bench PLATFORM=mylaptop           # full sweep -> bench_mylaptop.csv
make bench-omp PLATFORM=mylaptop       # + OpenMP baselines and the oversubscribed series
make analyze                           # summaries and figures from every bench_*.csv -> plots/
```

Without make: `g++ -std=c++17 -O2 -pthread bench_mutex.cpp -o bench_mutex && ./bench_mutex --help`.
`./bench_mutex --dry-run` prints the sweep and the number of runs without executing it.

**Contenders** (`--list`): own `tas`, `tas-pause`, `ttas`, `ttas-pause`, `ticket`, `ticket-prop`, `mcs`; baselines `std-mutex`, `pthread-mutex`, `pthread-spin` (Linux only; macOS has no `pthread_spin_lock`), and, when built with `-fopenmp -DUSE_OPENMP`, `omp-lock` and `omp-critical`. The slow variants (`*-sleep`, `*-yield`) are opt-in: `--locks=core,tas-sleep,ttas-sleep` or `--locks=all`. Everything is behind the same `acquire()`/`release()` interface (`omp-critical` has its own runner, because `#pragma omp critical` is a block, not a lock object).

**Metric.** The primary time is the whole-program execution time: from just before the first thread is created until the last join. It includes thread creation and the start-up skew, as a timed program would. Also recorded per run: `region_ms` (start gate to join, creation excluded) and the first/last per-thread finish times (their spread indicates unfairness). You can cross-check the primary figure with an external timer (`time ./bench_mutex --locks=mcs --threads=4 --outside=0 --cs-ns=1000 --reps=1 --warmup=0 --mode=fixed`, subtracting the start-up calibration).

**Two scaling modes** (`--mode`, default `both`):

| Mode | Work | Ideal behaviour | Reported |
|---|---|---|---|
| `fixed` | every thread runs `iters` iterations (the skeleton's `NUM_OF_ITERS`); total work grows with the threads | time stays flat | time, throughput (critical sections per microsecond) |
| `strong` | a fixed total, split across the threads | time falls as 1/threads | time, speed-up versus the same lock on 1 thread |

**Sweep.** Thread counts: powers of two up to the usable CPUs plus the CPU count itself (core series). `--oversub` adds 2x the usable CPUs as a secondary series, kept out of the headline plots. `OUTSIDE_WORK` defaults to 0, 1, 2, 4, 8, 16. The critical-section length is set by `--cs-ns` (default 100, 1000, 10000 ns): the program measures `is_prime` on the machine and picks the prime N that takes about that long, so platforms are compared at equal critical-section length (the CSV stores both N and the measured `cs_ns`). `--N=...` sets N directly. Iteration counts are sized so that one thread without contention runs about `--target-ms` (50 ms); in strong mode the total is that per-thread count times the usable CPUs.

**Protocol.** One warm-up pass, then 5 repetitions (`--reps`, `--warmup`). Locks are run in block-randomised order: each pass runs every lock once in a fresh random order (`--seed`), so drift in frequency or temperature is spread over the locks. `is_prime` reads N through a volatile reference so the compiler cannot fold or hoist the work. Every run is validated: completed iterations must equal the expected count and the summed `is_prime` results must equal `completed * (1 + OUTSIDE_WORK) * is_prime(N)`, otherwise the run is `invalid`. A run that exceeds `--max-run-ms` (20 s) is stopped through a flag checked once per iteration and recorded as `timeout`; both statuses are excluded from the statistics and reported at the end.

**Pinning.** `--pin` pins thread i to a CPU, physical cores first and SMT siblings last (Linux via `sched_getaffinity`/`pthread_setaffinity_np`, Windows via `SetThreadAffinityMask`; macOS has no affinity API, so the flag is ignored there with a warning). On hybrid P/E-core CPUs, note which cores were used. For `omp-critical`, set `OMP_PROC_BIND` / `OMP_PLACES` yourself.

**Output.** `bench_<platform>.csv` has one row per run (platform, CPU model, OS, arch, compiler, usable CPUs, pinning, series, mode, lock, threads, OUTSIDE_WORK, N, cs_ns, status, time_ms, region_ms, finish_min/max_ms, completed, throughput). Files from several machines can be passed to `analyze_bench.py` together; it writes `plots/bench_summary.csv` (median, min, sd, speed-up, ratio to `std-mutex`, finish spread) and figures per platform and N: fixed throughput, strong speed-up, time relative to `std-mutex` for both modes, and the oversubscribed series. Figure colour encodes the lock family and line style the variant (solid = plain, dashed = pause/proportional backoff or `omp-critical`, dotted = sleep, dash-dot = yield).

**Platform notes for benchmarking.** Close other applications, plug in power, disable battery saver, and let a laptop cool down between full sweeps. On Linux, `performance` governor (`cpupower frequency-set -g performance`) reduces noise. On Windows, use the High performance power plan; `sleep_for` has a 1-15 ms resolution there, so keep `*-sleep` out of the sweep or accept its cost. Run the same `make bench` on every machine; the number of platforms must be at least the group size.

## Limitations

- A lock must be released by the thread that acquired it (mutex semantics), as with `std::mutex`.
- MCS allocates one node per thread per concurrently held lock on first use (freed at thread exit).
- The mutation test shows the harness catches weakened release orderings under TSan and on ARM; on x86 without TSan such bugs may not show up natively.