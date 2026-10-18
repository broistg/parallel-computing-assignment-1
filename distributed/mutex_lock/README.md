# Mutex locks over MPI-3 RMA

Files: `src/mpi_lock.h` (locks), `src/test_correctness_mpi.cpp`, `src/bench_mpi_mutex.cpp` (timing and fairness), `src/analyze_mpi.py`, `Makefile`.
Binaries go to `build/`, CSV files and plots to `results/` (`make clean` removes only `build/`).
MPI installation and one-machine/multi-machine notes: see `../README.md`.

## Design
* A lock's state lives in an RMA window on a *home* rank (default 0), accessed through passive-target
  synchronisation (`MPI_Win_lock_all(MPI_MODE_NOCHECK)` for the lock's lifetime).
* Atomic operations: `MPI_Compare_and_swap`, `MPI_Fetch_and_op`, atomic stores via
  `MPI_Accumulate(MPI_REPLACE)`; every remote operation is followed by `MPI_Win_flush`.
* Locks are created collectively (`lock_xxx lock(comm)`), same interface as the shared-memory ones:
  `acquire/release`, `try_lock` (not for `winlock`), RAII `Guard<L>`.

| Name | Algorithm |
|---|---|
| `tas`, `tas-pause`, `tas-sleep` | test-and-set (CAS) with no / pause / sleep exponential backoff |
| `ttas`, `ttas-*` | test-and-test-and-set (remote read before CAS) |
| `ticket`, `ticket-prop` | ticket lock (FIFO), plain / proportional backoff |
| `mcs` | MCS queue lock (FIFO), each rank spins on its **own** window slot (local polling) |
| `winlock` | baseline: `MPI_Win_lock(MPI_LOCK_EXCLUSIVE)` provided by the MPI library |

Data protected in the tests lives on rank 0 and is accessed with non-atomic `MPI_Get/MPI_Put`
inside the critical section (completed before release), so a broken lock produces visible corruption.

## Usage
```
make test NP=4         # full suite, rank counts 1,2,4 (sub-communicators)
make test-quick NP=4
make test-control NP=4      # negative control: lock_none MUST be detected as broken
mpirun -n 2 build/test_correctness_mpi --help
make bench-quick NP=8  # shorter run: all series, 5 trials -> results/quick_mpi.csv
make bench NP=8        # the run for the report: all series, 10 trials -> results/bench_mpi.csv
make bench-fair NP=8   # fairness experiment -> results/fair_mpi.csv
make bench-analyze           # tables / plots per series (median, IQR) into results/plots/
```
Extra options go through `BENCHFLAGS`, e.g. `make bench NP=8 BENCHFLAGS="--trials=5 --series=scale,oversub"`.

## Benchmark
* **Fixed workload.** As in the shared-memory benchmark, the workload is the same on every platform: rounds per rank
  (`--ops`), busy work (`--cs-work`), trials and warm-up are fixed defaults and are never calibrated per machine.
  Every CSV row carries `config_id`, a fingerprint of these values, the duration (fairness) and the lock parameters;
  `analyze_mpi.py` warns when ids differ between files. The strong-scaling total is `ops x R_log`, where R_log is the
  rank count of the machine, so it differs between machines with a different logical-CPU count (pass `--ops` to
  keep it equal when that matters).
* **Series** (`--series=scale,contention,cs,strong,oversub|all`; the same five as the shared-memory
  benchmark). R_log = min(NP, logical CPUs) on one node, R_phys = physical cores. One unit of OUTSIDE_WORK k is
  `k x` the medium critical-section work (50 busy iterations by default, not calibrated: RMA round trips dominate).
  | Series | Mode | Ranks | OUTSIDE_WORK k | Critical section |
  |---|---|---|---|---|
  | `scale` | fixed rounds per rank | powers of 2 up to R_log, plus R_phys | 0, 2, 8 | medium |
  | `contention` | fixed per rank | 1, R_phys, R_log | 0, 1, 2, 4, 8, 16 | medium |
  | `cs` | fixed per rank | 1, R_phys, R_log | 0, 8 | 0, medium, 100 x medium |
  | `strong` | fixed **total** rounds (ops x R_log) split over the ranks | as `scale` | 2 | medium |
  | `oversub` | fixed per rank | R_log and 2 x R_log | 0, 8 | medium |

  Configurations that several series share run once and carry all their tags in the `experiment` column.
  `oversub` needs `-n >= 2 x R_log` on a single node (Open MPI also `--oversubscribe`), otherwise it is skipped
  with a printed note. For series 5 report throughput **and** whole-run time per lock: with fixed work per
  rank the ideal time ratio is 2 and the ideal throughput ratio is 1 (`analyze_mpi.py` writes
  `results/plots/mpi_oversub_table.csv`).
* **Timing mode:** fixed rounds per rank, remote counter in the critical section, whole-region timer (max over
  ranks), counter validated before a row is written. Per configuration: 1 discarded warm-up pass, then 10
  trials (`--trials`, `--warmup`); lock order is block-randomised in every pass. Reported: median and IQR
  (minimum and standard deviation as supplementary statistics on stderr).
* **Fairness mode** (`--mode=fairness`): all ranks compete for a fixed time (`--duration-ms`, default 1000) on
  at least 2 ranks. Every rank keeps a private acquisition counter and times every 32nd `acquire()` into a
  private log; counts and logs are gathered only after the run, so the measurement adds no communication.
  Reported: acquisitions per rank (min, median, max), max/min ratio, Jain's index, `acquire()` latency p50/p99/max
  and the spread of per-rank p99. FIFO order is not equal throughput, and the numbers describe the observed
  behaviour, not a guarantee.
* **Counting:** per-rank counters only; the single shared word is the protected counter that validates the run.
* **Metadata** (stderr header and every CSV row): MPI library and version, ranks, nodes and ranks per node,
  CPU model, logical and physical cores, kernel, compiler, compile flags (`-O2`, recorded via `-DBENCH_FLAGS`),
  how many CPUs each rank may use (1 = bound to a core), RMA memory model, and `lock_params`.
* **Lock parameters** (from the lock types): pause backoff 4..1024 `cpu_relax` (doubling, jitter in [d/2, d],
  yield once saturated); sleep backoff `sleep_for(nanoseconds(d))`, d from 1000 to 128000 ns (doubling);
  `ticket-prop` waits 1000 ns per position in line (cap 100000 ns); waiting loops use SpinWait (2048 polls, then
  yield) with an MPI progress poll every 64 polls. The state of every lock is in an RMA window on rank 0.
* Use the same compiler and optimisation for all variants (`-O2`, never a sanitizer, for timing). Bind ranks to
  cores with the launcher (`RUNFLAGS="--bind-to core"` for Open MPI) so the affinity column shows 1.
* On one machine the ranks communicate through shared memory; results validate the process model, not
  network behaviour. CSV files from an older version have different columns, so use new file names.

## Known limitations
* Developed without access to an MPI installation: logic was validated only on a thread-based emulation
  of the MPI calls (not included). First runs on real MPI may expose signature or progress issues.
* Correctness suites are probabilistic stress tests, not proofs.
* More ranks than cores makes spin locks slow; the FIFO fairness verdict is informational then.