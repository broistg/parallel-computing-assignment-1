# Mutex locks over MPI-3 RMA

Files: `mpi_lock.h` (locks), `test_correctness_mpi.cpp`, `bench_mpi_mutex.cpp`, `analyze_mpi.py`, `Makefile`. MPI installation and one-machine/multi-machine notes: see `../README.md`.

## Design
* A lock's state lives in an RMA window on a *home* rank (default 0), accessed through passive-target synchronisation (`MPI_Win_lock_all(MPI_MODE_NOCHECK)` for the lock's lifetime).
* Atomic operations: `MPI_Compare_and_swap`, `MPI_Fetch_and_op`, atomic stores via `MPI_Accumulate(MPI_REPLACE)`; every remote operation is followed by `MPI_Win_flush`.
* Locks are created collectively (`lock_xxx lock(comm)`), same interface as the shared-memory ones: `acquire/release`, `try_lock` (not for `winlock`), RAII `Guard<L>`.

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
make quick NP=4
make control NP=4      # negative control: lock_none MUST be detected as broken
./test_correctness_mpi --help
make bench NP=8        # CSV (results_mpi.csv); make analyze for tables/plots
```
Benchmark: fixed rounds per rank, remote counter in the critical section, whole-region timer (max over ranks), counter validated before a row is written, block-randomised lock order.

## Known limitations
* Developed without access to an MPI installation: logic was validated only on a thread-based emulation of the MPI calls (not included). First runs on real MPI may expose signature or progress issues.
* Correctness suites are probabilistic stress tests, not proofs.
* More ranks than cores makes spin locks slow; the FIFO fairness verdict is informational then.
