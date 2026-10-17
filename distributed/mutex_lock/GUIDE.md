# Distributed-memory versions (MPI-3 one-sided communication)

Status: **mutex locks** are implemented (`locks/`). Barriers and counters will follow.

Planned repository layout:

```
shared-memory/        (current top-level files will move here)
  locks/  barriers/  counters/
distributed-memory/
  locks/  barriers/  counters/    each with src/ build/ results/ Makefile README.md
```

## 1. Install MPI

Any MPI-3 implementation works (Open MPI or MPICH). Check with `mpicxx --version` and `mpirun --version`.

| Platform | Command |
|---|---|
| Ubuntu / Debian / WSL2 | `sudo apt install build-essential openmpi-bin libopenmpi-dev` (or `mpich`) |
| macOS (Homebrew) | `brew install open-mpi` (or `mpich`) |
| Fedora | `sudo dnf install openmpi openmpi-devel` then `module load mpi/openmpi-x86_64` |
| HPC cluster | `module avail mpi`, then `module load openmpi` (or the site MPI) |
| Windows | Use WSL2 (Ubuntu) and follow the Ubuntu row. |

## 2. Running several ranks on one machine

```
cd locks
make test NP=4                       # correctness
make bench-quick NP=4                # shorter run of all five series
make bench NP=8 RUNFLAGS="--oversubscribe"   # Open MPI: allow more ranks than cores
make bench-fair NP=4                 # fairness experiment (fixed duration, per-rank counters)
```

* Series 5 (oversubscription) needs `NP` of at least twice the logical CPU count of the node (ranks beyond that
  need `--oversubscribe`); with fewer ranks it is skipped and a note is printed.
* Open MPI refuses more ranks than cores unless `--oversubscribe`; MPICH does not need it.
* Pin ranks for stable numbers: Open MPI `--bind-to core`, MPICH `-bind-to core`.
* Ranks on one machine communicate through shared memory, so latency is far lower than across a network.
  Results validate the process model and algorithms, not network behaviour. Different machines
  need a hostfile (`mpirun --hostfile hosts -n 16 ...`) and the binary at the same path on each node.

## 3. Troubleshooting

* **Hang / "NO PROGRESS" message**: RMA progress is implementation specific. The locks poll with
  periodic `MPI_Iprobe` to drive progress. Try another MPI (Open MPI vs MPICH), fewer ranks, or run
  `--lock=<name>` to isolate the lock. Report the MPI version printed in the test header.
* **Compile errors about `MPI_Win_allocate`, `MPI_Compare_and_swap`, `MPI_INT64_T`**: the MPI is older than MPI-3.
* **`osc_ucx_component.c ... priority set inside component query failed`** (Open MPI 5, common on
  WSL2/laptops without a UCX transport): harmless noise, Open MPI falls back to another RMA component.
  Silence it, and avoid UCX entirely, with `make test RUNFLAGS="--mca osc ^ucx"` or
  `export OMPI_MCA_osc=^ucx` (MPICH ignores the variable). If RMA itself fails or hangs, try this first.