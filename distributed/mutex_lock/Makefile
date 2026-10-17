# Distributed-memory (MPI-3 RMA) mutex locks
#   make test            correctness suite (NP ranks)
#   make quick           fast smoke test
#   make control         negative control (harness must detect a missing lock)
#   make bench-quick     short benchmark -> results_mpi_quick.csv
#   make bench           full benchmark  -> results_mpi.csv
#   make analyze         plots/summary from results_mpi.csv
# Variables: MPICXX, MPIRUN, NP, RUNFLAGS (e.g. RUNFLAGS="--oversubscribe" or "--bind-to core")
MPICXX   ?= mpicxx
MPIRUN   ?= mpirun
NP       ?= 4
RUNFLAGS ?=
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

all: test_correctness_mpi bench_mpi_mutex

test_correctness_mpi: test_correctness_mpi.cpp mpi_lock.h
	$(MPICXX) $(CXXFLAGS) $< -o $@

bench_mpi_mutex: bench_mpi_mutex.cpp mpi_lock.h
	$(MPICXX) $(CXXFLAGS) $< -o $@

test: test_correctness_mpi
	$(MPIRUN) $(RUNFLAGS) -n $(NP) ./test_correctness_mpi

quick: test_correctness_mpi
	$(MPIRUN) $(RUNFLAGS) -n $(NP) ./test_correctness_mpi --quick

control: test_correctness_mpi
	$(MPIRUN) $(RUNFLAGS) -n $(NP) ./test_correctness_mpi --control

bench-quick: bench_mpi_mutex
	$(MPIRUN) $(RUNFLAGS) -n $(NP) ./bench_mpi_mutex --quick > results_mpi_quick.csv
	@echo "wrote results_mpi_quick.csv"

bench: bench_mpi_mutex
	$(MPIRUN) $(RUNFLAGS) -n $(NP) ./bench_mpi_mutex > results_mpi.csv
	@echo "wrote results_mpi.csv"

analyze:
	python3 analyze_mpi.py results_mpi.csv

clean:
	rm -f test_correctness_mpi bench_mpi_mutex results_mpi*.csv

.PHONY: all test quick control bench-quick bench analyze clean
