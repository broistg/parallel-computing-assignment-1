# Distributed-memory (MPI-3 RMA) mutex locks
#
#   src/       all sources (header, tests, benchmark, analysis script)
#   build/     compiler output (binaries)
#   results/   everything the programs write: CSV files and plots/
#
#   make test            correctness suite (NP ranks)
#   make test-quick      fast smoke test of the correctness suite
#   make test-control    negative control (harness must detect a missing lock)
#   make bench-quick     all series, 5 trials (shorter run)                -> results/quick_mpi.csv
#   make bench           all series, 10 trials (used for the report)       -> results/bench_mpi.csv
#   make bench-fair      fairness experiment                               -> results/fair_mpi.csv
#   make bench-analyze   tables / plots from results/bench_mpi.csv (and results/fair_mpi.csv if present)
#   make clean           remove build/ (results/ is kept; make distclean removes it too)
# Series: scale, contention, cs, strong, oversub (see README). 'oversub' needs NP >= 2 x logical CPUs
# on ONE node (and RUNFLAGS="--oversubscribe" for Open MPI); otherwise it is skipped with a note.
# Variables: MPICXX, MPIRUN, NP, RUNFLAGS (e.g. RUNFLAGS="--oversubscribe --bind-to core"),
#            BENCHFLAGS (extra options for the benchmark, e.g. BENCHFLAGS="--trials=5 --series=scale,oversub")
MPICXX     ?= mpicxx
MPIRUN     ?= mpirun
NP         ?= 4
RUNFLAGS   ?=
BENCHFLAGS ?=
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra

SRC := src
BIN := build
RES := results

all: $(BIN)/test_correctness_mpi $(BIN)/bench_mpi_mutex

$(BIN) $(RES):
	mkdir -p $@

$(BIN)/test_correctness_mpi: $(SRC)/test_correctness_mpi.cpp $(SRC)/mpi_lock.h | $(BIN)
	$(MPICXX) $(CXXFLAGS) $< -o $@

# The flags are recorded in every CSV row (-DBENCH_FLAGS).
$(BIN)/bench_mpi_mutex: $(SRC)/bench_mpi_mutex.cpp $(SRC)/mpi_lock.h | $(BIN)
	$(MPICXX) $(CXXFLAGS) -DBENCH_FLAGS='"$(CXXFLAGS)"' $< -o $@

test: $(BIN)/test_correctness_mpi
	$(MPIRUN) $(RUNFLAGS) -n $(NP) $(BIN)/test_correctness_mpi

test-quick: $(BIN)/test_correctness_mpi
	$(MPIRUN) $(RUNFLAGS) -n $(NP) $(BIN)/test_correctness_mpi --quick

test-control: $(BIN)/test_correctness_mpi
	$(MPIRUN) $(RUNFLAGS) -n $(NP) $(BIN)/test_correctness_mpi --control

bench-quick: $(BIN)/bench_mpi_mutex | $(RES)
	$(MPIRUN) $(RUNFLAGS) -n $(NP) $(BIN)/bench_mpi_mutex --trials=5 $(BENCHFLAGS) > $(RES)/quick_mpi.csv
	@echo "wrote $(RES)/quick_mpi.csv"

bench: $(BIN)/bench_mpi_mutex | $(RES)
	$(MPIRUN) $(RUNFLAGS) -n $(NP) $(BIN)/bench_mpi_mutex $(BENCHFLAGS) > $(RES)/bench_mpi.csv
	@echo "wrote $(RES)/bench_mpi.csv"

bench-fair: $(BIN)/bench_mpi_mutex | $(RES)
	$(MPIRUN) $(RUNFLAGS) -n $(NP) $(BIN)/bench_mpi_mutex --mode=fairness $(BENCHFLAGS) > $(RES)/fair_mpi.csv
	@echo "wrote $(RES)/fair_mpi.csv"

bench-analyze:
	@if [ -f $(RES)/bench_mpi.csv ]; then python3 $(SRC)/analyze_mpi.py $(RES)/bench_mpi.csv --out $(RES)/plots; fi
	@if [ -f $(RES)/fair_mpi.csv ]; then python3 $(SRC)/analyze_mpi.py $(RES)/fair_mpi.csv --out $(RES)/plots; fi

clean:
	rm -rf $(BIN)

distclean: clean
	rm -rf $(RES)

.PHONY: all test test-quick test-control bench-quick bench bench-fair bench-analyze clean distclean