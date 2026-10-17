# Makefile for Section 1 (Mutex Locks)
#
#   src/       all sources (headers, tests, benchmark, analysis scripts)
#   build/     compiler output (binaries)
#   results/   everything the programs write: CSV files and plots/
#
#   make                                           build test_correctness and bench_mutex into build/
#   make test | quick | control | tsan | mutate    correctness targets
#   make bench-quick | bench | bench-fair | analyze   benchmark targets (OMP=1 adds the OpenMP baselines)
#   make clean                                     remove build/ (results/ is kept; make distclean removes it too)

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread
LDFLAGS  ?= -pthread

SRC := src
BIN := build
RES := results

# Label stored in the CSV; override per machine: make bench PLATFORM=laptop-m2
PLATFORM ?= $(shell hostname)
# Extra options for the benchmark, e.g. BENCHFLAGS="--series=oversub --locks=tas,mcs"
BENCHFLAGS ?=

all: $(BIN)/test_correctness $(BIN)/bench_mutex

$(BIN) $(RES):
	mkdir -p $@

# ------------------------------------------------------------------ tests
$(BIN)/test_correctness: $(SRC)/test_correctness.cpp $(SRC)/lock.h | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

test: $(BIN)/test_correctness
	$(BIN)/test_correctness

quick: $(BIN)/test_correctness
	$(BIN)/test_correctness --quick

# Harness sensitivity check: lock_none MUST be detected as broken
control: $(BIN)/test_correctness
	$(BIN)/test_correctness --control

# ThreadSanitizer build. Slow by nature, and plain TAS (no backoff) livelocks under
# TSan with many threads (see README), so the thread count is capped; races are
# detected independently of how many threads run.
tsan: $(SRC)/test_correctness.cpp $(SRC)/lock.h | $(BIN)
	$(CXX) -std=c++17 -O1 -g -fsanitize=thread -pthread $< -o $(BIN)/test_tsan
	$(BIN)/test_tsan --quick --scale=0.1 --threads=1,2,4

# Mutation test: release stores weakened to relaxed. Failures / TSan races are EXPECTED.
mutate: $(SRC)/test_correctness.cpp $(SRC)/lock.h | $(BIN)
	$(CXX) -std=c++17 -O1 -g -fsanitize=thread -DLOCK_MUTATE_RELAXED_RELEASE -pthread $< -o $(BIN)/test_mutate
	-$(BIN)/test_mutate --quick --scale=0.1 --threads=2,4 --lock=tas,ttas,ticket,mcs

# -------------------------------------------------------------- benchmark
# The flags are recorded in every CSV row (-DBENCH_FLAGS).
$(BIN)/bench_mutex: $(SRC)/bench_mutex.cpp $(SRC)/lock.h | $(BIN)
	$(CXX) $(CXXFLAGS) -DBENCH_FLAGS='"$(CXXFLAGS)"' $< -o $@ $(LDFLAGS)

# Optional OpenMP baselines (omp-lock, omp-critical): add OMP=1 to any benchmark target, e.g.
#   make bench OMP=1   make bench-quick OMP=1   make bench-fair OMP=1
# On macOS use Homebrew LLVM / libomp: make bench OMP=1 CXX=$(brew --prefix llvm)/bin/clang++
$(BIN)/bench_mutex_omp: $(SRC)/bench_mutex.cpp $(SRC)/lock.h | $(BIN)
	$(CXX) $(CXXFLAGS) -fopenmp -DUSE_OPENMP -DBENCH_FLAGS='"$(CXXFLAGS) -fopenmp"' $< -o $@ $(LDFLAGS)

BENCH_BIN := $(BIN)/bench_mutex$(if $(OMP),_omp,)

# The benchmark runs five experiment series (scale, contention, cs, strong, oversub); see README.
# bench-quick: shorter run, all five series, 5 repetitions -> results/quick_<platform>.csv (not pooled with bench)
bench-quick: $(BENCH_BIN) | $(RES)
	$(BENCH_BIN) --pin --platform=$(PLATFORM) --csv=$(RES)/quick_$(PLATFORM).csv --reps=5 $(BENCHFLAGS)

# bench: the run used for the report, 10 repetitions -> results/bench_<platform>.csv
bench: $(BENCH_BIN) | $(RES)
	$(BENCH_BIN) --pin --platform=$(PLATFORM) --csv=$(RES)/bench_$(PLATFORM).csv --reps=10 $(BENCHFLAGS)

# Fairness experiment: fixed duration, per-thread counters (separate CSV, not mixed with timing data).
# With OMP=1 omp-lock is included (omp-critical is a block, not a lock object, so it has no fairness variant).
bench-fair: $(BENCH_BIN) | $(RES)
	$(BENCH_BIN) --mode=fairness --pin --platform=$(PLATFORM) --fair-csv=$(RES)/fair_$(PLATFORM).csv --reps=5 $(BENCHFLAGS)

# Summaries + figures from every results/bench_*.csv (and results/fair_*.csv) -> results/plots/
# (quick data: python3 src/analyze_bench.py results/quick_*.csv --out results/plots_quick)
analyze:
	@if ls $(RES)/bench_*.csv >/dev/null 2>&1; then python3 $(SRC)/analyze_bench.py $(RES)/bench_*.csv --out $(RES)/plots; else echo "no results/bench_*.csv yet: run make bench"; fi
	@if ls $(RES)/fair_*.csv >/dev/null 2>&1; then python3 $(SRC)/analyze_fairness.py $(RES)/fair_*.csv --out $(RES)/plots; fi

clean:
	rm -rf $(BIN)

distclean: clean
	rm -rf $(RES)

.PHONY: all test quick control tsan mutate bench-quick bench bench-fair analyze clean distclean