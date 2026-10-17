# Makefile for Section 1 (Mutex Locks)
#
#   make              build test_correctness and bench_mutex
#   make test | quick | control | tsan | mutate     correctness targets
#   make bench-quick | bench | bench-omp | analyze  benchmark targets
#   make clean

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread
LDFLAGS  ?= -pthread

# Label stored in the CSV; override per machine: make bench PLATFORM=laptop-m2
PLATFORM ?= $(shell hostname)

all: test_correctness bench_mutex

# ------------------------------------------------------------------ tests
test_correctness: test_correctness.cpp lock.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

test: test_correctness
	./test_correctness

quick: test_correctness
	./test_correctness --quick

# Harness sensitivity check: lock_none MUST be detected as broken
control: test_correctness
	./test_correctness --control

# ThreadSanitizer build. Slow by nature, and plain TAS (no backoff) livelocks under
# TSan with many threads (see README), so the thread count is capped; races are
# detected independently of how many threads run.
tsan: test_correctness.cpp lock.h
	$(CXX) -std=c++17 -O1 -g -fsanitize=thread -pthread $< -o test_tsan
	./test_tsan --quick --scale=0.1 --threads=1,2,4

# Mutation test: release stores weakened to relaxed. Failures / TSan races are EXPECTED.
mutate: test_correctness.cpp lock.h
	$(CXX) -std=c++17 -O1 -g -fsanitize=thread -DLOCK_MUTATE_RELAXED_RELEASE -pthread $< -o test_mutate
	-./test_mutate --quick --scale=0.1 --threads=2,4 --lock=tas,ttas,ticket,mcs

# -------------------------------------------------------------- benchmark
bench_mutex: bench_mutex.cpp lock.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Optional OpenMP baselines (omp_lock_t, omp critical). On macOS use Homebrew
# LLVM / libomp: make bench_mutex_omp CXX=$(brew --prefix llvm)/bin/clang++
bench_mutex_omp: bench_mutex.cpp lock.h
	$(CXX) $(CXXFLAGS) -fopenmp -DUSE_OPENMP $< -o $@ $(LDFLAGS)

# A few minutes: small sweep to check the setup
bench-quick: bench_mutex
	./bench_mutex --quick --pin --platform=$(PLATFORM) --csv=bench_$(PLATFORM).csv

# Report-sized sweep: ~1/6 of the full run, keeps the dimensions the report needs
# (low/high contention x short/long critical sections, both scaling modes)
bench-report: bench_mutex
	./bench_mutex --pin --platform=$(PLATFORM) --csv=bench_$(PLATFORM).csv \
	    --outside=0,4,16 --cs-ns=100,10000 --reps=3 --target-ms=30

# Full sweep (core locks, both scaling modes), appended to bench_<platform>.csv
bench: bench_mutex
	./bench_mutex --pin --platform=$(PLATFORM) --csv=bench_$(PLATFORM).csv

# Same plus the OpenMP baselines and the oversubscribed secondary series
bench-omp: bench_mutex_omp
	./bench_mutex_omp --pin --oversub --platform=$(PLATFORM) --csv=bench_$(PLATFORM).csv

# Summaries + figures from every bench_*.csv in this directory
analyze:
	python3 analyze_bench.py bench_*.csv --out plots

clean:
	rm -f test_correctness test_tsan test_mutate bench_mutex bench_mutex_omp

.PHONY: all test quick control tsan mutate bench-quick bench bench-omp analyze clean