// ============================================================================
// test_correctness_mpi.cpp -- correctness & stress harness for mpi_lock.h
//
//   Build : mpicxx -std=c++17 -O2 test_correctness_mpi.cpp -o test_correctness_mpi
//   Run   : mpirun -n 4 ./test_correctness_mpi [--help]
//
// Port of the shared-memory harness to the MPI process model. Data protected by
// a lock lives in an RMA window on rank 0 and is accessed with NON-atomic
// MPI_Get / MPI_Put inside the critical section (completed before release), so a
// broken lock shows up as lost updates, torn records or overlapping holders.
//
//   [1] mutual exclusion (remote in-critical-section counter must be 0 on entry)
//   [2] protected read-modify-write counter (lost updates)
//   [3] multi-field record (consistency across consecutive holders)
//   [4] fairness distribution + FIFO order (acquisition log)
//   [5] stress profiles (contention levels, holder yields while holding)
//   [6] API: try_lock, Guard, mixed styles
//   [7] several lock objects, nesting, non-LIFO release
//   [8] backoff policy unit checks
// Ranks are swept 1, 2, 4, ... up to the world size on sub-communicators. A
// per-rank watchdog thread (no MPI calls) terminates the job if the rank makes no
// progress for --stall-ms.
// ============================================================================

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mpi_lock.h"

using namespace dlock;

// ============================================================
// Configuration
// ============================================================

constexpr int EXCL_ITERATIONS    = 4'000;
constexpr int COUNTER_ITERATIONS = 3'000;
constexpr int RECORD_ITERATIONS  = 1'500;
constexpr int STRESS_ITERATIONS  = 2'500;
constexpr int FAIRNESS_ROUNDS    = 6'000;
constexpr int API_ITERATIONS     = 900;
constexpr int MULTI_ITERATIONS   = 900;

struct Options {
    int trials = 2;
    double scale = 1.0;
    int stall_ms = 8'000;
    int cpus = 0;                    // cores available to the job (0 = this node's hardware threads)
    std::vector<int> ranks;          // empty -> powers of two up to the world size
    std::vector<std::string> locks;  // empty -> all
    bool control = false;
    bool list = false;
    bool help = false;
};

// ============================================================
// Per-rank context and watchdog
// ============================================================

// Counts progress; a thread that never calls MPI watches it. Created per rank
// (no mutable globals, so ranks that are threads in an emulator stay independent).
class Monitor {
public:
    Monitor(int rank, int stall_ms) : rank_(rank), stall_ms_(stall_ms), th_([this] { loop(); }) {}
    ~Monitor() {
        stop_.store(true);
        th_.join();
    }
    void tick() { ticks_.store(ticks_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed); }
    void set_active(bool a) { active_.store(a); }
    void set_label(const std::string& s) {
        std::lock_guard<std::mutex> g(m_);
        label_ = s;
    }

private:
    void loop() {
        using clock = std::chrono::steady_clock;
        std::uint64_t last = ticks_.load();
        auto last_change = clock::now();
        while (!stop_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            const auto now = clock::now();
            const std::uint64_t cur = ticks_.load();
            if (!active_.load() || cur != last) {
                last = cur;
                last_change = now;
                continue;
            }
            const auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_change).count();
            if (idle > stall_ms_) {
                std::string label;
                {
                    std::lock_guard<std::mutex> g(m_);
                    label = label_;
                }
                std::fprintf(stderr,
                             "\n>>> rank %d: NO PROGRESS for %d ms (deadlock/livelock or hung RMA) in: %s\n"
                             "    Terminating the job (stuck ranks cannot be cancelled). Re-run with --lock=<name> "
                             "for the other locks; see README (progress / hangs).\n",
                             rank_, stall_ms_, label.c_str());
                std::fflush(stderr);
                std::_Exit(2);  // the launcher terminates the remaining ranks
            }
        }
    }

    int rank_, stall_ms_;
    std::atomic<std::uint64_t> ticks_{0};
    std::atomic<bool> active_{false};
    std::atomic<bool> stop_{false};
    std::mutex m_;
    std::string label_;
    std::thread th_;  // last member: starts after everything above is initialised
};

// Marks the watched part of a test (loops that must make progress).
class Work {
public:
    explicit Work(Monitor& m) : m_(m) { m_.set_active(true); }
    ~Work() { m_.set_active(false); }

private:
    Monitor& m_;
};

struct Env {
    Monitor* mon = nullptr;
    int world_rank = 0, world_size = 1;
    int cpus = 1;
    Options opt;
};

// ============================================================
// Utilities
// ============================================================

inline std::uint64_t& rng_state() {
    static thread_local std::uint64_t s = 0x853c49e6748fea9bULL;
    return s;
}
inline void seed_rng(int rank) {
    rng_state() = 0x853c49e6748fea9bULL ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(rank + 1));
}
inline std::uint64_t fast_rand() {
    std::uint64_t& s = rng_state();
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

inline void busy_work(int iterations) {
    static thread_local std::uint64_t local_sink = 0;
    std::uint64_t x = 0x123456789abcdefULL ^ local_sink;
    for (int i = 0; i < iterations; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    local_sink = x;
}
inline void randomized_busy_work(int lo, int hi) {
    if (hi <= lo) {
        busy_work(lo);
        return;
    }
    busy_work(lo + static_cast<int>(fast_rand() % static_cast<std::uint64_t>(hi - lo + 1)));
}

struct TestResult {
    bool passed;
    std::string message;
};
inline TestResult fail(std::string m) { return {false, std::move(m)}; }
inline TestResult ok(std::string m) { return {true, std::move(m)}; }

inline std::int64_t allsum(MPI_Comm c, std::int64_t v) {
    std::int64_t r = 0;
    MPI_Allreduce(&v, &r, 1, MPI_INT64_T, MPI_SUM, c);
    return r;
}
inline std::int64_t allmax(MPI_Comm c, std::int64_t v) {
    std::int64_t r = 0;
    MPI_Allreduce(&v, &r, 1, MPI_INT64_T, MPI_MAX, c);
    return r;
}

struct CommInfo {
    int me, n;
    explicit CommInfo(MPI_Comm c) {
        MPI_Comm_rank(c, &me);
        MPI_Comm_size(c, &n);
    }
};

// ============================================================
// Negative-control lock (does nothing)
// ============================================================

class lock_none {
public:
    static constexpr bool is_fifo = false;
    static constexpr bool has_try_lock = true;
    explicit lock_none(MPI_Comm, int = 0) {}
    void acquire() {}
    void release() {}
    bool try_lock() { return true; }
};

// ============================================================
// TEST 1 + 5: mutual exclusion / stress
// ============================================================

template <class L>
TestResult test_exclusion(Env& env, MPI_Comm c, int iters, int in_lo, int in_hi, int out_work,
                          int yield_pct) {
    CommInfo ci(c);
    Window data(c, 1);  // slot 0 on rank 0: number of ranks inside the critical section
    L lock(c);
    std::int64_t violations = 0;
    {
        Work work(*env.mon);
        for (int i = 0; i < iters; ++i) {
            if (out_work > 0) randomized_busy_work(out_work / 2, out_work * 3 / 2);
            lock.acquire();
            if (data.fetch_add(0, 0, 1) != 0) ++violations;
            if (in_hi > 0) randomized_busy_work(in_lo, in_hi);
            if (yield_pct > 0 && static_cast<int>(fast_rand() % 100) < yield_pct) std::this_thread::yield();
            data.fetch_add(0, 0, -1);
            lock.release();
            env.mon->tick();
        }
    }
    MPI_Barrier(c);
    const std::int64_t total_v = allsum(c, violations);
    const std::int64_t final_in = data.read(0, 0);
    if (total_v == 0 && final_in == 0) return ok("mutual exclusion preserved");
    return fail("violations=" + std::to_string(total_v) + ", final in-CS count=" + std::to_string(final_in));
}

// ============================================================
// TEST 2: protected read-modify-write counter
// ============================================================

template <class L>
TestResult test_counter(Env& env, MPI_Comm c, int iters) {
    CommInfo ci(c);
    Window data(c, 1);
    L lock(c);
    {
        Work work(*env.mon);
        for (int i = 0; i < iters; ++i) {
            lock.acquire();
            const std::int64_t v = data.get(0, 0);  // non-atomic read ...
            randomized_busy_work(5, 20);
            data.put(0, 0, v + 1);                  // ... modify, write
            lock.release();
            env.mon->tick();
        }
    }
    MPI_Barrier(c);
    const std::int64_t final_v = data.get(0, 0);
    const std::int64_t expected = static_cast<std::int64_t>(ci.n) * iters;
    if (final_v == expected) return ok("counter=" + std::to_string(final_v));
    return fail("counter=" + std::to_string(final_v) + ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 3: multi-field record
// ============================================================

template <class L>
TestResult test_record(Env& env, MPI_Comm c, int iters) {
    constexpr std::int64_t K = 1000000007LL;
    enum { SEQ, A, B, CK };
    CommInfo ci(c);
    Window rec(c, 4);
    L lock(c);
    std::int64_t failures = 0;
    {
        Work work(*env.mon);
        for (int i = 0; i < iters; ++i) {
            lock.acquire();
            const std::int64_t seq = rec.get(0, SEQ);
            const std::int64_t a = rec.get(0, A);
            const std::int64_t b = rec.get(0, B);
            const std::int64_t ck = rec.get(0, CK);
            if (seq > 0 && !(a == seq * K && b == ~a && ck == (a ^ b ^ seq))) ++failures;

            // Update field by field, with gaps in between.
            const std::int64_t ns = seq + 1;
            rec.put(0, SEQ, ns);
            rec.put(0, A, ns * K);
            busy_work(2);
            rec.put(0, B, ~(ns * K));
            busy_work(2);
            rec.put(0, CK, (ns * K) ^ ~(ns * K) ^ ns);
            lock.release();
            env.mon->tick();
        }
    }
    MPI_Barrier(c);
    const std::int64_t total_f = allsum(c, failures);
    const std::int64_t seq = rec.get(0, SEQ);
    const std::int64_t expected = static_cast<std::int64_t>(ci.n) * iters;
    if (total_f == 0 && seq == expected) return ok("record consistent across handovers");
    return fail("inconsistent reads=" + std::to_string(total_f) + ", final_seq=" + std::to_string(seq) +
                ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 4: fairness and FIFO order
// A global sequence number is drawn inside the critical section and the
// acquiring rank is logged at that index. For a FIFO lock the ranks must
// alternate roughly evenly once all ranks are competing; the number of
// acquisitions by others between "rank read the sequence" and "rank got the
// lock" is a soft overtaking indicator (reported, never failing).
// ============================================================

template <class L>
TestResult test_fairness(Env& env, MPI_Comm c, int total_rounds) {
    CommInfo ci(c);
    if (ci.n <= 1) return ok("single rank (trivially fair)");

    Window ctl(c, 1);
    Window log(c, total_rounds, {}, 1, -1);  // contiguous log, initialised to -1
    L lock(c);
    std::int64_t max_over = 0;
    {
        Work work(*env.mon);
        for (;;) {
            const std::int64_t seen = ctl.read(0, 0);
            if (seen >= total_rounds) break;
            lock.acquire();
            const std::int64_t s = ctl.fetch_add(0, 0, 1);
            if (s >= total_rounds) {
                lock.release();
                break;
            }
            log.put(0, static_cast<int>(s), ci.me);
            max_over = std::max(max_over, s - seen);
            randomized_busy_work(5, 15);
            lock.release();
            env.mon->tick();
            randomized_busy_work(2, 8);
        }
    }
    MPI_Barrier(c);
    const std::int64_t over_max = allmax(c, max_over);

    int verdict = 1;
    std::string msg;
    if (ci.me == 0) {
        std::vector<std::int64_t> order(total_rounds);
        log.get_range(0, 0, order.data(), total_rounds);
        std::vector<std::uint64_t> counts(ci.n, 0);
        bool complete = true;
        for (auto r : order) {
            if (r < 0 || r >= ci.n) complete = false;
            else counts[static_cast<size_t>(r)]++;
        }
        const std::uint64_t mn = *std::min_element(counts.begin(), counts.end());
        const std::uint64_t mx = *std::max_element(counts.begin(), counts.end());
        const double ratio = (mn == 0) ? INFINITY : static_cast<double>(mx) / static_cast<double>(mn);
        std::ostringstream st;
        st << "min=" << mn << ", max=" << mx << ", ratio=" << std::fixed << std::setprecision(2) << ratio
           << ", overtake_max=" << over_max;

        if (!complete) {
            verdict = 0;
            msg = "acquisition log incomplete (lost or duplicate sequence numbers): " + st.str();
        } else if (!L::is_fifo) {
            msg = "INFORMATIONAL (non-FIFO lock; " + st.str() + ")";
        } else if (ci.n > env.cpus) {
            msg = "INFORMATIONAL (more ranks than cores, scheduler dominates; " + st.str() + ")";
        } else {
            // Window in which every rank has appeared at least once.
            std::vector<char> seen_rank(ci.n, 0);
            int distinct = 0;
            std::uint64_t start = 0;
            for (std::uint64_t i = 0; i < order.size() && distinct < ci.n; ++i) {
                const auto r = static_cast<size_t>(order[i]);
                if (!seen_rank[r]) {
                    seen_rank[r] = 1;
                    ++distinct;
                }
                start = i + 1;
            }
            if (distinct < ci.n || order.size() - start < static_cast<std::uint64_t>(20 * ci.n)) {
                msg = "INFORMATIONAL (ranks did not overlap long enough for a verdict; " + st.str() + ")";
            } else {
                std::vector<std::uint64_t> wc(ci.n, 0);
                for (std::uint64_t i = start; i < order.size(); ++i) wc[static_cast<size_t>(order[i])]++;
                const std::uint64_t wmn = *std::min_element(wc.begin(), wc.end());
                const std::uint64_t wmx = *std::max_element(wc.begin(), wc.end());
                const double wr = (wmn == 0) ? INFINITY : static_cast<double>(wmx) / static_cast<double>(wmn);
                st << ", window[min=" << wmn << ", max=" << wmx << ", ratio=" << wr << "]";
                if (wmn == 0 || wr > 3.0) {
                    verdict = 0;
                    msg = "FIFO LOCK UNFAIR/STARVING: " + st.str();
                } else {
                    msg = "FAIR (" + st.str() + ")";
                    if (over_max > static_cast<std::int64_t>(16 * ci.n))
                        msg += " [WARN: overtaking above soft bound 16n]";
                }
            }
        }
    }
    MPI_Bcast(&verdict, 1, MPI_INT, 0, c);
    return {verdict != 0, msg};
}

// ============================================================
// TEST 6: API (try_lock, Guard, mixed styles)
// ============================================================

template <class L>
TestResult test_api(Env& env, MPI_Comm c, int iters) {
    CommInfo ci(c);
    bool good = true;
    std::string why;
    auto check = [&](bool cond, const char* m) {
        if (!cond && good) {
            good = false;
            why = m;
        }
    };

    if constexpr (L::has_try_lock) {
        L lock(c);
        if (ci.me == 0) check(lock.try_lock(), "try_lock() failed on a free lock");
        MPI_Barrier(c);
        if (ci.me != 0) {
            const bool got = lock.try_lock();
            if (got) lock.release();
            check(!got, "try_lock() succeeded on a held lock");
        }
        MPI_Barrier(c);
        if (ci.me == 0) lock.release();
        MPI_Barrier(c);
        if (ci.me == 0) {
            { Guard<L> g(lock); }
            const bool again = lock.try_lock();
            check(again, "lock not released by Guard");
            if (again) lock.release();
        }
        MPI_Barrier(c);
    }

    Window data(c, 1);
    L lock(c);
    {
        Work work(*env.mon);
        for (int i = 0; i < iters; ++i) {
            switch (i % 3) {
                case 0: {
                    Guard<L> g(lock);
                    data.put(0, 0, data.get(0, 0) + 1);
                    break;
                }
                case 1: {
                    if constexpr (L::has_try_lock) {
                        while (!lock.try_lock()) std::this_thread::yield();
                    } else {
                        lock.acquire();
                    }
                    data.put(0, 0, data.get(0, 0) + 1);
                    lock.release();
                    break;
                }
                default:
                    lock.acquire();
                    data.put(0, 0, data.get(0, 0) + 1);
                    lock.release();
                    break;
            }
            env.mon->tick();
        }
    }
    MPI_Barrier(c);
    const std::int64_t v = data.get(0, 0);
    const std::int64_t expected = static_cast<std::int64_t>(ci.n) * iters;
    check(v == expected, "mixed-style counter wrong");
    if (good) return ok("API semantics + mixed-style counter OK");
    return fail(why + " (counter=" + std::to_string(v) + ", expected=" + std::to_string(expected) + ")");
}

// ============================================================
// TEST 7: several locks, nesting, non-LIFO release
// ============================================================

template <class L>
TestResult test_multi_lock(Env& env, MPI_Comm c, int iters) {
    CommInfo ci(c);
    Window data(c, 2);
    L A(c), B(c);
    auto bump = [&](int slot) { data.put(0, slot, data.get(0, slot) + 1); };
    {
        Work work(*env.mon);
        for (int i = 0; i < iters; ++i) {
            switch (i % 3) {
                case 0:  // sequential use of two locks
                    A.acquire();
                    bump(0);
                    A.release();
                    B.acquire();
                    bump(1);
                    B.release();
                    break;
                case 1:  // nested, LIFO release
                    A.acquire();
                    B.acquire();
                    bump(0);
                    bump(1);
                    B.release();
                    A.release();
                    break;
                default:  // hand-over-hand: release order differs from acquire order
                    A.acquire();
                    bump(0);
                    B.acquire();
                    A.release();
                    bump(1);
                    B.release();
                    break;
            }
            env.mon->tick();
        }
    }
    MPI_Barrier(c);
    const std::int64_t ca = data.get(0, 0), cb = data.get(0, 1);
    const std::int64_t expected = static_cast<std::int64_t>(ci.n) * iters;
    if (ca == expected && cb == expected) return ok("multi-lock/nesting OK");
    return fail("ca=" + std::to_string(ca) + ", cb=" + std::to_string(cb) + ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 8: backoff policy unit checks
// ============================================================

static bool test_backoff_policies(std::string& why) {
    static_assert(PauseBackoff<4, 64>::next_delay(4) == 8, "double");
    static_assert(PauseBackoff<4, 64>::next_delay(64) == 64, "saturate");
    static_assert(PauseBackoff<3, 10>::next_delay(3) == 6, "double (non power of two cap)");
    static_assert(PauseBackoff<3, 10>::next_delay(6) == 10, "clamp to cap");
    static_assert(PauseBackoff<1, 1>::next_delay(1) == 1, "min == max");
    static_assert(PauseBackoff<0x80000000u, 0xFFFFFFFFu>::next_delay(0x80000000u) == 0xFFFFFFFFu,
                  "no unsigned overflow near UINT_MAX");
    static_assert(SleepBackoff<1, 1>::next_delay(1) == 1, "sleep min == max");
    {
        PauseBackoff<4, 64> b;
        const unsigned expect[] = {4, 8, 16, 32, 64, 64, 64};
        for (unsigned e : expect) {
            if (b.delay() != e) {
                why = "PauseBackoff<4,64> delay sequence wrong";
                return false;
            }
            b();
        }
    }
    {
        PauseBackoff<3, 10> b;
        const unsigned expect[] = {3, 6, 10, 10};
        for (unsigned e : expect) {
            if (b.delay() != e) {
                why = "PauseBackoff<3,10> delay sequence wrong";
                return false;
            }
            b();
        }
    }
    {
        SleepBackoff<100, 800> b;
        const unsigned expect[] = {100, 200, 400, 800, 800};
        for (unsigned e : expect) {
            if (b.delay() != e) {
                why = "SleepBackoff<100,800> delay sequence wrong";
                return false;
            }
            b();
        }
    }
    return true;
}

// ============================================================
// Section / suite runner
// ============================================================

struct SuiteSummary {
    std::string name;
    bool passed;
    double seconds;
};

// Runs fn(sub-communicator) for every rank count and trial. Ranks outside the
// sub-communicator wait in a polite (sleeping) barrier so they do not steal CPU.
// Returns this rank's view: false if any trial failed (identical on all active ranks).
template <class Fn>
bool run_section(Env& env, MPI_Comm world, const std::string& lock_name, const char* title, int trials,
                 bool show_msg, Fn fn) {
    if (env.world_rank == 0) std::cout << "\n" << title << "\n" << std::flush;
    bool section_ok = true;
    for (int k : env.opt.ranks) {
        MPI_Comm sub = MPI_COMM_NULL;
        MPI_Comm_split(world, env.world_rank < k ? 0 : MPI_UNDEFINED, env.world_rank, &sub);
        int passed = 0;
        std::string last_msg;
        for (int t = 1; t <= trials; ++t) {
            polite_barrier(world);
            int good = 1;
            if (sub != MPI_COMM_NULL) {
                env.mon->set_label(lock_name + " / " + title + " / ranks=" + std::to_string(k));
                TestResult r = fn(sub);
                int mine = r.passed ? 1 : 0;
                MPI_Allreduce(&mine, &good, 1, MPI_INT, MPI_MIN, sub);
                if (!r.passed && (env.world_rank != 0 || true)) {
                    std::printf("  [rank %d] Ranks=%3d Trial=%2d FAIL: %s\n", env.world_rank, k, t, r.message.c_str());
                    std::fflush(stdout);
                }
                if (env.world_rank == 0) last_msg = r.message;
            }
            polite_barrier(world);
            if (sub != MPI_COMM_NULL && good) ++passed;
            if (sub != MPI_COMM_NULL && !good) section_ok = false;
        }
        if (env.world_rank == 0 && passed == trials) {
            std::cout << "  Ranks=" << std::setw(3) << k << " : PASS";
            if (show_msg) std::cout << " (" << last_msg << ")";
            else std::cout << " (" << passed << "/" << trials << " trials)";
            std::cout << "\n" << std::flush;
        } else if (env.world_rank == 0) {
            std::cout << "  Ranks=" << std::setw(3) << k << " : FAIL (" << passed << "/" << trials << " trials passed)\n"
                      << std::flush;
        }
        if (sub != MPI_COMM_NULL) MPI_Comm_free(&sub);
    }
    return section_ok;
}

template <class L>
bool run_lock_suite(Env& env, MPI_Comm world, std::vector<SuiteSummary>& summary, const char* name,
                    double lock_scale, bool reduced) {
    const auto t0 = std::chrono::steady_clock::now();
    if (env.world_rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "TEST SUITE: " << name << (reduced ? "  (reduced suite)" : "")
                  << (L::is_fifo ? "  [FIFO]" : "  [non-FIFO]") << "\n";
        std::cout << "============================================================\n" << std::flush;
    }
    const double scale = env.opt.scale * lock_scale;
    auto iters = [scale](int base) { return std::max(100, static_cast<int>(std::lround(base * scale))); };
    const int trials = env.opt.trials;
    bool all = true;

    all &= run_section(env, world, name, "[1] Mutual exclusion (remote in-CS counter)", trials, false,
                       [&](MPI_Comm c) { return test_exclusion<L>(env, c, iters(EXCL_ITERATIONS), 100, 300, 0, 0); });
    all &= run_section(env, world, name, "[2] Protected read-modify-write counter (lost updates)", trials, false,
                       [&](MPI_Comm c) { return test_counter<L>(env, c, iters(COUNTER_ITERATIONS)); });
    if (!reduced) {
        all &= run_section(env, world, name, "[3] Multi-field record (consistency)", trials, false,
                           [&](MPI_Comm c) { return test_record<L>(env, c, iters(RECORD_ITERATIONS)); });
        all &= run_section(env, world, name, "[4] Fairness & FIFO order", 1, true,
                           [&](MPI_Comm c) { return test_fairness<L>(env, c, iters(FAIRNESS_ROUNDS)); });
    }

    struct Cfg {
        int in_lo, in_hi, out, yield_pct;
        const char* title;
    };
    const std::vector<Cfg> full = {
        {0, 0, 0, 0, "[5] Stress: high contention / zero work"},
        {50, 200, 0, 0, "[5] Stress: high contention / critical-section jitter"},
        {100, 200, 200, 0, "[5] Stress: balanced contention"},
        {50, 150, 100, 2, "[5] Stress: holder yields while holding (2%)"},
    };
    const std::vector<Cfg> cfgs = reduced ? std::vector<Cfg>{full[0]} : full;
    for (const auto& s : cfgs) {
        const int base = s.yield_pct > 0 ? 1'200 : STRESS_ITERATIONS;
        all &= run_section(env, world, name, s.title, trials, false, [&, s, base](MPI_Comm c) {
            return test_exclusion<L>(env, c, iters(base), s.in_lo, s.in_hi, s.out, s.yield_pct);
        });
    }

    if (!reduced) {
        all &= run_section(env, world, name, "[6] API: try_lock / Guard / mixed styles", trials, false,
                           [&](MPI_Comm c) { return test_api<L>(env, c, iters(API_ITERATIONS)); });
        all &= run_section(env, world, name, "[7] Several locks, nesting, non-LIFO release", trials, false,
                           [&](MPI_Comm c) { return test_multi_lock<L>(env, c, iters(MULTI_ITERATIONS)); });
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (env.world_rank == 0) {
        std::cout << "\n[RESULT] " << name << " : " << (all ? "ALL TESTS PASSED" : "FAILED") << "\n" << std::flush;
        summary.push_back({name, all, secs});
    }
    return all;
}

// ============================================================
// Registry
// ============================================================

using lock_tas_pause_edge  = basic_lock_tas<PauseBackoff<1, 1>>;
using lock_ttas_pause_edge = basic_lock_ttas<PauseBackoff<1, 1>>;
using lock_tas_sleep_edge  = basic_lock_tas<SleepBackoff<1, 1>>;

using SuiteFn = bool (*)(Env&, MPI_Comm, std::vector<SuiteSummary>&, const char*, double, bool);
struct Entry {
    const char* name;
    SuiteFn run;
    double scale;
    bool reduced;
};

static const std::vector<Entry>& registry() {
    static const std::vector<Entry> r = {
        {"tas", &run_lock_suite<lock_tas>, 1.0, false},
        {"tas-pause", &run_lock_suite<lock_tas_pause>, 1.0, false},
        {"tas-sleep", &run_lock_suite<lock_tas_sleep>, 0.1, false},
        {"ttas", &run_lock_suite<lock_ttas>, 1.0, false},
        {"ttas-pause", &run_lock_suite<lock_ttas_pause>, 1.0, false},
        {"ttas-sleep", &run_lock_suite<lock_ttas_sleep>, 0.1, false},
        {"ticket", &run_lock_suite<lock_ticket>, 1.0, false},
        {"ticket-prop", &run_lock_suite<lock_ticket_prop>, 1.0, false},
        {"mcs", &run_lock_suite<lock_mcs>, 1.0, false},
        {"winlock", &run_lock_suite<lock_winlock>, 1.0, false},
        {"tas-pause-1-1", &run_lock_suite<lock_tas_pause_edge>, 0.5, true},
        {"ttas-pause-1-1", &run_lock_suite<lock_ttas_pause_edge>, 0.5, true},
        {"tas-sleep-1-1", &run_lock_suite<lock_tas_sleep_edge>, 0.1, true},
    };
    return r;
}

// ============================================================
// Negative control
// ============================================================

static int run_negative_control(Env& env, MPI_Comm world) {
    if (env.world_rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "NEGATIVE CONTROL: lock_none (harness sensitivity check)\n";
        std::cout << "============================================================\n";
        std::cout << "These tests are EXPECTED to fail: a harness that cannot detect a\n"
                     "missing lock proves nothing about the real locks.\n\n";
    }
    if (env.world_size < 2) {
        if (env.world_rank == 0) std::cout << "[ERROR] the negative control needs at least 2 ranks\n";
        return 2;
    }
    env.mon->set_label("negative control");
    const int iters = std::max(100, static_cast<int>(std::lround(4000 * env.opt.scale)));
    TestResult m = test_exclusion<lock_none>(env, world, iters, 300, 600, 0, 0);
    TestResult cn = test_counter<lock_none>(env, world, iters);
    TestResult rc = test_record<lock_none>(env, world, iters / 2);
    if (env.world_rank == 0) {
        auto line = [](const char* what, const TestResult& r) {
            std::cout << "  " << std::left << std::setw(22) << what << std::right
                      << (r.passed ? "NOT detected" : "detected    ") << "  (" << r.message << ")\n";
        };
        line("mutual exclusion:", m);
        line("lost updates:", cn);
        line("record consistency:", rc);
        if (!m.passed) std::cout << "\n[OK] Harness detects a missing lock (overlap detector fired).\n";
        else std::cout << "\n[ERROR] Overlap detector did NOT fire on a lock that does nothing.\n";
    }
    return m.passed ? 1 : 0;
}

// ============================================================
// CLI
// ============================================================

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static void print_help() {
    std::cout <<
        "Usage: mpirun -n P ./test_correctness_mpi [options]\n"
        "  --lock=a,b,...   run only these variants (default: all; see --list)\n"
        "  --ranks=1,2,4    rank counts to test on sub-communicators (default: powers of two up to P, and P)\n"
        "  --trials=N       trials per configuration (default 2)\n"
        "  --scale=X        multiply all iteration counts (default 1.0)\n"
        "  --quick          shorthand for --trials=1 --scale=0.25\n"
        "  --stall-ms=N     terminate if a rank makes no progress for N ms (default 8000)\n"
        "  --cpus=N         cores available to the job, for the FIFO fairness check (default: this node's\n"
        "                   hardware threads; ranks above this only get informational fairness output)\n"
        "  --control        run only the negative control (lock_none must be detected; needs P >= 2)\n"
        "  --list           list lock variants\n"
        "  --help           this text\n";
}

static bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* key) -> const char* {
            const std::string k = std::string(key) + "=";
            return a.compare(0, k.size(), k) == 0 ? a.c_str() + k.size() : nullptr;
        };
        const char* v = nullptr;
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--list") o.list = true;
        else if (a == "--control") o.control = true;
        else if (a == "--quick") { o.trials = 1; o.scale = 0.25; }
        else if ((v = val("--lock"))) o.locks = split(v, ',');
        else if ((v = val("--ranks"))) { for (const auto& s : split(v, ',')) o.ranks.push_back(std::max(1, std::atoi(s.c_str()))); }
        else if ((v = val("--trials"))) o.trials = std::max(1, std::atoi(v));
        else if ((v = val("--scale"))) o.scale = std::max(0.001, std::atof(v));
        else if ((v = val("--stall-ms"))) o.stall_ms = std::max(100, std::atoi(v));
        else if ((v = val("--cpus"))) o.cpus = std::max(1, std::atoi(v));
        else return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  // watchdog thread never calls MPI
    MPI_Comm world = MPI_COMM_WORLD;

    Env env;
    MPI_Comm_rank(world, &env.world_rank);
    MPI_Comm_size(world, &env.world_size);
    seed_rng(env.world_rank);

    int rc = 0;
    {
        bool args_ok = parse_args(argc, argv, env.opt);
        if (!args_ok) {
            if (env.world_rank == 0) {
                std::cerr << "Unknown option\n";
                print_help();
            }
            MPI_Finalize();
            return 64;
        }
        if (env.opt.help || env.opt.list) {
            if (env.world_rank == 0) {
                if (env.opt.help) print_help();
                else for (const auto& e : registry()) std::cout << e.name << (e.reduced ? "  (reduced suite)" : "") << "\n";
            }
            MPI_Finalize();
            return 0;
        }

        env.cpus = env.opt.cpus > 0 ? env.opt.cpus
                                    : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        if (env.opt.ranks.empty()) {
            std::set<int> s = {1};
            for (int t = 2; t <= env.world_size; t *= 2) s.insert(t);
            s.insert(env.world_size);
            env.opt.ranks.assign(s.begin(), s.end());
        }
        for (int& k : env.opt.ranks) k = std::min(k, env.world_size);
        std::sort(env.opt.ranks.begin(), env.opt.ranks.end());
        env.opt.ranks.erase(std::unique(env.opt.ranks.begin(), env.opt.ranks.end()), env.opt.ranks.end());

        Monitor mon(env.world_rank, env.opt.stall_ms);
        env.mon = &mon;

        // ---- header ----
        {
            char ver[MPI_MAX_LIBRARY_VERSION_STRING];
            int len = 0;
            MPI_Get_library_version(ver, &len);
            std::string v(ver, static_cast<size_t>(len));
            v = v.substr(0, v.find('\n'));
            int model = -1;
            {
                Window probe(world, 1);
                model = probe.memory_model();
            }
            if (env.world_rank == 0) {
                std::cout << "============================================================\n";
                std::cout << "MPI RMA mutex lock correctness & stress harness\n";
                std::cout << "============================================================\n";
                std::cout << "MPI library:         " << v << "\n";
                std::cout << "Ranks (world):       " << env.world_size << "\n";
                std::cout << "Cores for fairness:  " << env.cpus << "\n";
                std::cout << "RMA memory model:    " << (model == 1 ? "unified" : model == 0 ? "separate" : "unknown") << "\n";
                std::cout << "Thread level:        " << (provided >= MPI_THREAD_FUNNELED ? "funneled (ok)" : "single (watchdog thread does not call MPI)") << "\n";
                std::cout << "Rank counts:         ";
                for (size_t i = 0; i < env.opt.ranks.size(); ++i)
                    std::cout << env.opt.ranks[i] << (i + 1 < env.opt.ranks.size() ? ", " : "\n");
                std::cout << "Trials / scale:      " << env.opt.trials << " / " << env.opt.scale << "\n";
                std::cout << "Watchdog:            no progress for " << env.opt.stall_ms << " ms\n";
                if (env.world_size > env.cpus)
                    std::cout << "Note: more ranks than cores: oversubscribed, expect slow spinlocks.\n";
            }
        }

        if (env.opt.control) {
            rc = run_negative_control(env, world);
        } else {
            std::vector<const Entry*> selected;
            if (env.opt.locks.empty() || (env.opt.locks.size() == 1 && env.opt.locks[0] == "all")) {
                for (const auto& e : registry()) selected.push_back(&e);
            } else {
                for (const auto& want : env.opt.locks) {
                    bool found = false;
                    for (const auto& e : registry())
                        if (want == e.name) {
                            selected.push_back(&e);
                            found = true;
                        }
                    if (!found) {
                        if (env.world_rank == 0) std::cerr << "Unknown lock '" << want << "' (see --list)\n";
                        MPI_Finalize();
                        return 64;
                    }
                }
            }

            int units_ok = 1;
            {
                std::string why;
                const bool good = test_backoff_policies(why);
                if (env.world_rank == 0) {
                    std::cout << "\n[8] Backoff policy unit checks : " << (good ? "PASS" : "FAIL (" + why + ")") << "\n";
                }
                if (!good) units_ok = 0;
            }

            std::vector<SuiteSummary> summary;
            int all = units_ok;
            for (const Entry* e : selected) {
                if (!e->run(env, world, summary, e->name, e->scale, e->reduced)) all = 0;
            }
            int global_all = 1;
            MPI_Allreduce(&all, &global_all, 1, MPI_INT, MPI_MIN, world);

            if (env.world_rank == 0) {
                std::cout << "\n============================================================\nSUMMARY\n"
                             "============================================================\n";
                for (const auto& s : summary)
                    std::cout << "  " << std::left << std::setw(20) << s.name << std::right
                              << (s.passed ? "PASS" : "FAIL") << "  (" << std::fixed << std::setprecision(1)
                              << s.seconds << " s)\n";
                std::cout << "\n============================================================\n"
                          << (global_all ? "ALL SELECTED LOCKS PASSED ALL CORRECTNESS SUITES\n"
                                         : "ONE OR MORE LOCKS FAILED CORRECTNESS SUITES\n")
                          << "============================================================\n" << std::flush;
            }
            rc = global_all ? 0 : 1;
        }
    }
    MPI_Finalize();
    return rc;
}
