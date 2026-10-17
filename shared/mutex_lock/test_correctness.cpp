// ============================================================================
// test_correctness.cpp -- correctness & stress harness for lock.h
//
//   Build : g++ -std=c++17 -O2 -pthread test_correctness.cpp -o test_correctness
//   Run   : ./test_correctness [--help]
//
// Tests per lock:
//   [1] mutual exclusion (overlap detector, seq_cst canary)
//   [2] protected NON-atomic counter (lost updates; TSan-visible races)
//   [3] multi-field invariant (happens-before / memory-ordering check)
//   [4] fairness distribution + FIFO overtaking
//   [5] stress profiles (contention levels, lock-holder preemption)
//   [6] API: try_lock, std::lock_guard, std::unique_lock
//   [7] multiple lock objects, nesting, non-LIFO (hand-over-hand) release
//   [8] backoff policy unit checks (once, not per lock)
// A watchdog declares a deadlock when NO thread makes progress for --stall-ms.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lock.h"

// ============================================================
// Configuration (base iteration counts; scaled by --scale)
// ============================================================

constexpr int MUTEX_ITERATIONS     = 20'000;
constexpr int COUNTER_ITERATIONS   = 20'000;
constexpr int INVARIANT_ITERATIONS = 15'000;
constexpr int STRESS_ITERATIONS    = 10'000;
constexpr int FAIRNESS_ROUNDS      = 20'000;
constexpr int API_ITERATIONS       = 6'000;
constexpr int MULTI_ITERATIONS     = 6'000;

struct Options {
    int trials = 3;
    double scale = 1.0;
    int stall_ms = 5'000;       // fail if no progress at all for this long
    int timeout_ms = 120'000;   // absolute cap per trial
    int hw = 1;                 // hardware threads
    std::vector<int> threads;   // empty -> derived from hw
    std::vector<std::string> locks;  // empty -> all
    bool control = false;
    bool list = false;
    bool help = false;
};

// ============================================================
// Platform description
// ============================================================

static std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

static const char* os_string() {
#if defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(_WIN32)
    return "Windows";
#elif defined(__FreeBSD__)
    return "FreeBSD";
#else
    return "unknown";
#endif
}

static const char* arch_string() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__)
    return "arm32";
#elif defined(__riscv)
    return "riscv";
#elif defined(__powerpc__)
    return "powerpc";
#else
    return "unknown";
#endif
}

static bool built_with_tsan() {
#if defined(__SANITIZE_THREAD__)
    return true;
#elif defined(__has_feature)
  #if __has_feature(thread_sanitizer)
    return true;
  #else
    return false;
  #endif
#else
    return false;
#endif
}

// ============================================================
// Utilities: PRNG, busy work, StartGate, padded counters
// ============================================================

inline std::uint64_t fast_rand() {
    static thread_local std::uint64_t state =
        0x853c49e6748fea9bULL + std::hash<std::thread::id>{}(std::this_thread::get_id());
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
}

// Thread-local busy work (no shared sink -> no false sharing / data race).
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

inline void randomized_busy_work(int min_iters, int max_iters) {
    if (max_iters <= min_iters) {
        busy_work(min_iters);
        return;
    }
    busy_work(min_iters + static_cast<int>(fast_rand() % (max_iters - min_iters + 1)));
}

class StartGate {
public:
    explicit StartGate(int n) : total_(n), ready_(0), go_(false) {}
    void arrive_and_wait() {
        ready_.fetch_add(1, std::memory_order_release);
        while (!go_.load(std::memory_order_acquire)) std::this_thread::yield();
    }
    void start() {
        while (ready_.load(std::memory_order_acquire) < total_) std::this_thread::yield();
        go_.store(true, std::memory_order_release);
    }

private:
    int total_;
    std::atomic<int> ready_;
    std::atomic<bool> go_;
};

struct alignas(kCacheLine) PaddedU64 {
    std::uint64_t v = 0;
};
struct alignas(kCacheLine) PaddedAtomic {
    std::atomic<std::uint64_t> v{0};
};

// Runs body(tid) on n threads, released simultaneously by a StartGate.
template <class Body>
void run_parallel(int n, Body body) {
    StartGate gate(n);
    std::vector<std::thread> threads;
    threads.reserve(n);
    for (int tid = 0; tid < n; ++tid) {
        threads.emplace_back([&gate, &body, tid]() {
            gate.arrive_and_wait();
            body(tid);
        });
    }
    gate.start();
    for (auto& t : threads) t.join();
}

// ============================================================
// Results, progress tracking, watchdog
// ============================================================

struct TestResult {
    bool passed;
    std::string message;
};

inline TestResult fail(std::string m) { return {false, std::move(m)}; }
inline TestResult ok(std::string m) { return {true, std::move(m)}; }

// Per-thread progress slots (own cache line, owner-only writes).
class Progress {
public:
    explicit Progress(int n) : slots_(n) {}
    void tick(int tid) noexcept {
        auto& s = slots_[tid].v;
        s.store(s.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }
    std::uint64_t total() const noexcept {
        std::uint64_t sum = 0;
        for (const auto& s : slots_) sum += s.v.load(std::memory_order_relaxed);
        return sum;
    }

private:
    std::vector<PaddedAtomic> slots_;
};

struct SuiteSummary {
    std::string name;
    bool passed;
    bool stalled;
    double seconds;
};
static std::vector<SuiteSummary> g_summary;

static void print_summary() {
    std::cout << "\n============================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "============================================================\n";
    for (const auto& s : g_summary) {
        std::cout << "  " << std::left << std::setw(20) << s.name << std::right
                  << (s.stalled ? "STALL (deadlock/livelock)" : (s.passed ? "PASS" : "FAIL"))
                  << "  (" << std::fixed << std::setprecision(1) << s.seconds << " s)\n";
    }
    std::cout << std::flush;
}

// A stalled trial leaves spinning threads we cannot cancel. They would steal
// CPU from everything that follows, so report and terminate the process.
[[noreturn]] static void abort_on_stall(const std::string& lock_name, const std::string& where,
                                        int threads, const std::string& msg) {
    std::cout << "\n  >>> " << lock_name << ": " << msg << " [" << where << ", threads="
              << threads << "]\n";
    g_summary.push_back({lock_name, false, true, 0.0});
    print_summary();
    if (built_with_tsan()) {
        std::cout << "\nNote: this is a ThreadSanitizer build. TSan serialises atomic operations on the same\n"
                     "address, so a lock whose waiters hammer one atomic without backoff (plain tas) can starve\n"
                     "the holder's release and stop making progress at high thread counts. This is a TSan\n"
                     "artefact, not a lock bug: use --threads=1,2,4 for sanitizer runs.\n";
    }
    std::cout << "\nTerminating: stuck threads cannot be cancelled. Re-run with --lock=<name> "
                 "to continue with other locks.\n"
              << std::flush;
    std::_Exit(2);
}

// Runs fn(Progress&) on a worker thread; declares failure only if the progress
// counter stops advancing for stall_ms (or the absolute timeout is hit).
template <class F>
TestResult run_trial(const Options& opt, int num_threads, F fn, bool& stalled) {
    struct State {
        Progress prog;
        std::atomic<bool> done{false};
        TestResult result{false, "not finished"};
        explicit State(int n) : prog(n) {}
    };
    auto st = std::make_shared<State>(num_threads);
    stalled = false;

    std::thread worker([st, fn]() mutable {
        st->result = fn(st->prog);
        st->done.store(true, std::memory_order_release);
    });

    using clock = std::chrono::steady_clock;
    const auto begin = clock::now();
    auto last_change = begin;
    std::uint64_t last = 0;

    while (!st->done.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto now = clock::now();
        const std::uint64_t cur = st->prog.total();
        if (cur != last) {
            last = cur;
            last_change = now;
        }
        const auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_change).count();
        const auto total = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count();
        if (idle > opt.stall_ms) {
            worker.detach();
            stalled = true;
            return fail("NO PROGRESS for " + std::to_string(opt.stall_ms) + " ms");
        }
        if (total > opt.timeout_ms) {
            worker.detach();
            stalled = true;
            return fail("TIMEOUT (exceeded " + std::to_string(opt.timeout_ms) + " ms)");
        }
    }
    worker.join();
    return st->result;
}

// ============================================================
// Negative-control lock (does nothing)
// ============================================================

class lock_none {
public:
    static constexpr bool is_fifo = false;
    void acquire() {}
    void release() {}
    void lock() {}
    void unlock() {}
    bool try_lock() { return true; }
};

// ============================================================
// TEST 1: mutual exclusion
// ============================================================

template <class L>
TestResult test_mutual_exclusion(Progress& prog, int n, int iters) {
    L lock;
    std::atomic<int> in_cs{0};
    std::atomic<std::uint64_t> violations{0};
    std::atomic<std::uint64_t> entries{0};

    run_parallel(n, [&](int tid) {
        for (int i = 0; i < iters; ++i) {
            lock.acquire();
            if (in_cs.fetch_add(1, std::memory_order_seq_cst) != 0) {
                violations.fetch_add(1, std::memory_order_relaxed);
            }
            entries.fetch_add(1, std::memory_order_relaxed);
            randomized_busy_work(20, 50);
            in_cs.fetch_sub(1, std::memory_order_seq_cst);
            lock.release();
            prog.tick(tid);
        }
    });

    const auto v = violations.load();
    const auto e = entries.load();
    const std::uint64_t expected = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(iters);
    if (v == 0 && e == expected && in_cs.load() == 0) return ok("mutual exclusion preserved");
    return fail("violations=" + std::to_string(v) + ", entries=" + std::to_string(e) +
                ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 2: protected non-atomic counter
// ============================================================

template <class L>
TestResult test_protected_counter(Progress& prog, int n, int iters) {
    L lock;
    long long counter = 0;  // deliberately NOT atomic

    run_parallel(n, [&](int tid) {
        for (int i = 0; i < iters; ++i) {
            lock.acquire();
            ++counter;
            randomized_busy_work(5, 20);
            lock.release();
            prog.tick(tid);
        }
    });

    const long long expected = static_cast<long long>(n) * iters;
    if (counter == expected) return ok("counter=" + std::to_string(counter));
    return fail("counter=" + std::to_string(counter) + ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 3: multi-field invariant (happens-before between holders)
// ============================================================

struct StateInvariant {
    std::uint64_t seq = 0;
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::uint64_t checksum = 0;
};

template <class L>
TestResult test_memory_invariant(Progress& prog, int n, int iters) {
    constexpr std::uint64_t K = 1000000007ULL;
    L lock;
    StateInvariant st;  // plain fields; only the lock protects them
    std::atomic<std::uint64_t> failures{0};

    run_parallel(n, [&](int tid) {
        for (int i = 0; i < iters; ++i) {
            lock.acquire();

            if (st.seq > 0) {
                const bool valid = (st.a == st.seq * K) && (st.b == ~st.a) &&
                                   (st.checksum == (st.a ^ st.b ^ st.seq));
                if (!valid) failures.fetch_add(1, std::memory_order_relaxed);
            }

            // Update the fields one by one with gaps in between to widen the
            // window in which a broken lock would expose a torn state.
            st.seq++;
            st.a = st.seq * K;
            busy_work(2);
            st.b = ~st.a;
            busy_work(2);
            st.checksum = st.a ^ st.b ^ st.seq;

            lock.release();
            prog.tick(tid);
        }
    });

    const std::uint64_t expected = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(iters);
    const auto f = failures.load();
    if (f == 0 && st.seq == expected && st.b == ~st.a &&
        st.checksum == (st.a ^ st.b ^ st.seq)) {
        return ok("invariants preserved across handovers");
    }
    return fail("invariant_failures=" + std::to_string(f) + ", final_seq=" + std::to_string(st.seq) +
                ", expected_seq=" + std::to_string(expected));
}

// ============================================================
// TEST 4: fairness distribution and overtaking
//
// Overtaking of an acquisition = number of OTHER acquisitions that completed
// between "thread read the global sequence number" and "thread got the lock".
// For a FIFO lock this is about n-1 at most, but a thread can be preempted
// between the read and its doorway, so it is a SOFT indicator (reported, never
// fails). The hard check for FIFO locks (only when threads <= hardware threads,
// otherwise OS scheduling dominates) is that no thread is starved and the
// max/min acquisition ratio stays below 3.
// ============================================================

template <class L>
TestResult test_fairness(Progress& prog, int n, int total_rounds, int hw) {
    if (n <= 1) return ok("single thread (trivially fair)");

    L lock;
    std::vector<PaddedU64> counts(n);
    std::vector<PaddedU64> max_overtake(n);
    alignas(kCacheLine) std::atomic<std::uint64_t> acq_seq{0};  // written inside the CS only
    const std::uint64_t total = static_cast<std::uint64_t>(total_rounds);
    std::vector<std::uint32_t> order(total_rounds);  // acquisition log, written inside the CS only

    run_parallel(n, [&](int tid) {
        for (;;) {
            const std::uint64_t seen = acq_seq.load(std::memory_order_relaxed);
            if (seen >= total) break;
            lock.acquire();
            const std::uint64_t s = acq_seq.load(std::memory_order_relaxed);
            if (s >= total) {
                lock.release();
                break;
            }
            acq_seq.store(s + 1, std::memory_order_relaxed);
            counts[tid].v++;
            order[s] = static_cast<std::uint32_t>(tid);
            const std::uint64_t over = s - seen;
            if (over > max_overtake[tid].v) max_overtake[tid].v = over;
            randomized_busy_work(5, 15);
            lock.release();
            prog.tick(tid);
            randomized_busy_work(2, 8);
        }
    });

    std::uint64_t mn = counts[0].v, mx = counts[0].v, over_max = 0;
    for (int t = 0; t < n; ++t) {
        mn = std::min(mn, counts[t].v);
        mx = std::max(mx, counts[t].v);
        over_max = std::max(over_max, max_overtake[t].v);
    }
    const double ratio = (mn == 0) ? INFINITY : static_cast<double>(mx) / static_cast<double>(mn);

    std::ostringstream stats;
    stats << "min=" << mn << ", max=" << mx << ", ratio=" << std::fixed << std::setprecision(2)
          << ratio << ", overtake_max=" << over_max;

    if (!L::is_fifo) {
        return ok("INFORMATIONAL (non-FIFO lock; " + stats.str() + ")");
    }
    if (n > hw) {
        return ok("INFORMATIONAL (oversubscribed, scheduler dominates; " + stats.str() + ")");
    }

    // Hard check on the "overlap window": thread start-up skew (a thread that
    // begins late, or is not scheduled for a while at the start) must not be
    // mistaken for unfairness, so only acquisitions after every thread has
    // appeared at least once are counted.
    std::vector<char> appeared(n, 0);
    int distinct = 0;
    std::uint64_t start = 0;
    for (std::uint64_t i = 0; i < total && distinct < n; ++i) {
        if (!appeared[order[i]]) {
            appeared[order[i]] = 1;
            ++distinct;
        }
        start = i + 1;
    }
    if (distinct < n || total - start < static_cast<std::uint64_t>(20 * n)) {
        return ok("INFORMATIONAL (threads did not overlap long enough for a fairness verdict; " +
                  stats.str() + ")");
    }
    std::vector<std::uint64_t> wc(n, 0);
    for (std::uint64_t i = start; i < total; ++i) wc[order[i]]++;
    const std::uint64_t wmn = *std::min_element(wc.begin(), wc.end());
    const std::uint64_t wmx = *std::max_element(wc.begin(), wc.end());
    const double wratio = (wmn == 0) ? INFINITY : static_cast<double>(wmx) / static_cast<double>(wmn);
    stats << ", window[min=" << wmn << ", max=" << wmx << ", ratio=" << wratio << "]";
    if (wmn == 0 || wratio > 3.0) {
        return fail("FIFO LOCK UNFAIR/STARVING: " + stats.str());
    }
    std::string msg = "FAIR (" + stats.str() + ")";
    if (over_max > static_cast<std::uint64_t>(16 * n)) msg += " [WARN: overtaking above soft bound 16n]";
    return ok(msg);
}

// ============================================================
// TEST 5: stress profiles
// ============================================================

template <class L>
TestResult test_stress(Progress& prog, int n, int iters, int inside_work, int outside_work,
                       int yield_prob_pct) {
    L lock;
    std::atomic<int> in_cs{0};
    std::atomic<std::uint64_t> violations{0};
    std::atomic<std::uint64_t> completed{0};

    run_parallel(n, [&](int tid) {
        for (int i = 0; i < iters; ++i) {
            if (outside_work > 0) randomized_busy_work(outside_work / 2, outside_work * 3 / 2);

            lock.acquire();
            if (in_cs.fetch_add(1, std::memory_order_seq_cst) != 0) {
                violations.fetch_add(1, std::memory_order_relaxed);
            }
            if (inside_work > 0) randomized_busy_work(inside_work / 2, inside_work * 3 / 2);

            // Yield while holding the lock: lets waiters pile up and exercises
            // queue formation / handover to descheduled successors.
            if (yield_prob_pct > 0 && static_cast<int>(fast_rand() % 100) < yield_prob_pct) {
                std::this_thread::yield();
            }

            completed.fetch_add(1, std::memory_order_relaxed);
            in_cs.fetch_sub(1, std::memory_order_seq_cst);
            lock.release();
            prog.tick(tid);
        }
    });

    const std::uint64_t expected = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(iters);
    if (violations.load() == 0 && completed.load() == expected && in_cs.load() == 0) {
        return ok("stress test passed");
    }
    return fail("violations=" + std::to_string(violations.load()) +
                ", entries=" + std::to_string(completed.load()) +
                ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 6: API (try_lock, lock_guard, unique_lock)
// ============================================================

template <class L>
TestResult test_api(Progress& prog, int n, int iters) {
    // ---- single-lock semantics ----
    {
        L lock;
        if (!lock.try_lock()) return fail("try_lock() failed on a free lock");

        bool other = true;
        std::thread t1([&] {
            other = lock.try_lock();
            if (other) lock.unlock();
        });
        t1.join();
        if (other) return fail("try_lock() succeeded on a held lock");
        lock.unlock();

        { std::lock_guard<L> g(lock); }

        {
            std::unique_lock<L> u(lock, std::defer_lock);
            if (u.owns_lock()) return fail("defer_lock owns the lock");
            u.lock();
            if (!u.owns_lock()) return fail("unique_lock::lock did not take ownership");
            u.unlock();
        }

        {
            std::unique_lock<L> u(lock, std::try_to_lock);
            if (!u.owns_lock()) return fail("try_to_lock failed on a free lock");
            bool o2 = true;
            std::thread t2([&] {
                o2 = lock.try_lock();
                if (o2) lock.unlock();
            });
            t2.join();
            if (o2) return fail("try_lock() succeeded while unique_lock holds the lock");
        }

        if (!lock.try_lock()) return fail("lock not released by unique_lock destructor");
        lock.unlock();
    }

    // ---- concurrent mix of lock styles ----
    L lock;
    long long counter = 0;
    run_parallel(n, [&](int tid) {
        for (int i = 0; i < iters; ++i) {
            switch (i % 3) {
                case 0: {
                    std::lock_guard<L> g(lock);
                    ++counter;
                    break;
                }
                case 1: {
                    std::unique_lock<L> u(lock, std::try_to_lock);
                    if (!u.owns_lock()) u.lock();
                    ++counter;
                    break;
                }
                default: {
                    while (!lock.try_lock()) std::this_thread::yield();
                    ++counter;
                    lock.unlock();
                    break;
                }
            }
            prog.tick(tid);
        }
    });

    const long long expected = static_cast<long long>(n) * iters;
    if (counter == expected) return ok("API semantics + mixed-style counter OK");
    return fail("mixed-style counter=" + std::to_string(counter) + ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 7: multiple locks, nesting, non-LIFO release
// (the original MCS used one thread-local node and would corrupt here)
// ============================================================

template <class L>
TestResult test_multi_lock(Progress& prog, int n, int iters) {
    L A, B;
    long long ca = 0, cb = 0;

    run_parallel(n, [&](int tid) {
        for (int i = 0; i < iters; ++i) {
            switch (i % 3) {
                case 0:  // sequential use of two locks by the same thread
                    A.acquire();
                    ++ca;
                    A.release();
                    B.acquire();
                    ++cb;
                    B.release();
                    break;
                case 1:  // nested, LIFO release
                    A.acquire();
                    B.acquire();
                    ++ca;
                    ++cb;
                    B.release();
                    A.release();
                    break;
                default:  // hand-over-hand: release order differs from acquire order
                    A.acquire();
                    ++ca;
                    B.acquire();
                    A.release();
                    ++cb;
                    B.release();
                    break;
            }
            prog.tick(tid);
        }
    });

    const long long expected = static_cast<long long>(n) * iters;
    if (ca == expected && cb == expected) return ok("multi-lock/nesting OK");
    return fail("ca=" + std::to_string(ca) + ", cb=" + std::to_string(cb) +
                ", expected=" + std::to_string(expected));
}

// ============================================================
// TEST 8: backoff policy unit checks
// ============================================================

static bool test_backoff_policies(std::string& why) {
    // compile-time checks of the delay recurrence (incl. overflow edge)
    static_assert(PauseBackoff<4, 64>::next_delay(4) == 8, "double");
    static_assert(PauseBackoff<4, 64>::next_delay(64) == 64, "saturate");
    static_assert(PauseBackoff<3, 10>::next_delay(3) == 6, "double (non power of two cap)");
    static_assert(PauseBackoff<3, 10>::next_delay(6) == 10, "clamp to cap");
    static_assert(PauseBackoff<1, 1>::next_delay(1) == 1, "min == max");
    static_assert(PauseBackoff<0x80000000u, 0xFFFFFFFFu>::next_delay(0x80000000u) == 0xFFFFFFFFu,
                  "no unsigned overflow near UINT_MAX");
    static_assert(SleepBackoff<1, 1>::next_delay(1) == 1, "sleep min == max");

    // runtime checks of the actual state machine
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
        PauseBackoff<1, 1> b;  // degenerate: always 1 pause + yield
        for (int i = 0; i < 100; ++i) b();
        if (b.delay() != 1) {
            why = "PauseBackoff<1,1> delay changed";
            return false;
        }
    }
    {
        YieldBackoff<1, 4> b;
        const unsigned expect[] = {1, 2, 4, 4};
        for (unsigned e : expect) {
            if (b.delay() != e) {
                why = "YieldBackoff<1,4> delay sequence wrong";
                return false;
            }
            b();
        }
    }
    {
        SleepBackoff<1, 1> b;  // sleeps 1 ns three times
        b();
        b();
        b();
        if (b.delay() != 1) {
            why = "SleepBackoff<1,1> delay changed";
            return false;
        }
        SleepBackoff<100, 800> c;
        const unsigned expect[] = {100, 200, 400, 800, 800};
        for (unsigned e : expect) {
            if (c.delay() != e) {
                why = "SleepBackoff<100,800> delay sequence wrong";
                return false;
            }
            c();
        }
    }
    return true;
}

// ============================================================
// Suite runner
// ============================================================

template <class Fn>
bool run_section(const Options& opt, const std::string& lock_name, const char* title, int trials,
                 bool show_msg, Fn fn) {
    std::cout << "\n" << title << "\n";
    bool section_ok = true;
    for (int n : opt.threads) {
        int passed = 0;
        std::string last_msg;
        for (int t = 1; t <= trials; ++t) {
            bool stalled = false;
            TestResult r = run_trial(opt, n, [fn, n](Progress& p) { return fn(p, n); }, stalled);
            if (stalled) abort_on_stall(lock_name, title, n, r.message);
            if (r.passed) {
                ++passed;
            } else {
                section_ok = false;
                std::cout << "  Threads=" << std::setw(3) << n << " Trial=" << std::setw(2) << t
                          << " FAIL: " << r.message << "\n" << std::flush;
            }
            last_msg = r.message;
        }
        if (passed == trials) {
            std::cout << "  Threads=" << std::setw(3) << n << " : PASS";
            if (show_msg) std::cout << " (" << last_msg << ")";
            else std::cout << " (" << passed << "/" << trials << " trials)";
            std::cout << "\n" << std::flush;
        }
    }
    return section_ok;
}

template <class L>
bool run_lock_suite(const Options& opt, const char* name, double lock_scale, bool reduced) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "\n============================================================\n";
    std::cout << "TEST SUITE: " << name << (reduced ? "  (reduced suite)" : "")
              << (L::is_fifo ? "  [FIFO]" : "  [non-FIFO]") << "\n";
    std::cout << "============================================================\n" << std::flush;

    const double scale = opt.scale * lock_scale;
    auto iters = [scale](int base) { return std::max(200, static_cast<int>(std::lround(base * scale))); };
    const int hw = opt.hw;
    bool all = true;

    all &= run_section(opt, name, "[1] Mutual exclusion (overlap detection)", opt.trials, false,
                       [iters](Progress& p, int n) {
                           return test_mutual_exclusion<L>(p, n, iters(MUTEX_ITERATIONS));
                       });

    all &= run_section(opt, name, "[2] Protected non-atomic counter (lost updates)", opt.trials, false,
                       [iters](Progress& p, int n) {
                           return test_protected_counter<L>(p, n, iters(COUNTER_ITERATIONS));
                       });

    if (!reduced) {
        all &= run_section(opt, name, "[3] Multi-field invariant (happens-before)", opt.trials, false,
                           [iters](Progress& p, int n) {
                               return test_memory_invariant<L>(p, n, iters(INVARIANT_ITERATIONS));
                           });

        all &= run_section(opt, name, "[4] Fairness & overtaking", 1, true,
                           [iters, hw](Progress& p, int n) {
                               return test_fairness<L>(p, n, iters(FAIRNESS_ROUNDS), hw);
                           });
    }

    struct StressConfig {
        int inside, outside, yield_pct;
        const char* name;
    };
    const std::vector<StressConfig> full_cfgs = {
        {0, 0, 0, "high contention / zero work"},
        {40, 0, 0, "high contention / critical-section jitter"},
        {25, 25, 0, "balanced contention"},
        {20, 10, 2, "preemption stress (holder yields 2%)"},
    };
    const std::vector<StressConfig> cfgs =
        reduced ? std::vector<StressConfig>{full_cfgs[0]} : full_cfgs;

    for (const auto& c : cfgs) {
        const std::string title = std::string("[5] Stress: ") + c.name;
        const int base = (c.yield_pct > 0) ? std::min(STRESS_ITERATIONS, 1'500) : STRESS_ITERATIONS;
        all &= run_section(opt, name, title.c_str(), opt.trials, false,
                           [iters, base, c](Progress& p, int n) {
                               return test_stress<L>(p, n, iters(base), c.inside, c.outside,
                                                     c.yield_pct);
                           });
    }

    if (!reduced) {
        all &= run_section(opt, name, "[6] API: try_lock / lock_guard / unique_lock", opt.trials, false,
                           [iters](Progress& p, int n) { return test_api<L>(p, n, iters(API_ITERATIONS)); });

        all &= run_section(opt, name, "[7] Multiple locks, nesting, non-LIFO release", opt.trials, false,
                           [iters](Progress& p, int n) {
                               return test_multi_lock<L>(p, n, iters(MULTI_ITERATIONS));
                           });
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n[RESULT] " << name << " : " << (all ? "ALL TESTS PASSED" : "FAILED") << "\n";
    g_summary.push_back({name, all, false, secs});
    return all;
}

// ============================================================
// Registry of lock variants
// ============================================================

// Degenerate backoff parameters (edge cases); reduced suite.
using lock_tas_pause_edge  = basic_lock_tas<PauseBackoff<1, 1>>;
using lock_ttas_pause_edge = basic_lock_ttas<PauseBackoff<1, 1>>;
using lock_tas_sleep_edge  = basic_lock_tas<SleepBackoff<1, 1>>;

using SuiteFn = bool (*)(const Options&, const char*, double, bool);
struct Entry {
    const char* name;
    SuiteFn run;
    double scale;  // iteration scale (sleep-based variants are far slower)
    bool reduced;
};

static const std::vector<Entry>& registry() {
    static const std::vector<Entry> r = {
        {"tas", &run_lock_suite<lock_tas>, 1.0, false},
        {"tas-pause", &run_lock_suite<lock_tas_pause>, 1.0, false},
        {"tas-yield", &run_lock_suite<lock_tas_yield>, 0.5, false},
        {"tas-sleep", &run_lock_suite<lock_tas_sleep>, 0.05, false},
        {"ttas", &run_lock_suite<lock_ttas>, 1.0, false},
        {"ttas-pause", &run_lock_suite<lock_ttas_pause>, 1.0, false},
        {"ttas-yield", &run_lock_suite<lock_ttas_yield>, 0.5, false},
        {"ttas-sleep", &run_lock_suite<lock_ttas_sleep>, 0.05, false},
        {"ticket", &run_lock_suite<lock_ticket>, 1.0, false},
        {"ticket-prop", &run_lock_suite<lock_ticket_prop>, 1.0, false},
        {"mcs", &run_lock_suite<lock_mcs>, 1.0, false},
        {"tas-pause-1-1", &run_lock_suite<lock_tas_pause_edge>, 0.5, true},
        {"ttas-pause-1-1", &run_lock_suite<lock_ttas_pause_edge>, 0.5, true},
        {"tas-sleep-1-1", &run_lock_suite<lock_tas_sleep_edge>, 0.05, true},
    };
    return r;
}

// ============================================================
// Negative control: the harness must be able to FAIL
// ============================================================

static int run_negative_control(const Options& opt) {
    std::cout << "\n============================================================\n";
    std::cout << "NEGATIVE CONTROL: lock_none (harness sensitivity check)\n";
    std::cout << "============================================================\n";
    std::cout << "These tests are EXPECTED to fail: a harness that cannot detect a\n"
                 "missing lock proves nothing about the real locks.\n\n";

    const int n = std::max(4, std::min(opt.hw, 8));
    const int iters = std::max(200, static_cast<int>(std::lround(20'000 * opt.scale)));
    Progress p1(n), p2(n), p3(n);
    TestResult m = test_mutual_exclusion<lock_none>(p1, n, iters);
    TestResult c = test_protected_counter<lock_none>(p2, n, iters);
    TestResult v = test_memory_invariant<lock_none>(p3, n, iters);

    auto line = [](const char* what, const TestResult& r) {
        std::cout << "  " << std::left << std::setw(22) << what << std::right
                  << (r.passed ? "NOT detected" : "detected    ") << "  (" << r.message << ")\n";
    };
    line("mutual exclusion:", m);
    line("lost updates:", c);
    line("multi-field invariant:", v);

    if (!m.passed) {
        std::cout << "\n[OK] Harness detects a missing lock (overlap detector fired).\n";
        return 0;
    }
    std::cout << "\n[ERROR] Overlap detector did NOT fire on a lock that does nothing. "
                 "Results of the real suites cannot be trusted on this machine.\n";
    return 1;
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

static std::vector<int> default_thread_counts(int hw) {
    std::set<int> s = {1};
    for (int t = 2; t <= hw; t *= 2) s.insert(t);
    s.insert(hw);
    s.insert(std::min(2 * hw, std::max(hw, 64)));  // oversubscription point
    return std::vector<int>(s.begin(), s.end());
}

static void print_help() {
    std::cout <<
        "Usage: test_correctness [options]\n"
        "  --lock=a,b,...   run only these variants (default: all; see --list)\n"
        "  --threads=1,2,4  thread counts (default: powers of two up to the hardware thread\n"
        "                   count, the hardware count itself, and 2x for oversubscription)\n"
        "  --trials=N       trials per configuration (default 3)\n"
        "  --scale=X        multiply all iteration counts (default 1.0; use 0.1 under TSan)\n"
        "  --quick          shorthand for --trials=1 --scale=0.25\n"
        "  --stall-ms=N     fail if no thread makes progress for N ms (default 5000)\n"
        "  --timeout-ms=N   absolute per-trial cap (default 120000)\n"
        "  --control        run only the negative control (lock_none must be detected)\n"
        "  --list           list lock variants\n"
        "  --help           this text\n";
}

static bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* key) -> const char* {
            const std::string k = std::string(key) + "=";
            return a.compare(0, k.size(), k) == 0 ? a.c_str() + k.size() : nullptr;
        };
        const char* v = nullptr;
        if (a == "--help" || a == "-h") opt.help = true;
        else if (a == "--list") opt.list = true;
        else if (a == "--control") opt.control = true;
        else if (a == "--quick") { opt.trials = 1; opt.scale = 0.25; }
        else if ((v = val("--lock"))) opt.locks = split(v, ',');
        else if ((v = val("--threads"))) {
            for (const auto& s : split(v, ',')) opt.threads.push_back(std::max(1, std::atoi(s.c_str())));
        }
        else if ((v = val("--trials"))) opt.trials = std::max(1, std::atoi(v));
        else if ((v = val("--scale"))) opt.scale = std::max(0.001, std::atof(v));
        else if ((v = val("--stall-ms"))) opt.stall_ms = std::max(100, std::atoi(v));
        else if ((v = val("--timeout-ms"))) opt.timeout_ms = std::max(1000, std::atoi(v));
        else {
            std::cerr << "Unknown option: " << a << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    Options opt;
    opt.hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (!parse_args(argc, argv, opt)) {
        print_help();
        return 64;
    }
    if (opt.help) {
        print_help();
        return 0;
    }
    if (opt.list) {
        for (const auto& e : registry()) std::cout << e.name << (e.reduced ? "  (reduced suite)" : "") << "\n";
        return 0;
    }
    if (opt.threads.empty()) opt.threads = default_thread_counts(opt.hw);

    std::cout << "============================================================\n";
    std::cout << "Mutex lock correctness & stress harness\n";
    std::cout << "============================================================\n";
    std::cout << "Platform:            " << os_string() << " / " << arch_string() << "\n";
    std::cout << "Compiler:            " << compiler_string() << "\n";
    std::cout << "Hardware threads:    " << opt.hw << "\n";
    std::cout << "Cache-line padding:  " << kCacheLine << " bytes\n";
    std::cout << "ThreadSanitizer:     " << (built_with_tsan() ? "yes" : "no") << "\n";
    std::cout << "Mutation (relaxed release): " << (kMutatedRelease ? "ENABLED (expect failures)" : "no") << "\n";
    std::cout << "Thread counts:       ";
    for (size_t i = 0; i < opt.threads.size(); ++i)
        std::cout << opt.threads[i] << (i + 1 < opt.threads.size() ? ", " : "\n");
    std::cout << "Trials / scale:      " << opt.trials << " / " << opt.scale << "\n";
    std::cout << "Watchdog:            no progress for " << opt.stall_ms << " ms (cap "
              << opt.timeout_ms << " ms per trial)\n";

    if (opt.control) return run_negative_control(opt);

    // Select variants.
    std::vector<const Entry*> selected;
    if (opt.locks.empty() || (opt.locks.size() == 1 && opt.locks[0] == "all")) {
        for (const auto& e : registry()) selected.push_back(&e);
    } else {
        for (const auto& want : opt.locks) {
            bool found = false;
            for (const auto& e : registry()) {
                if (want == e.name) {
                    selected.push_back(&e);
                    found = true;
                }
            }
            if (!found) {
                std::cerr << "Unknown lock '" << want << "' (see --list)\n";
                return 64;
            }
        }
    }

    // Unit checks that do not depend on the lock under test.
    {
        std::string why;
        std::cout << "\n[8] Backoff policy unit checks : ";
        if (test_backoff_policies(why)) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL (" << why << ")\n";
            return 1;
        }
    }

    bool all_passed = true;
    for (const Entry* e : selected) {
        all_passed &= e->run(opt, e->name, e->scale, e->reduced);
    }

    print_summary();
    std::cout << "\n============================================================\n";
    std::cout << (all_passed ? "ALL SELECTED LOCKS PASSED ALL CORRECTNESS SUITES\n"
                             : "ONE OR MORE LOCKS FAILED CORRECTNESS SUITES\n");
    std::cout << "============================================================\n";
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}