// ============================================================================
// bench_mutex.cpp -- performance & scalability study of the spinlocks in lock.h
//
// Workload (assignment skeleton): every thread executes
//     for i in [0, iters):  acquire(); is_prime(N); release();
//                           for j in [0, OUTSIDE_WORK): is_prime(N);
//
// Metric (primary): execution time of the WHOLE PROGRAM = wall time from just
// before the first thread is created to just after the last join.
// Secondary: region time (start gate -> last join, thread creation excluded)
// and the first/last per-thread finish times (spread = fairness indicator).
//
// Two scaling modes (see README / report):
//   fixed  : every thread runs `iters` iterations (the skeleton's NUM_OF_ITERS).
//            Total work grows with the thread count; ideal time stays flat.
//            Reported throughput = completed critical sections / second.
//   strong : the TOTAL iteration count is fixed and split across the threads.
//            Ideal time falls as 1/threads; speed-up is relative to 1 thread.
//
// Build : g++ -std=c++17 -O2 -pthread bench_mutex.cpp -o bench_mutex
//         (add -fopenmp -DUSE_OPENMP for the optional OpenMP baselines)
// Run   : ./bench_mutex --help
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lock.h"

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <pthread.h>
  #include <sys/utsname.h>
  #include <unistd.h>
#endif
#if defined(__linux__)
  #include <sched.h>
#endif
#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif
#if defined(USE_OPENMP)
  #include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

// ============================================================================
// Workload
// ============================================================================

// Exactly the function from the assignment. The argument is read through a
// volatile reference, so the compiler can neither fold the call for a constant
// N nor hoist it out of the loop (it is otherwise a pure function).
static inline int is_prime(const volatile int& pv) {
    const int p = pv;
    int d;
    for (d = 2; d < p; d = d + 1)
        if (p % d == 0) return 0;
    return 1;
}

static bool is_prime_plain(long n) {
    if (n < 2) return false;
    for (long d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

static int next_prime(long n) {
    if (n < 2) n = 2;
    while (!is_prime_plain(n)) ++n;
    return static_cast<int>(n);
}

// Average cost of one is_prime(N) call in ns (single thread, best of 3).
static double measure_cs_ns(int N) {
    volatile int np = N;
    long calls = 1000;
    double best = 1e30;
    // grow the batch until it takes at least ~10 ms
    for (;;) {
        const auto t0 = Clock::now();
        std::uint64_t sink = 0;
        for (long i = 0; i < calls; ++i) sink += is_prime(np);
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        if (sink == 0xFFFFFFFFFFFFULL) std::puts("");  // keep sink alive
        if (ns > 1e7) {
            best = ns / calls;
            break;
        }
        calls *= (ns < 1e6) ? 10 : 2;
    }
    for (int rep = 0; rep < 2; ++rep) {
        const auto t0 = Clock::now();
        std::uint64_t sink = 0;
        for (long i = 0; i < calls; ++i) sink += is_prime(np);
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        if (sink == 0xFFFFFFFFFFFFULL) std::puts("");
        best = std::min(best, ns / calls);
    }
    return best;
}

// ============================================================================
// Baseline locks behind the same acquire()/release() interface
// ============================================================================

class StdMutex {
public:
    void acquire() { m_.lock(); }
    void release() { m_.unlock(); }

private:
    std::mutex m_;
};

#if !defined(_WIN32)
class PthreadMutex {
public:
    PthreadMutex() { pthread_mutex_init(&m_, nullptr); }
    ~PthreadMutex() { pthread_mutex_destroy(&m_); }
    PthreadMutex(const PthreadMutex&) = delete;
    PthreadMutex& operator=(const PthreadMutex&) = delete;
    void acquire() { pthread_mutex_lock(&m_); }
    void release() { pthread_mutex_unlock(&m_); }

private:
    pthread_mutex_t m_;
};
#endif

#if defined(__linux__)
// pthread_spin_* is POSIX but absent on macOS; only enabled on Linux here.
class PthreadSpin {
public:
    PthreadSpin() { pthread_spin_init(&s_, PTHREAD_PROCESS_PRIVATE); }
    ~PthreadSpin() { pthread_spin_destroy(&s_); }
    PthreadSpin(const PthreadSpin&) = delete;
    PthreadSpin& operator=(const PthreadSpin&) = delete;
    void acquire() { pthread_spin_lock(&s_); }
    void release() { pthread_spin_unlock(&s_); }

private:
    pthread_spinlock_t s_;
};
#endif

#if defined(USE_OPENMP)
class OmpLock {
public:
    OmpLock() { omp_init_lock(&l_); }
    ~OmpLock() { omp_destroy_lock(&l_); }
    OmpLock(const OmpLock&) = delete;
    OmpLock& operator=(const OmpLock&) = delete;
    void acquire() { omp_set_lock(&l_); }
    void release() { omp_unset_lock(&l_); }

private:
    omp_lock_t l_;
};
#endif

// ============================================================================
// Platform helpers
// ============================================================================

static std::string trim(std::string s) {
    const char* ws = " \t\r\n\"";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

static std::string cpu_model() {
#if defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line, fallback;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        if (key == "model name") return trim(line.substr(colon + 1));
        if ((key == "Model" || key == "Hardware") && fallback.empty())
            fallback = trim(line.substr(colon + 1));
    }
    if (!fallback.empty()) return fallback;
#elif defined(__APPLE__)
    char buf[256];
    std::size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) return trim(buf);
#elif defined(_WIN32)
    if (const char* p = std::getenv("PROCESSOR_IDENTIFIER")) return trim(p);
#endif
    return "unknown";
}

static const char* os_name() {
#if defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(_WIN32)
    return "Windows";
#else
    return "other";
#endif
}

static const char* arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__)
    return "arm32";
#else
    return "other";
#endif
}

static std::string compiler_name() {
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

// Compile flags: the Makefile passes -DBENCH_FLAGS='"<flags>"' so every CSV row records them.
#ifndef BENCH_FLAGS
#define BENCH_FLAGS "not recorded (build with the Makefile)"
#endif

static bool built_with_tsan() {
#if defined(__SANITIZE_THREAD__)
    return true;
#elif defined(__has_feature)
  #if __has_feature(thread_sanitizer)
    return true;
  #endif
#endif
    return false;
}

static bool built_optimised() {
#if defined(__OPTIMIZE__) || defined(NDEBUG) || defined(_MSC_VER)
    return true;
#else
    return false;
#endif
}

static std::string kernel_string() {
#if !defined(_WIN32)
    struct utsname u;
    if (uname(&u) == 0) return std::string(u.sysname) + " " + u.release;
#endif
    return "unknown";
}

static int g_primary_cpus = 0;  // set by cpu_order() on Linux: allowed CPUs that are not SMT siblings

// Physical cores usable by this process where the OS tells us; 0 = unknown.
static int physical_cores() {
#if defined(__linux__)
    return g_primary_cpus;
#elif defined(__APPLE__)
    int v = 0;
    std::size_t len = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &len, nullptr, 0) == 0) return v;
    return 0;
#elif defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len == 0) return 0;
    std::vector<char> buf(len);
    auto* p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buf.data());
    if (!GetLogicalProcessorInformation(p, &len)) return 0;
    int n = 0;
    for (DWORD i = 0; i < len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
        if (p[i].Relationship == RelationProcessorCore) ++n;
    return n;
#else
    return 0;
#endif
}

constexpr bool kPinSupported =
#if defined(__linux__) || defined(_WIN32)
    true;
#else
    false;
#endif

// CPUs this process may use, physical cores first, SMT siblings last.
static std::vector<int> cpu_order() {
    std::vector<int> order;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> allowed;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &set)) allowed.push_back(c);
    }
    std::vector<int> primary, secondary;
    for (int c : allowed) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) +
                        "/topology/thread_siblings_list");
        int first = c;
        if (f) {
            std::string s;
            std::getline(f, s);
            first = std::atoi(s.c_str());
        }
        (first == c ? primary : secondary).push_back(c);
    }
    g_primary_cpus = static_cast<int>(primary.size());
    order = primary;
    order.insert(order.end(), secondary.begin(), secondary.end());
#endif
    if (order.empty()) {
        const int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        for (int i = 0; i < n; ++i) order.push_back(i);
    }
    return order;
}

static void pin_self(int cpu) {
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#elif defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << (cpu % 64));
#else
    (void)cpu;
#endif
}

// ============================================================================
// Run harness: threads, start gate, whole-program timer, watchdog
// ============================================================================

struct RunSpec {
    int threads = 1;
    int outside = 0;
    int N = 2;
    bool strong = false;
    long iters = 0;   // fixed mode: per thread
    long total = 0;   // strong mode: across all threads
    long max_run_ms = 20000;
    long grace_ms = 3000;
    long duration_ms = 0;  // > 0: fairness run, the main thread raises the stop flag after this time
    bool pin = false;
    const std::vector<int>* cpus = nullptr;

    long iters_of(int tid) const {
        if (!strong) return iters;
        return total / threads + (tid < total % threads ? 1 : 0);
    }
    long expected() const { return strong ? total : iters * threads; }
};

struct alignas(kCacheLine) Slot {
    std::uint64_t completed = 0;
    std::uint64_t sink = 0;
    std::int64_t finish_ns = 0;
};
struct alignas(kCacheLine) StopFlag {
    std::atomic<bool> v{false};
};

struct RunResult {
    double total_ms = 0;   // whole program: before thread creation -> after join
    double region_ms = 0;  // start gate -> after join
    double finish_min_ms = 0, finish_max_ms = 0;  // per-thread finish, from the gate
    std::uint64_t completed = 0;
    std::uint64_t sink = 0;
    bool timed_out = false;
    std::vector<std::uint64_t> per_thread;  // completed iterations of each thread
};

// Sleeps on a condition variable (no CPU use). If the run exceeds the cap it
// raises the stop flag; if the threads still do not finish (true deadlock) it
// aborts the process.
class Watchdog {
public:
    Watchdog(long cap_ms, long grace_ms, StopFlag& stop, std::atomic<bool>& timed_out)
        : stop_(stop), timed_out_(timed_out), th_([this, cap_ms, grace_ms] { run(cap_ms, grace_ms); }) {}
    ~Watchdog() { finish(); }
    void finish() {
        {
            std::lock_guard<std::mutex> g(m_);
            finished_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

private:
    void run(long cap_ms, long grace_ms) {
        std::unique_lock<std::mutex> lk(m_);
        if (cv_.wait_for(lk, std::chrono::milliseconds(cap_ms), [this] { return finished_; })) return;
        timed_out_.store(true);
        stop_.v.store(true);
        if (!cv_.wait_for(lk, std::chrono::milliseconds(grace_ms), [this] { return finished_; })) {
            std::fprintf(stderr, "\n[bench] threads did not stop after the cap (deadlock?). Aborting.\n");
            std::_Exit(3);
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    bool finished_ = false;
    StopFlag& stop_;
    std::atomic<bool>& timed_out_;
    std::thread th_;  // declared last: starts after everything above is initialised
};

static void collect(const std::vector<Slot>& slots, std::int64_t gate_ns, RunResult& r) {
    std::int64_t mn = INT64_MAX, mx = 0;
    for (const auto& s : slots) {
        r.per_thread.push_back(s.completed);
        r.completed += s.completed;
        r.sink += s.sink;
        mn = std::min(mn, s.finish_ns);
        mx = std::max(mx, s.finish_ns);
    }
    r.finish_min_ms = static_cast<double>(mn - gate_ns) / 1e6;
    r.finish_max_ms = static_cast<double>(mx - gate_ns) / 1e6;
}

static std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

template <class Body>
static RunResult run_harness(const RunSpec& rs, Body&& body) {
    const int n = rs.threads;
    std::vector<Slot> slots(n);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    StopFlag stop;
    std::atomic<bool> timed_out{false};
    Watchdog wd(rs.max_run_ms + rs.duration_ms, rs.grace_ms, stop, timed_out);

    std::vector<std::thread> threads;
    threads.reserve(n);

    const auto t_begin = Clock::now();  // ---- whole-program timer starts ----
    for (int tid = 0; tid < n; ++tid) {
        threads.emplace_back([&, tid] {
            if (rs.pin && !rs.cpus->empty()) pin_self((*rs.cpus)[tid % rs.cpus->size()]);
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            body(tid, rs.iters_of(tid), slots[tid], stop.v);
            slots[tid].finish_ns = now_ns();
        });
    }
    while (ready.load(std::memory_order_acquire) < n) std::this_thread::yield();
    const std::int64_t gate_ns = now_ns();
    const auto t_gate = Clock::now();
    go.store(true, std::memory_order_release);
    if (rs.duration_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rs.duration_ms));
        stop.v.store(true, std::memory_order_relaxed);
    }
    for (auto& t : threads) t.join();
    const auto t_end = Clock::now();  // ---- whole-program timer stops ----
    wd.finish();

    RunResult r;
    r.total_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
    r.region_ms = std::chrono::duration<double, std::milli>(t_end - t_gate).count();
    r.timed_out = timed_out.load();
    collect(slots, gate_ns, r);
    return r;
}

// The assignment skeleton, for any lock with acquire()/release().
template <class Lock>
static RunResult run_lock(const RunSpec& rs) {
    Lock lock;
    const int outside = rs.outside;
    const int N = rs.N;
    return run_harness(rs, [&](int, long iters, Slot& s, const std::atomic<bool>& stop) {
        volatile int np = N;  // per-thread, opaque to the optimiser
        std::uint64_t sink = 0;
        long i = 0;
        for (; i < iters; ++i) {
            if (stop.load(std::memory_order_relaxed)) break;
            lock.acquire();
            sink += is_prime(np);
            lock.release();
            for (int j = 0; j < outside; ++j) sink += is_prime(np);
        }
        s.completed = static_cast<std::uint64_t>(i);
        s.sink = sink;
    });
}

// ----------------------------------------------------------------------------
// Fairness experiment: all threads compete for a fixed time. Every thread keeps
// a private acquisition counter; every kSampleEvery-th acquisition additionally
// times acquire() (two clock reads, only on sampled iterations) into a private,
// pre-allocated log. Nothing is shared during the run; everything is merged
// after the join, so the measurement does not add contention.
// ----------------------------------------------------------------------------
constexpr std::uint64_t kSampleEvery = 32;
constexpr std::size_t kMaxWaitSamples = 1u << 17;

struct alignas(kCacheLine) WaitLog {
    std::vector<std::uint32_t> v;  // acquire() latency in ns (clipped to 32 bit)
};

struct FairResult {
    RunResult base;
    std::vector<std::vector<std::uint32_t>> waits;
};

using FairFn = FairResult (*)(const RunSpec&);

template <class Lock>
static FairResult run_fair(const RunSpec& rs) {
    Lock lock;
    std::vector<WaitLog> logs(static_cast<std::size_t>(rs.threads));
    for (auto& l : logs) l.v.reserve(kMaxWaitSamples);
    const int outside = rs.outside;
    const int N = rs.N;
    FairResult fr;
    fr.base = run_harness(rs, [&](int tid, long, Slot& s, const std::atomic<bool>& stop) {
        volatile int np = N;
        std::uint64_t sink = 0, cnt = 0;
        auto& log = logs[static_cast<std::size_t>(tid)].v;
        while (!stop.load(std::memory_order_relaxed)) {
            const bool sample = (cnt % kSampleEvery == 0) && log.size() < kMaxWaitSamples;
            std::int64_t t0 = 0;
            if (sample) t0 = now_ns();
            lock.acquire();
            if (sample) {
                const std::int64_t w = now_ns() - t0;
                log.push_back(static_cast<std::uint32_t>(std::min<std::int64_t>(w, 0xFFFFFFFFLL)));
            }
            sink += is_prime(np);
            lock.release();
            ++cnt;
            for (int j = 0; j < outside; ++j) sink += is_prime(np);
        }
        s.completed = cnt;
        s.sink = sink;
    });
    for (auto& l : logs) fr.waits.push_back(std::move(l.v));
    return fr;
}

#if defined(USE_OPENMP)
// `#pragma omp critical` is a structured block, not an acquire/release pair,
// so it needs its own runner. Thread creation is done by the OpenMP runtime.
static RunResult run_omp_critical(const RunSpec& rs) {
    const int n = rs.threads;
    std::vector<Slot> slots(n);
    StopFlag stop;
    std::atomic<bool> timed_out{false};
    Watchdog wd(rs.max_run_ms, rs.grace_ms, stop, timed_out);
    std::int64_t gate_ns = 0;
    Clock::time_point t_gate;

    const auto t_begin = Clock::now();
#pragma omp parallel num_threads(n)
    {
        const int tid = omp_get_thread_num();
        if (tid < n) {
            volatile int np = rs.N;
            std::uint64_t sink = 0;
            long i = 0;
            const long iters = rs.iters_of(tid);
#pragma omp barrier
#pragma omp master
            {
                gate_ns = now_ns();
                t_gate = Clock::now();
            }
            for (; i < iters; ++i) {
                if (stop.v.load(std::memory_order_relaxed)) break;
#pragma omp critical(bench_cs)
                { sink += is_prime(np); }
                for (int j = 0; j < rs.outside; ++j) sink += is_prime(np);
            }
            slots[tid].completed = static_cast<std::uint64_t>(i);
            slots[tid].sink = sink;
            slots[tid].finish_ns = now_ns();
        }
    }
    const auto t_end = Clock::now();
    wd.finish();

    RunResult r;
    r.total_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
    r.region_ms = std::chrono::duration<double, std::milli>(t_end - t_gate).count();
    r.timed_out = timed_out.load();
    collect(slots, gate_ns, r);
    return r;
}
#endif

// ============================================================================
// Registry
// ============================================================================

using RunFn = RunResult (*)(const RunSpec&);
struct LockEntry {
    const char* name;
    RunFn fn;
    FairFn fair;        // nullptr: no fairness variant (omp critical is a block, not a lock object)
    bool core;          // part of the default set; "extended" adds the slow variants
    std::string params; // parameters that define the variant, written to the CSV
};

// Human-readable parameters taken from the lock types themselves, so the
// documentation cannot drift away from the code.
template <int N> struct Rank : Rank<N - 1> {};
template <> struct Rank<0> {};

template <class B>
static std::string backoff_desc() {
    std::string s = std::string("backoff=") + B::kKind;
    if (B::kMax != 0)
        s += " " + std::to_string(B::kMin) + ".." + std::to_string(B::kMax) + " " + B::kUnit;
    return s + " (" + B::kNote + ")";
}
template <class L>
auto params_impl(Rank<2>) -> decltype((void)sizeof(typename L::backoff_type), std::string()) {
    return backoff_desc<typename L::backoff_type>();
}
template <class L>
auto params_impl(Rank<1>) -> decltype((void)L::pause_unit, std::string()) {
    return L::pause_unit == 0 ? "poll serving counter with SpinWait(" + std::to_string(LOCK_SPIN_LIMIT) +
                                    " cpu_relax, then yield)"
                              : "proportional backoff, " + std::to_string(L::pause_unit) +
                                    " cpu_relax per position in line (cap 16384)";
}
template <class L>
std::string params_impl(Rank<0>) {
    return "";
}

template <class L>
static LockEntry mk(const char* name, bool core, const std::string& extra = "") {
    std::string p = params_impl<L>(Rank<2>{});
    if (p.empty()) p = extra.empty() ? "library implementation" : extra;
    else if (!extra.empty()) p += "; " + extra;
    return LockEntry{name, &run_lock<L>, &run_fair<L>, core, p};
}

static std::vector<LockEntry> registry() {
    const std::string spin = "waiting loops use SpinWait(" + std::to_string(LOCK_SPIN_LIMIT) +
                             " cpu_relax, then yield)";
    std::vector<LockEntry> r;
    r.push_back(mk<lock_tas>("tas", true));
    r.push_back(mk<lock_tas_pause>("tas-pause", true));
    r.push_back(mk<lock_ttas>("ttas", true, spin));
    r.push_back(mk<lock_ttas_pause>("ttas-pause", true, spin));
    r.push_back(mk<lock_ticket>("ticket", true));
    r.push_back(mk<lock_ticket_prop>("ticket-prop", true));
    r.push_back(mk<lock_mcs>("mcs", true, "queue lock, each thread spins on its own node; " + spin));
    r.push_back(mk<StdMutex>("std-mutex", true));
#if !defined(_WIN32)
    r.push_back(mk<PthreadMutex>("pthread-mutex", true));
#endif
#if defined(__linux__)
    r.push_back(mk<PthreadSpin>("pthread-spin", true));
#endif
#if defined(USE_OPENMP)
    r.push_back(mk<OmpLock>("omp-lock", true));
    r.push_back(LockEntry{"omp-critical", &run_omp_critical, nullptr, true, "#pragma omp critical"});
#endif
    // Slow under contention (system call / timer per retry): opt-in.
    r.push_back(mk<lock_tas_sleep>("tas-sleep", false));
    r.push_back(mk<lock_ttas_sleep>("ttas-sleep", false, spin));
    r.push_back(mk<lock_tas_yield>("tas-yield", false));
    r.push_back(mk<lock_ttas_yield>("ttas-yield", false, spin));
    return r;
}

// ============================================================================
// Options
// ============================================================================

struct Options {
    std::string platform;
    std::string csv = "bench_results.csv";
    std::string mode = "timing";      // timing | fairness
    std::string fair_csv = "fair_results.csv";
    long duration_ms = 1000;          // fairness: length of one run
    std::string series = "all";       // scale,contention,cs,strong,oversub | all
    double medium_ns = 1000;          // "medium" critical-section length of the series
    std::vector<std::string> locks;
    long iters = 0;                   // fixed mode, per thread (0 = auto)
    long total = 0;                   // strong mode, total (0 = auto)
    int target_ms = 50;
    int reps = 5;
    int warmup = 1;
    long max_run_ms = 20000;
    unsigned seed = 12345;
    bool pin = false;
    bool dry_run = false;
    bool help = false;
};

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
        "Usage: bench_mutex [options]\n"
        "  --locks=a,b|core|extended|all   locks to run (default core); see --list\n"
        "  --series=a,b|all                experiment series (default all): scale, contention, cs, strong, oversub\n"
        "  --medium-ns=T                   medium critical-section length (default 1000 ns); small = T/10, large = 10*T\n"
        "  --mode=timing|fairness          timing series (default) or the separate duration-based fairness experiment\n"
        "  --duration-ms=T                 fairness: length of one run (default 1000)\n"
        "  --fair-csv=FILE                 fairness: output file (default fair_results.csv)\n"
        "  --iters=K                       fixed-work series: iterations per thread (default: auto)\n"
        "  --total=K                       strong series: total iterations (default: auto)\n"
        "  --target-ms=T                   auto sizing: one thread, no contention, about T ms (default 50)\n"
        "  --reps=R  --warmup=W            repetitions (default 5) and discarded warm-up runs (default 1)\n"
        "  --max-run-ms=M                  cap per run; capped runs are recorded as timeout (default 20000)\n"
        "  --pin                           pin thread i to a CPU (physical cores first; Linux/Windows)\n"
        "  --platform=NAME                 label stored in the CSV (default: hostname)\n"
        "  --csv=FILE                      append rows to FILE (default bench_results.csv)\n"
        "  --seed=S                        seed for the run-order shuffle (default 12345)\n"
        "  --dry-run                       print the plan and exit\n"
        "  --list                          list lock names\n";
}

static bool parse_args(int argc, char** argv, Options& o, bool& list) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* key) -> const char* {
            const std::string k = std::string(key) + "=";
            return a.compare(0, k.size(), k) == 0 ? a.c_str() + k.size() : nullptr;
        };
        const char* v = nullptr;
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--list") list = true;
        else if (a == "--pin") o.pin = true;
        else if (a == "--dry-run") o.dry_run = true;
        else if ((v = val("--series"))) o.series = v;
        else if ((v = val("--medium-ns"))) o.medium_ns = std::max(1.0, std::atof(v));
        else if ((v = val("--locks"))) o.locks = split(v, ',');
        else if ((v = val("--mode"))) o.mode = v;
        else if ((v = val("--iters"))) o.iters = std::max(1L, std::atol(v));
        else if ((v = val("--total"))) o.total = std::max(1L, std::atol(v));
        else if ((v = val("--target-ms"))) o.target_ms = std::max(1, std::atoi(v));
        else if ((v = val("--reps"))) o.reps = std::max(1, std::atoi(v));
        else if ((v = val("--warmup"))) o.warmup = std::max(0, std::atoi(v));
        else if ((v = val("--max-run-ms"))) o.max_run_ms = std::max(100L, std::atol(v));
        else if ((v = val("--platform"))) o.platform = v;
        else if ((v = val("--csv"))) o.csv = v;
        else if ((v = val("--duration-ms"))) o.duration_ms = std::max(100L, std::atol(v));
        else if ((v = val("--fair-csv"))) o.fair_csv = v;
        else if ((v = val("--seed"))) o.seed = static_cast<unsigned>(std::atol(v));
        else {
            std::cerr << "Unknown option: " << a << "\n";
            return false;
        }
    }
    if (o.mode != "timing" && o.mode != "fairness") {
        std::cerr << "--mode must be timing or fairness\n";
        return false;
    }
    return true;
}

// ============================================================================
// Statistics, CSV
// ============================================================================

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Linear-interpolation quantile (type 7, the default of R and numpy).
static double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}
static double iqr(const std::vector<double>& v) { return quantile(v, 0.75) - quantile(v, 0.25); }

static double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = 0;
    for (double x : v) m += x;
    m /= static_cast<double>(v.size());
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

static std::string q(const std::string& s) {
    std::string out = "\"";
    for (char c : s) out += (c == '"') ? '\'' : c;
    return out + "\"";
}

struct PlatformInfo {
    std::string label, cpu, os, arch, compiler, kernel, flags, cpu_list;
    int hw = 1;
    int phys = 0;
    bool pinned = false;
};

static void print_platform(const PlatformInfo& pi, const std::vector<const LockEntry*>& sel) {
    std::cout << "Platform label : " << pi.label << "\n";
    std::cout << "CPU            : " << pi.cpu << "\n";
    std::cout << "Cores          : " << (pi.phys > 0 ? std::to_string(pi.phys) : std::string("unknown"))
              << " physical / " << pi.hw << " logical usable\n";
    std::cout << "OS / kernel    : " << pi.os << " / " << pi.kernel << " (" << pi.arch << ")\n";
    std::cout << "Compiler       : " << pi.compiler << "\n";
    std::cout << "Flags          : " << pi.flags << "\n";
    std::cout << "Pinning        : "
              << (pi.pinned ? "on, CPU order (physical cores first): " + pi.cpu_list : std::string("off")) << "\n";
    std::cout << "Padding        : " << kCacheLine << " bytes\n";
    std::cout << "Lock parameters:\n";
    for (const auto* e : sel) std::cout << "  " << std::left << std::setw(14) << e->name << std::right << e->params << "\n";
    if (built_with_tsan())
        std::cout << "WARNING: built with ThreadSanitizer, do not use these timings (correctness diagnostics only).\n";
    else if (!built_optimised())
        std::cout << "WARNING: built without optimisation (-O2 expected for performance runs).\n";
}

static const char* kCsvHeader =
    "platform,cpu_model,os,arch,compiler,usable_cpus,pinned,series,mode,lock,threads,outside_work,N,"
    "cs_ns,iters_per_thread,total_iters,rep,status,time_ms,region_ms,finish_min_ms,finish_max_ms,"
    "completed,expected,throughput_mops,physical_cores,kernel,flags,cpu_list,lock_params,experiment";

static const char* kFairHeader =
    "platform,cpu_model,os,arch,compiler,usable_cpus,pinned,series,lock,threads,outside_work,N,cs_ns,"
    "duration_ms,region_ms,rep,status,total_acq,min_acq,median_acq,max_acq,max_min_ratio,jain,"
    "wait_p50_ns,wait_p99_ns,wait_max_ns,thread_p99_min_ns,thread_p99_max_ns,wait_samples,"
    "physical_cores,kernel,flags,cpu_list,lock_params,per_thread_acq";

// Opens `path` for appending; writes `header` if the file is new. Refuses a file written with another
// column layout (an older version), because appended rows would not line up.
static bool open_csv(const std::string& path, const char* header, std::ofstream& out) {
    {
        std::ifstream probe(path);
        std::string first;
        if (probe.good() && std::getline(probe, first) && !first.empty() && first != header) {
            std::cerr << path << " has a different column layout (written by an older version?).\n"
                      << "Use another file name or move the old file away.\n";
            return false;
        }
    }
    bool need_header;
    {
        std::ifstream probe(path);
        need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    }
    out.open(path, std::ios::app);
    if (!out) {
        std::cerr << "Cannot open " << path << " for writing\n";
        return false;
    }
    if (need_header) out << header << "\n";
    return true;
}

static std::string platform_columns(const PlatformInfo& pi) {
    std::ostringstream o;
    o << q(pi.label) << ',' << q(pi.cpu) << ',' << pi.os << ',' << pi.arch << ',' << q(pi.compiler) << ','
      << pi.hw << ',' << (pi.pinned ? 1 : 0);
    return o.str();
}
static std::string meta_columns(const PlatformInfo& pi) {
    std::ostringstream o;
    o << pi.phys << ',' << q(pi.kernel) << ',' << q(pi.flags) << ',' << q(pi.cpu_list);
    return o.str();
}

// ---- fairness metrics ------------------------------------------------------
struct FairMetrics {
    double min = 0, med = 0, max = 0, ratio = 0, jain = 0;
    double p50 = 0, p99 = 0, pmax = 0, tp99_min = 0, tp99_max = 0;
    std::size_t samples = 0;
    std::uint64_t total = 0;
};

static double pct_u32(std::vector<std::uint32_t>& v, double p) {
    if (v.empty()) return 0.0;
    const std::size_t k = std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return static_cast<double>(v[k]);
}

static FairMetrics fair_metrics(const FairResult& fr) {
    FairMetrics m;
    std::vector<double> c(fr.base.per_thread.begin(), fr.base.per_thread.end());
    double sum = 0, sq = 0;
    for (double x : c) {
        sum += x;
        sq += x * x;
    }
    m.total = fr.base.completed;
    m.min = *std::min_element(c.begin(), c.end());
    m.max = *std::max_element(c.begin(), c.end());
    m.med = quantile(c, 0.5);
    m.ratio = m.min > 0 ? m.max / m.min : INFINITY;
    m.jain = sq > 0 ? (sum * sum) / (static_cast<double>(c.size()) * sq) : 0.0;
    std::vector<std::uint32_t> all;
    m.tp99_min = INFINITY;
    for (const auto& w : fr.waits) {
        all.insert(all.end(), w.begin(), w.end());
        std::vector<std::uint32_t> t = w;
        const double p = pct_u32(t, 0.99);
        if (!w.empty()) {
            m.tp99_min = std::min(m.tp99_min, p);
            m.tp99_max = std::max(m.tp99_max, p);
        }
    }
    if (!(m.tp99_min < INFINITY)) m.tp99_min = 0;
    m.samples = all.size();
    m.p50 = pct_u32(all, 0.50);
    m.p99 = pct_u32(all, 0.99);
    m.pmax = all.empty() ? 0.0 : static_cast<double>(*std::max_element(all.begin(), all.end()));
    return m;
}


// ============================================================================
// Fairness experiment (--mode=fairness)
// ============================================================================

static int run_fairness_mode(const Options& opt, const PlatformInfo& pi, const std::vector<int>& cpus,
                             const std::vector<const LockEntry*>& sel, std::vector<int> threads,
                             int N, double cs_ns) {
    threads.erase(std::remove_if(threads.begin(), threads.end(), [](int t) { return t < 2; }), threads.end());
    if (threads.empty()) {
        std::cerr << "[bench] fairness needs at least 2 threads\n";
        return 64;
    }
    std::vector<const LockEntry*> lk;
    for (const auto* e : sel)
        if (e->fair) lk.push_back(e);

    struct Cfg { int threads, outside, N; double cs_ns; std::string series; };
    std::vector<Cfg> cfgs;
    for (int t : threads) cfgs.push_back({t, 0, N, cs_ns, t > pi.hw ? "oversub" : "core"});  // sustained contention

    std::cout << "============================================================\n";
    std::cout << "Mutex lock fairness experiment (fixed duration, per-thread counters)\n";
    std::cout << "============================================================\n";
    print_platform(pi, lk);
    std::cout << "Duration       : " << opt.duration_ms << " ms per run, reps " << opt.reps << " (+" << opt.warmup << " warm-up)\n";
    std::cout << "Wait sampling  : acquire() timed on every " << kSampleEvery << "th acquisition (per-thread log, merged after join)\n";
    std::cout << "Plan           : " << cfgs.size() << " configurations x " << lk.size() * (opt.reps + opt.warmup)
              << " runs = " << cfgs.size() * lk.size() * (opt.reps + opt.warmup) << " runs\n";
    if (opt.dry_run) return 0;
    if (opt.reps < 5) std::cout << "Note: fewer than 5 repetitions, the medians are rough.\n";

    std::ofstream csv;
    if (!open_csv(opt.fair_csv, kFairHeader, csv)) return 1;

    std::mt19937 rng(opt.seed);
    int invalid = 0, timeouts = 0;
    std::size_t no = 0;
    for (const Cfg& c : cfgs) {
        ++no;
        RunSpec rs;
        rs.threads = c.threads;
        rs.outside = c.outside;
        rs.N = c.N;
        rs.max_run_ms = opt.max_run_ms;
        rs.duration_ms = opt.duration_ms;
        rs.pin = pi.pinned;
        rs.cpus = &cpus;
        const int expected_prime = is_prime_plain(c.N) ? 1 : 0;
        std::cout << "\n[" << no << "/" << cfgs.size() << "] threads=" << c.threads << (c.series == "oversub" ? "*" : "")
                  << " outside=" << c.outside << " N=" << c.N << " (cs " << std::fixed << std::setprecision(0)
                  << c.cs_ns << " ns)\n" << std::flush;

        std::vector<std::vector<FairMetrics>> ms(lk.size());
        auto one = [&](std::size_t li, int rep) {
            const FairResult fr = lk[li]->fair(rs);
            const char* status = "ok";
            if (fr.base.timed_out) status = "timeout";
            else if (fr.base.sink != fr.base.completed * static_cast<std::uint64_t>(1 + c.outside) *
                                         static_cast<std::uint64_t>(expected_prime)) status = "invalid";
            if (rep < 0) return;
            if (std::strcmp(status, "ok") != 0) {
                (std::strcmp(status, "timeout") == 0 ? timeouts : invalid)++;
                return;
            }
            const FairMetrics m = fair_metrics(fr);
            ms[li].push_back(m);
            std::string per;
            for (std::size_t i = 0; i < fr.base.per_thread.size(); ++i)
                per += (i ? ";" : "") + std::to_string(fr.base.per_thread[i]);
            csv << platform_columns(pi) << ',' << c.series << ',' << lk[li]->name << ',' << c.threads << ','
                << c.outside << ',' << c.N << ',' << std::fixed << std::setprecision(1) << c.cs_ns << ','
                << opt.duration_ms << ',' << std::setprecision(2) << fr.base.region_ms << ',' << rep << ',' << status
                << ',' << m.total << ',' << static_cast<std::uint64_t>(m.min) << ',' << std::setprecision(1) << m.med
                << ',' << static_cast<std::uint64_t>(m.max) << ',' << std::setprecision(4) << m.ratio << ',' << m.jain
                << ',' << std::setprecision(1) << m.p50 << ',' << m.p99 << ',' << m.pmax << ',' << m.tp99_min << ','
                << m.tp99_max << ',' << m.samples << ',' << meta_columns(pi) << ',' << q(lk[li]->params) << ',' << per
                << "\n";
            csv.flush();
        };
        std::vector<std::size_t> order(lk.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        for (int w = 0; w < opt.warmup; ++w) {
            std::shuffle(order.begin(), order.end(), rng);
            for (std::size_t li : order) one(li, -1);
        }
        for (int rep = 0; rep < opt.reps; ++rep) {
            std::shuffle(order.begin(), order.end(), rng);
            for (std::size_t li : order) one(li, rep);
        }

        std::cout << "  " << std::left << std::setw(15) << "lock" << std::right << std::setw(10) << "min acq"
                  << std::setw(10) << "median" << std::setw(10) << "max" << std::setw(9) << "max/min" << std::setw(8)
                  << "Jain" << std::setw(13) << "p50 wait" << std::setw(13) << "p99 wait" << std::setw(14) << "max wait"
                  << "\n";
        auto med_of = [&](std::size_t li, double FairMetrics::*f) {
            std::vector<double> v;
            for (const auto& m : ms[li]) v.push_back(m.*f);
            return quantile(v, 0.5);
        };
        for (std::size_t li = 0; li < lk.size(); ++li) {
            std::cout << "  " << std::left << std::setw(15) << lk[li]->name << std::right;
            if (ms[li].empty()) {
                std::cout << "  (no valid runs)\n";
                continue;
            }
            std::cout << std::fixed << std::setprecision(0) << std::setw(10) << med_of(li, &FairMetrics::min)
                      << std::setw(10) << med_of(li, &FairMetrics::med) << std::setw(10) << med_of(li, &FairMetrics::max)
                      << std::setprecision(2) << std::setw(9) << med_of(li, &FairMetrics::ratio) << std::setprecision(3)
                      << std::setw(8) << med_of(li, &FairMetrics::jain) << std::setprecision(0) << std::setw(10)
                      << med_of(li, &FairMetrics::p50) << " ns" << std::setw(10) << med_of(li, &FairMetrics::p99) << " ns"
                      << std::setw(11) << med_of(li, &FairMetrics::pmax) << " ns\n";
        }
        std::cout << std::flush;
    }
    std::cout << "\n============================================================\n";
    std::cout << "Done. Rows appended to " << opt.fair_csv << "\n";
    std::cout << "These numbers describe the observed behaviour in this environment; they are not a fairness guarantee.\n";
    if (invalid) std::cout << "WARNING: " << invalid << " run(s) failed validation\n";
    if (timeouts) std::cout << "WARNING: " << timeouts << " run(s) timed out (excluded)\n";
    std::cout << "============================================================\n";
    return invalid > 0 ? 1 : 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    Options opt;
    bool list = false;
    if (!parse_args(argc, argv, opt, list)) {
        print_help();
        return 64;
    }
    if (opt.help) {
        print_help();
        return 0;
    }
    const std::vector<LockEntry> reg = registry();
    if (list) {
        for (const auto& e : reg) std::cout << e.name << (e.core ? "" : "  (extended)") << "\n";
        return 0;
    }

    // ---- platform -----------------------------------------------------
    const std::vector<int> cpus = cpu_order();
    PlatformInfo pi;
    pi.hw = static_cast<int>(cpus.size());
    pi.cpu = cpu_model();
    pi.os = os_name();
    pi.arch = arch_name();
    pi.compiler = compiler_name();
    pi.kernel = kernel_string();
    pi.flags = BENCH_FLAGS;
    pi.phys = physical_cores();
    for (std::size_t i = 0; i < cpus.size(); ++i) pi.cpu_list += (i ? " " : "") + std::to_string(cpus[i]);
    if (!opt.platform.empty()) {
        pi.label = opt.platform;
    } else if (const char* h = std::getenv("HOSTNAME")) {
        pi.label = h;
    } else if (const char* c = std::getenv("COMPUTERNAME")) {
        pi.label = c;
    } else {
        pi.label = "platform";
    }
    pi.pinned = opt.pin && kPinSupported;
    if (opt.pin && !kPinSupported)
        std::cerr << "[bench] warning: thread pinning is not supported on this OS; running unpinned\n";

    // ---- select locks -------------------------------------------------
    std::vector<const LockEntry*> sel;
    if (opt.locks.empty()) opt.locks = {"core"};
    for (const auto& want : opt.locks) {
        if (want == "core" || want == "all" || want == "extended") {
            for (const auto& e : reg)
                if (want == "all" || want == "extended" || e.core) {
                    if (std::find(sel.begin(), sel.end(), &e) == sel.end()) sel.push_back(&e);
                }
            continue;
        }
        bool found = false;
        for (const auto& e : reg)
            if (want == e.name) {
                if (std::find(sel.begin(), sel.end(), &e) == sel.end()) sel.push_back(&e);
                found = true;
            }
        if (!found) {
            std::cerr << "Unknown lock '" << want << "' (see --list)\n";
            return 64;
        }
    }

    // ---- thread counts: powers of two up to the logical CPUs, plus logical and physical counts
    std::vector<int> core_threads;
    for (int t = 1; t <= pi.hw; t *= 2) core_threads.push_back(t);
    if (core_threads.back() != pi.hw) core_threads.push_back(pi.hw);
    if (pi.phys > 0 && pi.phys < pi.hw) core_threads.push_back(pi.phys);  // SMT boundary
    std::sort(core_threads.begin(), core_threads.end());
    core_threads.erase(std::unique(core_threads.begin(), core_threads.end()), core_threads.end());

    // ---- critical-section sizes (N): small / medium / large = about 0.1x / 1x / 10x --medium-ns
    std::vector<int> Ns, Nt;
    {
        // cost of one loop iteration of is_prime, measured on a large prime
        const double ns_per_d = measure_cs_ns(10007) / 10005.0;
        for (double target : {0.1 * opt.medium_ns, opt.medium_ns, 10.0 * opt.medium_ns}) {
            const int n = next_prime(std::max(5L, std::lround(target / ns_per_d)));
            Nt.push_back(n);
            if (std::find(Ns.begin(), Ns.end(), n) == Ns.end()) Ns.push_back(n);
        }
    }
    std::vector<double> cs_of(Ns.size());
    for (std::size_t i = 0; i < Ns.size(); ++i) {
        if (!is_prime_plain(Ns[i]))
            std::cerr << "[bench] warning: N=" << Ns[i]
                      << " is not prime; is_prime returns early and the critical section is tiny\n";
        cs_of[i] = measure_cs_ns(Ns[i]);
    }

    if (opt.mode == "fairness") return run_fairness_mode(opt, pi, cpus, sel, core_threads, Nt.at(1), cs_of[std::find(Ns.begin(), Ns.end(), Nt.at(1)) - Ns.begin()]);

    // ---- plan ---------------------------------------------------------
    struct Config {
        std::string mode;
        int threads, outside, N;
        double cs_ns;
        long iters, total;
        std::string series;  // core | oversub (threads above the logical CPUs)
        std::string exp;     // experiment series this configuration belongs to ('+' joined)
    };
    std::vector<Config> configs;
    std::map<int, double> cs_by_N;
    for (std::size_t i = 0; i < Ns.size(); ++i) cs_by_N[Ns[i]] = cs_of[i];

    auto add = [&](const std::string& exp, const std::string& mode, int thr, int out, int N) {
        for (auto& c : configs)
            if (c.mode == mode && c.threads == thr && c.outside == out && c.N == N) {
                if (("+" + c.exp + "+").find("+" + exp + "+") == std::string::npos) c.exp += "+" + exp;
                return;
            }
        Config c;
        c.mode = mode;
        c.threads = thr;
        c.outside = out;
        c.N = N;
        c.cs_ns = cs_by_N.at(N);
        const double per_iter_ns = (1.0 + out) * c.cs_ns;
        long auto_iters = static_cast<long>(opt.target_ms * 1e6 / per_iter_ns);
        auto_iters = std::min(20'000'000L, std::max(100L, auto_iters));
        c.iters = opt.iters > 0 ? opt.iters : auto_iters;
        c.total = opt.total > 0 ? opt.total : auto_iters * pi.hw;
        c.total = std::max<long>(c.total, thr);
        c.series = thr > pi.hw ? "oversub" : "core";
        c.exp = exp;
        configs.push_back(c);
    };

    int N_med = 0;
    {
        // N values of the 0.1 / medium / 10 us critical sections (calibrated once, above)
        const int N_small = Nt.at(0), N_large = Nt.at(2);
        N_med = Nt.at(1);

        std::set<std::string> want;
        for (const auto& t : split(opt.series, ',')) {
            if (t == "all") {
                for (const char* n : {"scale", "contention", "cs", "strong", "oversub"}) want.insert(n);
            } else if (t == "scale" || t == "contention" || t == "cs" || t == "strong" || t == "oversub") {
                want.insert(t);
            } else {
                std::cerr << "Unknown series '" << t << "' (scale, contention, cs, strong, oversub, all)\n";
                return 64;
            }
        }
        // thread-count sets: powers of two up to the logical CPUs, plus logical and physical counts
        std::vector<int> t_scale = core_threads;  // already holds powers of two, hw and phys
        std::vector<int> t3 = {1, pi.hw};
        if (pi.phys > 0 && pi.phys < pi.hw) t3.push_back(pi.phys);
        std::sort(t3.begin(), t3.end());
        if (want.count("scale"))
            for (int out : {0, 2, 8})
                for (int t : t_scale) add("scale", "fixed", t, out, N_med);
        if (want.count("contention"))
            for (int out : {0, 1, 2, 4, 8, 16})
                for (int t : t3) add("contention", "fixed", t, out, N_med);
        if (want.count("cs"))
            for (int n : {N_small, N_med, N_large})
                for (int out : {0, 8})
                    for (int t : t3) add("cs", "fixed", t, out, n);
        if (want.count("strong"))
            for (int t : t_scale) add("strong", "strong", t, 2, N_med);
        if (want.count("oversub"))
            for (int out : {0, 8})
                for (int t : {pi.hw, 2 * pi.hw}) add("oversub", "fixed", t, out, N_med);
    }

    const long runs_per_cfg = static_cast<long>(sel.size()) * (opt.reps + opt.warmup);
    std::cout << "============================================================\n";
    std::cout << "Mutex lock benchmark\n";
    std::cout << "============================================================\n";
    print_platform(pi, sel);
    {
        std::map<std::string, int> per;
        for (const auto& c : configs)
            for (const auto& t : split(c.exp, '+')) ++per[t];
        std::cout << "Series         : ";
        bool first = true;
        for (const char* n : {"scale", "contention", "cs", "strong", "oversub"})
            if (per.count(n)) { std::cout << (first ? "" : ", ") << n << " (" << per[n] << " configs)"; first = false; }
        std::cout << "\n";
        std::cout << "Medium N       : " << N_med << " (" << std::fixed << std::setprecision(0) << cs_by_N.at(N_med) << " ns critical section)\n";
    }
    {
        std::set<int> th;
        std::set<int> ows;
        std::set<int> nn;
        for (const auto& c : configs) { th.insert(c.threads); ows.insert(c.outside); nn.insert(c.N); }
        std::cout << "Threads        : ";
        for (int t : th) std::cout << t << (t > pi.hw ? "*" : "") << " ";
        std::cout << "(logical " << pi.hw << ", physical " << (pi.phys > 0 ? std::to_string(pi.phys) : std::string("unknown")) << ")\n";
        std::cout << "OUTSIDE_WORK   : ";
        for (int o : ows) std::cout << o << " ";
        std::cout << "\nN (cs length)  : ";
        for (int n : nn) std::cout << n << " (" << std::fixed << std::setprecision(0) << cs_by_N.at(n) << " ns)  ";
        std::cout << "\n";
    }
    std::cout << "Mode / reps    : " << "per series" << " / " << opt.reps << " (+" << opt.warmup << " warm-up)\n";
    std::cout << "Plan           : " << configs.size() << " configurations x " << runs_per_cfg << " runs = "
              << configs.size() * runs_per_cfg << " runs (cap " << opt.max_run_ms << " ms each)\n";
    std::cout << "(* = oversubscribed secondary series)\n";
    if (opt.dry_run) return 0;
    if (opt.reps < 5) std::cout << "Note: fewer than 5 repetitions: the IQR is not meaningful (smoke test only).\n";
#if defined(__linux__) || defined(__APPLE__)
    if (pi.hw >= 2 && !pi.pinned)
        std::cout << "Note: threads are not pinned; use --pin for steadier numbers (Linux/Windows).\n";
#endif

    // ---- CSV ----------------------------------------------------------
    std::ofstream csv;
    if (!open_csv(opt.csv, kCsvHeader, csv)) return 1;

    std::mt19937 rng(opt.seed);
    int invalid_runs = 0, timeout_runs = 0;
    std::size_t cfg_no = 0;

    for (const Config& c : configs) {
        ++cfg_no;
        RunSpec rs;
        rs.threads = c.threads;
        rs.outside = c.outside;
        rs.N = c.N;
        rs.strong = (c.mode == "strong");
        rs.iters = c.iters;
        rs.total = c.total;
        rs.max_run_ms = opt.max_run_ms;
        rs.pin = pi.pinned;
        rs.cpus = &cpus;

        std::cout << "\n[" << cfg_no << "/" << configs.size() << "] [" << c.exp << "] mode=" << c.mode << " threads=" << c.threads
                  << (c.series == "oversub" ? "*" : "") << " outside=" << c.outside << " N=" << c.N << " (cs "
                  << std::fixed << std::setprecision(0) << c.cs_ns << " ns) "
                  << (rs.strong ? "total=" + std::to_string(c.total) : "iters/thread=" + std::to_string(c.iters))
                  << "\n" << std::flush;

        const std::uint64_t expected = static_cast<std::uint64_t>(rs.expected());
        const int expected_prime = is_prime_plain(c.N) ? 1 : 0;
        std::vector<std::vector<double>> times(sel.size());
        std::vector<int> bad(sel.size(), 0);

        auto one_run = [&](std::size_t li, int rep) {
            const RunResult r = sel[li]->fn(rs);
            const std::uint64_t want_sink = r.completed * static_cast<std::uint64_t>(1 + c.outside) *
                                            static_cast<std::uint64_t>(expected_prime);
            const char* status = "ok";
            if (r.timed_out) status = "timeout";
            else if (r.completed != expected || r.sink != want_sink) status = "invalid";
            if (rep >= 0) {
                if (std::strcmp(status, "ok") == 0) times[li].push_back(r.total_ms);
                else {
                    ++bad[li];
                    (std::strcmp(status, "timeout") == 0 ? timeout_runs : invalid_runs)++;
                }
                const double mops = r.total_ms > 0 ? static_cast<double>(r.completed) / (r.total_ms * 1e3) : 0.0;
                csv << platform_columns(pi) << ',' << c.series << ',' << c.mode << ','
                    << sel[li]->name << ',' << c.threads << ',' << c.outside << ',' << c.N << ',' << std::fixed
                    << std::setprecision(1) << c.cs_ns << ',' << (rs.strong ? 0 : c.iters) << ','
                    << (rs.strong ? c.total : c.iters * c.threads) << ',' << rep << ',' << status << ','
                    << std::setprecision(4) << r.total_ms << ',' << r.region_ms << ',' << r.finish_min_ms << ','
                    << r.finish_max_ms << ',' << r.completed << ',' << expected << ',' << std::setprecision(4)
                    << mops << ',' << meta_columns(pi) << ',' << q(sel[li]->params) << ',' << c.exp << "\n";
                csv.flush();
            }
        };

        std::vector<std::size_t> order(sel.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        // Block-randomised: every pass runs every lock once, in a fresh random
        // order, so slow drift (thermal, frequency, background load) is spread
        // across the locks instead of favouring whichever runs first.
        for (int w = 0; w < opt.warmup; ++w) {
            std::shuffle(order.begin(), order.end(), rng);
            for (std::size_t li : order) one_run(li, -1);
        }
        for (int rep = 0; rep < opt.reps; ++rep) {
            std::shuffle(order.begin(), order.end(), rng);
            for (std::size_t li : order) one_run(li, rep);
        }

        // ---- summary table for this configuration ----
        double ref = 0;
        for (std::size_t li = 0; li < sel.size(); ++li)
            if (std::strcmp(sel[li]->name, "std-mutex") == 0 && !times[li].empty()) ref = median(times[li]);
        std::cout << "  " << std::left << std::setw(15) << "lock" << std::right << std::setw(11) << "median ms"
                  << std::setw(10) << "iqr" << std::setw(10) << "min" << std::setw(10) << "sd" << std::setw(14)
                  << "vs std-mutex" << "\n";
        for (std::size_t li = 0; li < sel.size(); ++li) {
            std::cout << "  " << std::left << std::setw(15) << sel[li]->name << std::right;
            if (times[li].empty()) {
                std::cout << std::setw(11) << "-" << "  (" << bad[li] << " failed runs)\n";
                continue;
            }
            const double med = median(times[li]);
            std::cout << std::fixed << std::setprecision(2) << std::setw(11) << med << std::setw(10)
                      << iqr(times[li]) << std::setw(10) << *std::min_element(times[li].begin(), times[li].end()) << std::setw(10)
                      << stddev(times[li]);
            if (ref > 0) std::cout << std::setw(11) << std::setprecision(2) << med / ref << "x";
            if (bad[li] > 0) std::cout << "  (" << bad[li] << " failed runs)";
            std::cout << "\n";
        }
        std::cout << std::flush;
    }

    std::cout << "\n============================================================\n";
    std::cout << "Done. Rows appended to " << opt.csv << "\n";
    if (invalid_runs > 0)
        std::cout << "WARNING: " << invalid_runs << " run(s) failed validation (completed count or work sum wrong)\n";
    if (timeout_runs > 0)
        std::cout << "WARNING: " << timeout_runs << " run(s) hit the " << opt.max_run_ms
                  << " ms cap (recorded with status=timeout, excluded from the summaries)\n";
    std::cout << "============================================================\n";
    return (invalid_runs > 0) ? 1 : 0;
}