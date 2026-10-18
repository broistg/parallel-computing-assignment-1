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
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <stdexcept>
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
// Canonical configuration (config/reference.cfg)
// ============================================================================
// The primary benchmark runs the same workload on every platform: the same N,
// the same OUTSIDE_WORK sets and the same iteration counts. They are frozen in
// a text file that was produced once by `--pilot` on the reference machine and
// is shared by the group (see README, "Fixed workload"). The program only looks
// the counts up; it never derives them, and a missing entry is an error.
//
//   iters   N OUTSIDE_WORK count   iterations PER THREAD (fixed-work series)
//   strong  N OUTSIDE_WORK total   TOTAL iterations split across the threads (strong-scaling series)
//
// The fingerprint (config_id) hashes the normalised file, the repetitions,
// warm-up, iteration scale and the parameters of the lock variants, and is
// written to the header and to every CSV row.

struct BenchConfig {
    std::string path;
    std::string norm;  // normalised text without comments and the free-text reference line
    int version = 0;
    std::string reference;
    int target_ms = 0, ref_cpus = 0, reps = 0, warmup = 0;
    int N[3] = {0, 0, 0};            // small, medium, large
    double cs_ns[3] = {0, 0, 0};     // is_prime(N) durations measured on the reference machine
    std::map<std::string, std::vector<int>> outside;  // series -> OUTSIDE_WORK values
    std::map<std::pair<int, int>, long> iters;        // (N, OUTSIDE_WORK) -> iterations per thread
    std::map<std::pair<int, int>, long> strong;       // (N, OUTSIDE_WORK) -> total iterations
};

static const char* const kSeriesNames[5] = {"scale", "contention", "cs", "strong", "oversub"};

static bool load_config(const std::string& path, BenchConfig& c, std::string& err) {
    std::ifstream in(path);
    if (!in) {
        err = "cannot open " + path + " (run from the project root or pass --config=FILE)";
        return false;
    }
    c.path = path;
    std::string line;
    int lineno = 0;
    auto bad = [&](const std::string& m) {
        err = path + ":" + std::to_string(lineno) + ": " + m;
        return false;
    };
    while (std::getline(in, line)) {
        ++lineno;
        const auto h = line.find('#');
        if (h != std::string::npos) line.erase(h);
        std::istringstream ss(line);
        std::vector<std::string> t;
        std::string w;
        while (ss >> w) t.push_back(w);
        if (t.empty()) continue;
        const std::string& k = t[0];
        if (k == "reference") {
            c.reference = line.substr(line.find("reference") + 9);
            const auto b = c.reference.find_first_not_of(" \t");
            c.reference = b == std::string::npos ? "" : c.reference.substr(b);
            continue;  // free text, not part of the fingerprint
        }
        for (const auto& x : t) c.norm += x + " ";
        c.norm += "\n";
        try {
            if (k == "version" && t.size() == 2) c.version = std::stoi(t[1]);
            else if (k == "target_ms" && t.size() == 2) c.target_ms = std::stoi(t[1]);
            else if (k == "reference_logical_cpus" && t.size() == 2) c.ref_cpus = std::stoi(t[1]);
            else if (k == "repetitions" && t.size() == 2) c.reps = std::stoi(t[1]);
            else if (k == "warmup" && t.size() == 2) c.warmup = std::stoi(t[1]);
            else if (k == "n" && t.size() == 4) {
                const int idx = t[1] == "small" ? 0 : t[1] == "medium" ? 1 : t[1] == "large" ? 2 : -1;
                if (idx < 0) return bad("n must be small, medium or large");
                c.N[idx] = std::stoi(t[2]);
                c.cs_ns[idx] = std::stod(t[3]);
            } else if (k == "outside" && t.size() >= 3) {
                std::vector<int> v;
                for (std::size_t i = 2; i < t.size(); ++i) v.push_back(std::stoi(t[i]));
                c.outside[t[1]] = v;
            } else if ((k == "iters" || k == "strong") && t.size() == 4) {
                const std::pair<int, int> key{std::stoi(t[1]), std::stoi(t[2])};
                (k == "iters" ? c.iters : c.strong)[key] = std::stol(t[3]);
            } else {
                return bad("unknown or malformed line starting with '" + k + "'");
            }
        } catch (const std::exception&) {
            return bad("cannot parse a number on the '" + k + "' line");
        }
    }
    if (c.version <= 0 || c.target_ms <= 0 || c.ref_cpus <= 0 || c.reps <= 0 || c.warmup < 0 || !c.N[0] || !c.N[1] ||
        !c.N[2]) {
        err = path + ": missing version, target_ms, reference_logical_cpus, repetitions, warmup or n lines";
        return false;
    }
    for (const char* s : kSeriesNames)
        if (!c.outside.count(s) || c.outside[s].empty()) {
            err = path + ": missing 'outside " + s + " ...' line";
            return false;
        }
    return true;
}

static long cfg_count(const std::map<std::pair<int, int>, long>& m, int N, int out, const char* table,
                      const std::string& path) {
    const auto it = m.find({N, out});
    if (it == m.end())
        throw std::runtime_error(path + " has no '" + table + "' entry for N=" + std::to_string(N) +
                                 " OUTSIDE_WORK=" + std::to_string(out) +
                                 " (regenerate the file with --pilot, or add the entry and bump the version)");
    return it->second;
}

// --iters-scale is for smoke tests only: it multiplies every count; rows are marked with the scale.
static long scaled(long v, double s) { return s == 1.0 ? v : std::max(1L, std::lround(static_cast<double>(v) * s)); }

static std::uint64_t fnv1a(const std::string& s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : s) {
        h ^= ch;
        h *= 1099511628211ULL;
    }
    return h;
}
static std::string hex16(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

// ============================================================================
// Options
// ============================================================================

struct Options {
    std::string platform;
    std::string config = "config/reference.cfg";  // canonical configuration
    std::string csv = "bench_results.csv";
    std::string mode = "timing";      // timing | fairness
    std::string fair_csv = "fair_results.csv";
    long duration_ms = 1000;          // fairness: length of one run
    std::string series = "all";       // scale,contention,cs,strong,oversub | all
    std::string filter;               // restrict the plan, e.g. "N=739,outside=0" (re-running one configuration)
    std::vector<std::string> locks;
    double iters_scale = 1.0;         // smoke tests only (rows are marked, analysis excludes them)
    int reps = -1;                    // -1: from the config file
    int warmup = -1;
    long max_run_ms = 20000;
    unsigned seed = 12345;
    bool pin = false;
    bool dry_run = false;
    bool help = false;
    bool check = false;               // platform report and sanity checks (warn only)
    bool pilot = false;               // one-time pilot on the reference machine: writes a candidate config
    int target_ms = 50;               // pilot: duration of one uncontended thread
    double pilot_ns = 1000;           // pilot: medium critical-section length (small = /10, large = x10)
    int pilot_cpus = 0;               // pilot: reference logical-CPU count (0 = this machine)
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

// --filter=K=V,K=V : keys n (number or small|medium|large), outside, threads, series, mode; values joined by '|'.
// A configuration is run when it matches every key (used to re-run one configuration on every platform).
struct Filter {
    std::map<std::string, std::set<std::string>> k;
    bool empty() const { return k.empty(); }
    bool ok(const std::string& series, const std::string& mode, int thr, int out, int N) const {
        auto has = [&](const char* key, const std::string& v) {
            const auto it = k.find(key);
            return it == k.end() || it->second.count(v) > 0;
        };
        return has("series", series) && has("mode", mode) && has("threads", std::to_string(thr)) &&
               has("outside", std::to_string(out)) && has("n", std::to_string(N));
    }
};

static void print_help() {
    std::cout <<
        "Usage: bench_mutex [options]\n"
        "  --locks=a,b|core|extended|all   locks to run (default core); see --list\n"
        "  --config=FILE                   canonical configuration (default config/reference.cfg)\n"
        "  --series=a,b|all                experiment series (default all): scale, contention, cs, strong, oversub\n"
        "  --filter=K=V,K=V                run only matching configurations; keys n (number or small|medium|large),\n"
        "                                  outside, threads, series, mode; values separated by |\n"
        "  --mode=timing|fairness          timing series (default) or the separate duration-based fairness experiment\n"
        "  --duration-ms=T                 fairness: length of one run (default 1000)\n"
        "  --fair-csv=FILE                 fairness: output file (default fair_results.csv)\n"
        "  --iters-scale=K                 SMOKE TESTS ONLY: multiply all iteration counts (rows are marked, not analysed)\n"
        "  --check                         platform report and sanity checks (warnings only), then exit\n"
        "  --pilot                         ONE-TIME, on the reference machine: measure the real loop and write a\n"
        "                                  candidate config/pilot_<platform>.cfg (options: --target-ms=T, --pilot-ns=T,\n"
        "                                  --pilot-cpus=K)\n"
        "  --reps=R  --warmup=W            repetitions and discarded warm-up runs (default: from the config file)\n"
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
        else if (a == "--check") o.check = true;
        else if (a == "--pilot") o.pilot = true;
        else if ((v = val("--series"))) o.series = v;
        else if ((v = val("--config"))) o.config = v;
        else if ((v = val("--filter"))) o.filter = v;
        else if ((v = val("--iters-scale"))) o.iters_scale = std::atof(v);
        else if ((v = val("--pilot-ns"))) o.pilot_ns = std::max(1.0, std::atof(v));
        else if ((v = val("--pilot-cpus"))) o.pilot_cpus = std::max(1, std::atoi(v));
        else if ((v = val("--locks"))) o.locks = split(v, ',');
        else if ((v = val("--mode"))) o.mode = v;
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
    if (!(o.iters_scale > 0.0)) {
        std::cerr << "--iters-scale must be positive\n";
        return false;
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
    "completed,expected,throughput_mops,physical_cores,kernel,flags,cpu_list,lock_params,experiment,config_id,iters_scale";

static const char* kFairHeader =
    "platform,cpu_model,os,arch,compiler,usable_cpus,pinned,series,lock,threads,outside_work,N,cs_ns,"
    "duration_ms,region_ms,rep,status,total_acq,min_acq,median_acq,max_acq,max_min_ratio,jain,"
    "wait_p50_ns,wait_p99_ns,wait_max_ns,thread_p99_min_ns,thread_p99_max_ns,wait_samples,"
    "physical_cores,kernel,flags,cpu_list,lock_params,config_id,per_thread_acq";

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
                             int N, double cs_ns, const Filter& flt, const std::string& config_id) {
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
    for (int t : threads)  // sustained contention
        if (flt.ok("fairness", "fairness", t, 0, N)) cfgs.push_back({t, 0, N, cs_ns, t > pi.hw ? "oversub" : "core"});
    if (cfgs.empty()) {
        std::cerr << "[bench] the filter leaves no fairness configuration\n";
        return 64;
    }

    std::cout << "============================================================\n";
    std::cout << "Mutex lock fairness experiment (fixed duration, per-thread counters)\n";
    std::cout << "============================================================\n";
    print_platform(pi, lk);
    std::cout << "Config id      : " << config_id << "\n";
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
                << m.tp99_max << ',' << m.samples << ',' << meta_columns(pi) << ',' << q(lk[li]->params) << ',' << config_id << ',' << per
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
// Platform checks (--check, and quietly before every run) and the one-time pilot
// ============================================================================
// Everything here only WARNS. Nothing adjusts the frozen configuration.

static const double kMinRunMs = 20.0;           // the fastest configuration should run at least this long
static const double kFifoOversubFactor = 15.0;  // worst slow-down seen for FIFO locks at 2x oversubscription

static const LockEntry* find_lock(const std::vector<LockEntry>& reg, const char* name) {
    for (const auto& e : reg)
        if (std::strcmp(e.name, name) == 0) return &e;
    return nullptr;
}

// Total time in ns of one batch of `calls` calls of is_prime(N). Large batches are timed instead of single calls.
static double batch_ns(int N, long calls) {
    volatile int np = N;
    const auto t0 = Clock::now();
    std::uint64_t sink = 0;
    for (long i = 0; i < calls; ++i) sink += is_prime(np);
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    if (sink == 0xFFFFFFFFFFFFULL) std::puts("");
    return ns;
}

static long round3(double x) {
    if (x < 1000) return std::max(1L, std::lround(x));
    const double p = std::pow(10.0, std::floor(std::log10(x)) - 2);
    return std::lround(std::round(x / p) * p);
}

// Time of one iteration of the REAL benchmark loop (acquire, is_prime(N), release, then `out` calls of
// is_prime(N)) with one thread, in ns. The batch grows until a run lasts >= 20 ms; median of 3 runs.
static double loop_ns_per_iter(const LockEntry& L, int N, int out, double* spread = nullptr) {
    RunSpec rs;
    rs.threads = 1;
    rs.outside = out;
    rs.N = N;
    rs.iters = 200;
    for (;;) {
        const RunResult r = L.fn(rs);
        if (r.total_ms >= 20.0 || rs.iters > 400'000'000L) break;
        const double f = r.total_ms > 0.01 ? std::min(10.0, std::max(2.0, 25.0 / r.total_ms)) : 10.0;
        rs.iters = static_cast<long>(static_cast<double>(rs.iters) * f);
    }
    std::vector<double> v;
    for (int i = 0; i < 3; ++i) v.push_back(L.fn(rs).total_ms * 1e6 / static_cast<double>(rs.iters));
    const double m = median(v);
    if (spread) *spread = (*std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end())) / m;
    return m;
}

static std::vector<std::string> unavailable_baselines() {
    std::vector<std::string> u;
#if defined(_WIN32)
    u.push_back("pthread-mutex (no pthreads on Windows)");
#endif
#if !defined(__linux__)
    u.push_back("pthread-spin (pthread_spinlock_t is only used on Linux; macOS has none)");
#endif
#if !defined(USE_OPENMP)
    u.push_back("omp-lock, omp-critical (not built with OpenMP; optional, make OMP=1)");
#endif
    return u;
}

struct CheckReport {
    std::vector<std::string> lines, warnings;
    void note(const std::string& s) { lines.push_back(s); }
    void warn(const std::string& s) {
        lines.push_back("WARNING: " + s);
        warnings.push_back(s);
    }
};

static std::string fmt(double v, int prec = 1) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

// cs[i]: is_prime(N[i]) duration measured on THIS machine.
static CheckReport run_checks(const BenchConfig& cfg, const Options& opt, const std::vector<LockEntry>& reg,
                              const std::vector<const LockEntry*>& sel, const double (&cs)[3]) {
    CheckReport R;
    const char* nm[3] = {"small", "medium", "large"};
    // 1. durations of the fixed N on this machine (a platform-dependent consequence of fixed N, recorded as cs_ns)
    for (int i = 0; i < 3; ++i)
        R.note(std::string("is_prime(") + std::to_string(cfg.N[i]) + ") [" + nm[i] + "]: " + fmt(cs[i]) +
               " ns on this machine, " + fmt(cfg.cs_ns[i]) + " ns on the reference (x" + fmt(cs[i] / cfg.cs_ns[i], 2) + ")");

    // 2. optimisation sanity: doubling the batch must double the time (no elimination / hoisting of the calls)
    {
        const long k = std::max(1000L, static_cast<long>(1.5e7 / std::max(1.0, cs[1])));
        double t1 = 1e300, t2 = 1e300;
        for (int i = 0; i < 3; ++i) {
            t1 = std::min(t1, batch_ns(cfg.N[1], k));
            t2 = std::min(t2, batch_ns(cfg.N[1], 2 * k));
        }
        const double ratio = t2 / t1;
        const std::string m = "batch of 2k calls / batch of k calls = " + fmt(ratio, 2) + " (expected 2)";
        if (ratio < 1.8 || ratio > 2.2) R.warn(m + ": is_prime calls may be eliminated or hoisted by the compiler");
        else R.note("sanity 1 ok: " + m);
    }
    // 3. is_prime cost must scale with N like on the reference machine
    {
        const double mr = cs[1] / cs[0], rr = cfg.cs_ns[1] / cfg.cs_ns[0];
        const double lr = cs[2] / cs[1], lrr = cfg.cs_ns[2] / cfg.cs_ns[1];
        const std::string m = "cost ratio medium/small " + fmt(mr, 2) + " (reference " + fmt(rr, 2) + "), large/medium " +
                              fmt(lr, 2) + " (reference " + fmt(lrr, 2) + ")";
        if (mr < 0.75 * rr || mr > 1.25 * rr || lr < 0.75 * lrr || lr > 1.25 * lrr)
            R.warn(m + ": cost does not scale with N like on the reference machine");
        else R.note("sanity 2 ok: " + m);
    }
    // 4 + 5. the real loop: time must grow with 1 + OUTSIDE_WORK, and the summed results must be right
    const LockEntry* rep = find_lock(reg, "std-mutex");
    if (!rep && !sel.empty()) rep = sel[0];
    try {
        const long base = cfg_count(cfg.iters, cfg.N[1], 0, "iters", cfg.path);
        const long cnt = std::max(100L, base / 5);
        const int ows[3] = {0, 2, 8};
        double t[3];
        bool sum_ok = true;
        for (int i = 0; i < 3; ++i) {
            RunSpec rs;
            rs.threads = 1;
            rs.outside = ows[i];
            rs.N = cfg.N[1];
            rs.iters = cnt;
            double best = 1e300;
            for (int rr = 0; rr < 3; ++rr) {
                const RunResult r = rep->fn(rs);
                best = std::min(best, r.total_ms * 1e6 / static_cast<double>(cnt));
                if (r.completed != static_cast<std::uint64_t>(cnt) ||
                    r.sink != r.completed * static_cast<std::uint64_t>(1 + ows[i]) * (is_prime_plain(cfg.N[1]) ? 1u : 0u))
                    sum_ok = false;
            }
            t[i] = best;
        }
        const double r2 = t[1] / t[0], r8 = t[2] / t[0];
        const std::string m = std::string("real loop (") + rep->name + ", 1 thread): ns/iteration " + fmt(t[0]) + " / " +
                              fmt(t[1]) + " / " + fmt(t[2]) + " for OUTSIDE_WORK 0/2/8, ratios " + fmt(r2, 2) + " / " +
                              fmt(r8, 2) + " (expected 3 / 9)";
        if (r2 < 3 * 0.85 || r2 > 3 * 1.15 || r8 < 9 * 0.85 || r8 > 9 * 1.15)
            R.warn(m + ": the outside-work calls do not scale as expected (eliminated, hoisted or overhead-dominated)");
        else R.note("sanity 3 ok: " + m);
        if (!sum_ok) R.warn("the summed is_prime results or the completed counts of the real loop are wrong");
        else R.note("sanity 4 ok: completed counts and summed results of the real loop are correct");
    } catch (const std::exception& e) {
        R.warn(std::string("real-loop checks skipped: ") + e.what());
    }
    // 6. pilot-style estimate for the FASTEST configuration (N small, OUTSIDE_WORK 0, one thread)
    try {
        RunSpec rs;
        rs.threads = 1;
        rs.outside = 0;
        rs.N = cfg.N[0];
        rs.iters = scaled(cfg_count(cfg.iters, cfg.N[0], 0, "iters", cfg.path), opt.iters_scale);
        const RunResult r = rep->fn(rs);
        const std::string m = "fastest configuration (N=" + std::to_string(cfg.N[0]) + ", OUTSIDE_WORK 0, 1 thread, " +
                              std::to_string(rs.iters) + " iterations) runs " + fmt(r.total_ms) + " ms here (about " +
                              std::to_string(cfg.target_ms) + " ms on the reference)";
        if (opt.iters_scale == 1.0 && r.total_ms < kMinRunMs)
            R.warn(m + ": shorter than " + fmt(kMinRunMs, 0) + " ms, timing resolution and start-up costs may matter; "
                       "if it stays short on every fast platform, revise the count in the canonical config");
        else R.note(m);
    } catch (const std::exception& e) {
        R.warn(std::string("fastest-configuration check skipped: ") + e.what());
    }
    // 7. baselines and pinning
    {
        std::string have;
        for (const auto* e : sel) have += std::string(have.empty() ? "" : ", ") + e->name;
        R.note("locks in this run: " + have);
        for (const auto& u : unavailable_baselines()) R.note("not available: " + u);
    }
    return R;
}

static void save_platform_file(const PlatformInfo& pi, const BenchConfig& cfg, const std::string& id, const CheckReport& R) {
    std::error_code ec;
    std::filesystem::create_directories("results", ec);
    std::ofstream f("results/platform_" + pi.label + ".txt");
    if (!f) return;
    f << "platform        " << pi.label << "\ncpu             " << pi.cpu << "\ncores           "
      << (pi.phys > 0 ? std::to_string(pi.phys) : "unknown") << " physical / " << pi.hw << " logical\nos / kernel     "
      << pi.os << " / " << pi.kernel << " (" << pi.arch << ")\ncompiler        " << pi.compiler << "\nflags           "
      << pi.flags << "\npinned          " << (pi.pinned ? "yes: " + pi.cpu_list : std::string("no")) << "\nconfig          "
      << cfg.path << " version " << cfg.version << ", id " << id << " (reference: " << cfg.reference << ")\n\n";
    for (const auto& l : R.lines) f << l << "\n";
}

// ---- one-time pilot ---------------------------------------------------------
static int run_pilot(const Options& opt, const PlatformInfo& pi, const std::vector<LockEntry>& reg,
                     const std::vector<const LockEntry*>& sel) {
    const LockEntry* rep = find_lock(reg, "std-mutex");
    if (!rep) {
        std::cerr << "pilot needs the std-mutex baseline\n";
        return 1;
    }
    BenchConfig c;
    c.path = "(pilot)";
    c.version = 1;
    c.target_ms = opt.target_ms;
    c.ref_cpus = opt.pilot_cpus > 0 ? opt.pilot_cpus : pi.hw;
    c.reps = 10;
    c.warmup = 1;
    c.reference = pi.label + ": " + pi.cpu + "; " + pi.compiler + "; " + pi.flags;
    c.outside = {{"scale", {0, 2, 8}}, {"contention", {0, 1, 2, 4, 8, 16}}, {"cs", {0, 8}}, {"strong", {2}}, {"oversub", {0, 8}}};

    std::cout << "============================================================\n";
    std::cout << "Pilot (one-time, on the reference machine)\n";
    std::cout << "============================================================\n";
    std::cout << "Machine        : " << pi.label << ", " << pi.cpu << ", " << pi.compiler << ", " << pi.flags << "\n";
    std::cout << "Representative : " << rep->name << ", 1 thread, the real benchmark loop\n";
    std::cout << "Target         : " << c.target_ms << " ms for one uncontended thread; strong total = per-thread count x "
              << c.ref_cpus << " (reference logical CPUs)\n\n";

    // N for about 0.1 / 1 / 10 x pilot_ns (cost of is_prime is linear in N)
    const double ns_per_d = measure_cs_ns(10007) / 10005.0;
    const double targets[3] = {0.1 * opt.pilot_ns, opt.pilot_ns, 10.0 * opt.pilot_ns};
    for (int i = 0; i < 3; ++i) {
        c.N[i] = next_prime(std::max(5L, std::lround(targets[i] / ns_per_d)));
        c.cs_ns[i] = measure_cs_ns(c.N[i]);
        std::cout << "N " << (i == 0 ? "small " : i == 1 ? "medium" : "large ") << " = " << std::setw(5) << c.N[i] << "  is_prime "
                  << fmt(c.cs_ns[i]) << " ns (target " << fmt(targets[i], 0) << ")\n";
    }

    // (N, OUTSIDE_WORK) pairs that the series use
    std::set<std::pair<int, int>> pairs;
    for (const char* s : {"scale", "contention", "oversub"})
        for (int o : c.outside[s]) pairs.insert({c.N[1], o});
    for (int i = 0; i < 3; ++i)
        for (int o : c.outside["cs"]) pairs.insert({c.N[i], o});
    for (int o : c.outside["strong"]) pairs.insert({c.N[1], o});
    std::map<std::pair<int, int>, double> nsit;
    std::cout << "\nReal-loop timings (batches >= 20 ms, median of 3):\n";
    std::cout << "     N  outside  ns/iteration  spread   iters/thread  strong total  1-thread ms\n";
    std::vector<std::string> warnings;
    for (const auto& pr : pairs) {
        double sp = 0;
        const double ns = loop_ns_per_iter(*rep, pr.first, pr.second, &sp);
        nsit[pr] = ns;
        const long it = std::max(100L, round3(c.target_ms * 1e6 / ns));
        c.iters[pr] = it;
        std::string tot = "-";
        if (std::count(c.outside["strong"].begin(), c.outside["strong"].end(), pr.second) && pr.first == c.N[1]) {
            c.strong[pr] = round3(static_cast<double>(it) * c.ref_cpus);
            tot = std::to_string(c.strong[pr]);
        }
        std::cout << std::setw(6) << pr.first << std::setw(9) << pr.second << std::setw(14) << fmt(ns) << std::setw(7)
                  << fmt(sp * 100, 1) << "%" << std::setw(15) << it << std::setw(14) << tot << std::setw(13)
                  << fmt(static_cast<double>(it) * ns / 1e6) << "\n";
        if (sp > 0.05) warnings.push_back("timing spread above 5% for N=" + std::to_string(pr.first) + " OUTSIDE_WORK=" +
                                          std::to_string(pr.second) + ": rerun the pilot on an idle machine");
    }

    // all locks must agree on the single-thread loop time, otherwise one count does not fit all of them
    std::cout << "\nSingle-thread loop time per lock at N=" << c.N[1] << ", OUTSIDE_WORK 0 (relative to " << rep->name << "):\n";
    const double base = nsit.at({c.N[1], 0});
    for (const auto* e : sel) {
        const double ns = loop_ns_per_iter(*e, c.N[1], 0);
        std::cout << "  " << std::left << std::setw(15) << e->name << std::right << fmt(ns) << " ns  x" << fmt(ns / base, 3) << "\n";
        if (ns / base > 1.05 || ns / base < 0.95)
            warnings.push_back(std::string(e->name) + " differs by more than 5% from " + rep->name + " in the single-thread loop");
    }

    // projection of the worst case: a fully serialised run at 2x the reference CPUs; FIFO locks can be much slower
    std::cout << "\nProjection at " << 2 * c.ref_cpus << " threads (2x oversubscription): serialised critical sections, and the\n"
              << "worst case x" << fmt(kFifoOversubFactor, 0) << " seen for FIFO locks; cap per run " << opt.max_run_ms << " ms\n";
    for (const auto& kv : c.iters) {
        const double idx = (kv.first.first == c.N[0] ? c.cs_ns[0] : kv.first.first == c.N[1] ? c.cs_ns[1] : c.cs_ns[2]);
        const double serial_ms = static_cast<double>(kv.second) * 2.0 * c.ref_cpus * idx / 1e6;
        if (serial_ms * kFifoOversubFactor > static_cast<double>(opt.max_run_ms))
            warnings.push_back("N=" + std::to_string(kv.first.first) + " OUTSIDE_WORK=" + std::to_string(kv.first.second) +
                               ": projected worst case " + fmt(serial_ms * kFifoOversubFactor, 0) + " ms exceeds the cap");
    }
    std::cout << "  (no configuration exceeds the cap unless listed in the warnings below)\n";

    // sanity checks on this machine with the new configuration
    double cs[3] = {c.cs_ns[0], c.cs_ns[1], c.cs_ns[2]};
    const CheckReport R = run_checks(c, opt, reg, sel, cs);
    std::cout << "\nSanity checks:\n";
    for (const auto& l : R.lines) std::cout << "  " << l << "\n";
    for (const auto& w : warnings) std::cout << "WARNING: " << w << "\n";

    // write the candidate config (never the canonical file itself)
    std::error_code ec;
    std::filesystem::create_directories("config", ec);
    const std::string out = "config/pilot_" + pi.label + ".cfg";
    std::ofstream f(out);
    if (!f) {
        std::cerr << "cannot write " << out << "\n";
        return 1;
    }
    f << "# Canonical benchmark configuration. FROZEN: every platform of the comparison uses the same file.\n"
         "# Produced by `bench_mutex --pilot` on the reference machine (candidate file: review, then copy to\n"
         "# config/reference.cfg). Revise a count only when a configuration is impractical: edit it here, bump\n"
         "# `version`, and rerun THAT configuration on every platform (--filter). See README, \"Fixed workload\".\n"
         "version 1\n"
         "reference " << c.reference << "\n"
         "target_ms " << c.target_ms << "\n"
         "reference_logical_cpus " << c.ref_cpus << "\n"
         "repetitions " << c.reps << "\nwarmup " << c.warmup << "\n\n"
         "# critical-section sizes: is_prime(N) duration on the reference machine in ns\n";
    const char* nmn[3] = {"small", "medium", "large"};
    for (int i = 0; i < 3; ++i) f << "n " << nmn[i] << " " << c.N[i] << " " << fmt(c.cs_ns[i]) << "\n";
    f << "\n# OUTSIDE_WORK values per series\n";
    for (const char* s : kSeriesNames) {
        f << "outside " << s;
        for (int o : c.outside[s]) f << " " << o;
        f << "\n";
    }
    f << "\n# iterations PER THREAD (fixed-work series): N OUTSIDE_WORK count   # ns per loop iteration\n";
    for (const auto& kv : c.iters)
        f << "iters " << kv.first.first << " " << kv.first.second << " " << kv.second << "   # " << fmt(nsit[kv.first]) << " ns\n";
    f << "\n# TOTAL iterations of the strong-scaling series (split across the threads): N OUTSIDE_WORK total\n";
    for (const auto& kv : c.strong) f << "strong " << kv.first.first << " " << kv.first.second << " " << kv.second << "\n";
    std::cout << "\nWrote candidate configuration: " << out << "\n";
    return 0;
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

    // ---- one-time pilot (reference machine only): does not need the canonical config ---------------
    if (opt.pilot) return run_pilot(opt, pi, reg, sel);

    // ---- canonical configuration ---------------------------------------
    BenchConfig cfg;
    {
        std::string err;
        if (!load_config(opt.config, cfg, err)) {
            std::cerr << "[bench] " << err << "\n";
            return 2;
        }
    }
    if (opt.reps < 0) opt.reps = cfg.reps;
    if (opt.warmup < 0) opt.warmup = cfg.warmup;
    std::string config_id;
    {
        std::ostringstream h;
        h << cfg.norm << "reps " << opt.reps << "\nwarmup " << opt.warmup << "\nscale " << std::setprecision(17)
          << opt.iters_scale << "\n";
        for (const auto& e : reg) h << "lock " << e.name << " " << e.params << "\n";
        if (opt.mode == "fairness") h << "duration_ms " << opt.duration_ms << "\n";
        config_id = hex16(fnv1a(h.str()));
    }

    // ---- critical-section sizes (N): fixed, identical on every platform --------
    // The durations are measured here and stored (cs_ns); they never change N or the iteration counts.
    const int Nt[3] = {cfg.N[0], cfg.N[1], cfg.N[2]};
    double cs_of[3];
    for (int i = 0; i < 3; ++i) {
        if (!is_prime_plain(Nt[i]))
            std::cerr << "[bench] warning: N=" << Nt[i] << " is not prime; is_prime returns early and the critical section is tiny\n";
        cs_of[i] = measure_cs_ns(Nt[i]);
    }
    auto cs_for = [&](int N) { return N == Nt[0] ? cs_of[0] : N == Nt[1] ? cs_of[1] : cs_of[2]; };

    // ---- filter ---------------------------------------------------------
    Filter flt;
    for (const auto& kv : split(opt.filter, ',')) {
        const auto eq = kv.find('=');
        const std::string key = eq == std::string::npos ? "" : kv.substr(0, eq);
        if (key != "n" && key != "outside" && key != "threads" && key != "series" && key != "mode") {
            std::cerr << "--filter: unknown key in '" << kv << "' (n, outside, threads, series, mode)\n";
            return 64;
        }
        for (std::string v : split(kv.substr(eq + 1), '|')) {
            if (key == "n") {
                const int idx = v == "small" ? 0 : v == "medium" ? 1 : v == "large" ? 2 : -1;
                if (idx >= 0) v = std::to_string(Nt[idx]);
            }
            flt.k[key].insert(v);
        }
    }

    if (opt.check) {
        const CheckReport R = run_checks(cfg, opt, reg, sel, cs_of);
        std::cout << "Platform check (" << pi.label << ", config " << cfg.path << " version " << cfg.version << ", id " << config_id << ")\n";
        for (const auto& l : R.lines) std::cout << "  " << l << "\n";
        save_platform_file(pi, cfg, config_id, R);
        std::cout << "\n" << R.warnings.size() << " warning(s); saved results/platform_" << pi.label << ".txt\n";
        return 0;
    }

    if (opt.mode == "fairness") {
        if (!opt.dry_run) {
            const CheckReport R = run_checks(cfg, opt, reg, sel, cs_of);
            for (const auto& w : R.warnings) std::cout << "WARNING: " << w << "\n";
            save_platform_file(pi, cfg, config_id, R);
        }
        return run_fairness_mode(opt, pi, cpus, sel, core_threads, Nt[1], cs_of[1], flt, config_id);
    }

    // ---- plan ---------------------------------------------------------
    struct Planned {
        std::string mode;
        int threads, outside, N;
        double cs_ns;
        long iters, total;
        std::string series;  // core | oversub (threads above the logical CPUs)
        std::string exp;     // experiment series this configuration belongs to ('+' joined)
    };
    std::vector<Planned> configs;

    // Counts come from the two frozen tables: 'iters' (per thread) for fixed-work, 'strong' (total) for strong scaling.
    auto add = [&](const std::string& exp, const std::string& mode, int thr, int out, int N) {
        if (!flt.ok(exp, mode, thr, out, N)) return;
        for (auto& c : configs)
            if (c.mode == mode && c.threads == thr && c.outside == out && c.N == N) {
                if (("+" + c.exp + "+").find("+" + exp + "+") == std::string::npos) c.exp += "+" + exp;
                return;
            }
        Planned c;
        c.mode = mode;
        c.threads = thr;
        c.outside = out;
        c.N = N;
        c.cs_ns = cs_for(N);
        c.iters = 0;
        c.total = 0;
        if (mode == "strong") c.total = std::max<long>(scaled(cfg_count(cfg.strong, N, out, "strong", cfg.path), opt.iters_scale), thr);
        else c.iters = scaled(cfg_count(cfg.iters, N, out, "iters", cfg.path), opt.iters_scale);
        c.series = thr > pi.hw ? "oversub" : "core";
        c.exp = exp;
        configs.push_back(c);
    };

    const int N_med = Nt[1];
    try {
        std::set<std::string> want;
        for (const auto& t : split(opt.series, ',')) {
            if (t == "all") {
                for (const char* n : kSeriesNames) want.insert(n);
            } else if (std::find(std::begin(kSeriesNames), std::end(kSeriesNames), t) != std::end(kSeriesNames)) {
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
            for (int out : cfg.outside["scale"])
                for (int t : t_scale) add("scale", "fixed", t, out, N_med);
        if (want.count("contention"))
            for (int out : cfg.outside["contention"])
                for (int t : t3) add("contention", "fixed", t, out, N_med);
        if (want.count("cs"))
            for (int n : Nt)
                for (int out : cfg.outside["cs"])
                    for (int t : t3) add("cs", "fixed", t, out, n);
        if (want.count("strong"))
            for (int out : cfg.outside["strong"])
                for (int t : t_scale) add("strong", "strong", t, out, N_med);
        if (want.count("oversub"))
            for (int out : cfg.outside["oversub"])
                for (int t : {pi.hw, 2 * pi.hw}) add("oversub", "fixed", t, out, N_med);
    } catch (const std::exception& e) {
        std::cerr << "[bench] " << e.what() << "\n";
        return 2;
    }
    if (configs.empty()) {
        std::cerr << "[bench] the series and filter leave no configuration to run\n";
        return 64;
    }

    const long runs_per_cfg = static_cast<long>(sel.size()) * (opt.reps + opt.warmup);
    std::cout << "============================================================\n";
    std::cout << "Mutex lock benchmark\n";
    std::cout << "============================================================\n";
    print_platform(pi, sel);
    std::cout << "Config         : " << cfg.path << " version " << cfg.version << ", id " << config_id << " (reference: " << cfg.reference << ")\n";
    if (opt.iters_scale != 1.0)
        std::cout << "WARNING: SMOKE RUN: iteration counts scaled by " << opt.iters_scale
                  << "; rows are marked and excluded from the analysis.\n";
    if (!opt.filter.empty()) std::cout << "Filter         : " << opt.filter << "\n";
    {
        std::map<std::string, int> per;
        for (const auto& c : configs)
            for (const auto& t : split(c.exp, '+')) ++per[t];
        std::cout << "Series         : ";
        bool first = true;
        for (const char* n : kSeriesNames)
            if (per.count(n)) { std::cout << (first ? "" : ", ") << n << " (" << per[n] << " configs)"; first = false; }
        std::cout << "\n";
        std::cout << "Workload       : fixed N and iteration counts from the canonical config (identical on every platform)\n";
        std::cout << "Medium N       : " << N_med << " (" << std::fixed << std::setprecision(0) << cs_for(N_med) << " ns critical section here, "
                  << cfg.cs_ns[1] << " ns on the reference)\n";
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
        for (int n : nn) std::cout << n << " (" << std::fixed << std::setprecision(0) << cs_for(n) << " ns)  ";
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
    {
        const CheckReport R = run_checks(cfg, opt, reg, sel, cs_of);
        for (const auto& w : R.warnings) std::cout << "WARNING: " << w << "\n";
        save_platform_file(pi, cfg, config_id, R);
    }

    // ---- CSV ----------------------------------------------------------
    std::ofstream csv;
    if (!open_csv(opt.csv, kCsvHeader, csv)) return 1;

    std::mt19937 rng(opt.seed);
    int invalid_runs = 0, timeout_runs = 0;
    std::size_t cfg_no = 0;

    for (const Planned& c : configs) {
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
                    << mops << ',' << meta_columns(pi) << ',' << q(sel[li]->params) << ',' << c.exp << ',' << config_id << ',' << std::setprecision(6) << opt.iters_scale << "\n";
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