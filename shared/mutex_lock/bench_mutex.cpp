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
    Watchdog wd(rs.max_run_ms, rs.grace_ms, stop, timed_out);

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
    bool core;  // part of the default set; "extended" adds the slow variants
};

static std::vector<LockEntry> registry() {
    std::vector<LockEntry> r = {
        {"tas", &run_lock<lock_tas>, true},
        {"tas-pause", &run_lock<lock_tas_pause>, true},
        {"ttas", &run_lock<lock_ttas>, true},
        {"ttas-pause", &run_lock<lock_ttas_pause>, true},
        {"ticket", &run_lock<lock_ticket>, true},
        {"ticket-prop", &run_lock<lock_ticket_prop>, true},
        {"mcs", &run_lock<lock_mcs>, true},
        {"std-mutex", &run_lock<StdMutex>, true},
#if !defined(_WIN32)
        {"pthread-mutex", &run_lock<PthreadMutex>, true},
#endif
#if defined(__linux__)
        {"pthread-spin", &run_lock<PthreadSpin>, true},
#endif
#if defined(USE_OPENMP)
        {"omp-lock", &run_lock<OmpLock>, true},
        {"omp-critical", &run_omp_critical, true},
#endif
        // Slow under contention (system call / timer per retry): opt-in.
        {"tas-sleep", &run_lock<lock_tas_sleep>, false},
        {"ttas-sleep", &run_lock<lock_ttas_sleep>, false},
        {"tas-yield", &run_lock<lock_tas_yield>, false},
        {"ttas-yield", &run_lock<lock_ttas_yield>, false},
    };
    return r;
}

// ============================================================================
// Options
// ============================================================================

struct Options {
    std::string platform;
    std::string csv = "bench_results.csv";
    std::string mode = "both";  // fixed | strong | both
    std::vector<std::string> locks;
    std::vector<int> threads;
    std::vector<int> outside = {0, 1, 2, 4, 8, 16};
    std::vector<int> N;               // explicit N overrides cs_ns
    std::vector<double> cs_ns = {100, 1000, 10000};
    long iters = 0;                   // fixed mode, per thread (0 = auto)
    long total = 0;                   // strong mode, total (0 = auto)
    int target_ms = 50;
    int reps = 5;
    int warmup = 1;
    long max_run_ms = 20000;
    unsigned seed = 12345;
    bool oversub = false;
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
        "  --mode=fixed|strong|both        fixed work per thread, strong scaling, or both (default both)\n"
        "  --threads=1,2,4                 thread counts (default: powers of two up to the usable CPUs,\n"
        "                                  plus the CPU count itself)\n"
        "  --oversub                       add a secondary series with 2x the usable CPUs\n"
        "  --outside=0,1,2,4,8,16          OUTSIDE_WORK values\n"
        "  --cs-ns=100,1000,10000          target critical-section lengths; N is the prime whose\n"
        "                                  is_prime(N) takes about this long on THIS machine\n"
        "  --N=311,3109                    explicit N values (override --cs-ns)\n"
        "  --iters=K                       fixed mode: iterations per thread (default: auto)\n"
        "  --total=K                       strong mode: total iterations (default: auto)\n"
        "  --target-ms=T                   auto sizing: one thread, no contention, about T ms (default 50)\n"
        "  --reps=R  --warmup=W            repetitions (default 5) and discarded warm-up runs (default 1)\n"
        "  --max-run-ms=M                  cap per run; capped runs are recorded as timeout (default 20000)\n"
        "  --pin                           pin thread i to a CPU (physical cores first; Linux/Windows)\n"
        "  --platform=NAME                 label stored in the CSV (default: hostname)\n"
        "  --csv=FILE                      append rows to FILE (default bench_results.csv)\n"
        "  --seed=S                        seed for the run-order shuffle (default 12345)\n"
        "  --quick                         small preset: reps=3, outside=0,4, cs-ns=1000, target-ms=20\n"
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
        else if (a == "--oversub") o.oversub = true;
        else if (a == "--pin") o.pin = true;
        else if (a == "--dry-run") o.dry_run = true;
        else if (a == "--quick") {
            o.reps = 3;
            o.outside = {0, 4};
            o.cs_ns = {1000};
            o.target_ms = 20;
        }
        else if ((v = val("--locks"))) o.locks = split(v, ',');
        else if ((v = val("--mode"))) o.mode = v;
        else if ((v = val("--threads"))) { o.threads.clear(); for (auto& s : split(v, ',')) o.threads.push_back(std::max(1, std::atoi(s.c_str()))); }
        else if ((v = val("--outside"))) { o.outside.clear(); for (auto& s : split(v, ',')) o.outside.push_back(std::max(0, std::atoi(s.c_str()))); }
        else if ((v = val("--cs-ns"))) { o.cs_ns.clear(); for (auto& s : split(v, ',')) o.cs_ns.push_back(std::max(1.0, std::atof(s.c_str()))); }
        else if ((v = val("--N"))) { o.N.clear(); for (auto& s : split(v, ',')) o.N.push_back(std::max(2, std::atoi(s.c_str()))); }
        else if ((v = val("--iters"))) o.iters = std::max(1L, std::atol(v));
        else if ((v = val("--total"))) o.total = std::max(1L, std::atol(v));
        else if ((v = val("--target-ms"))) o.target_ms = std::max(1, std::atoi(v));
        else if ((v = val("--reps"))) o.reps = std::max(1, std::atoi(v));
        else if ((v = val("--warmup"))) o.warmup = std::max(0, std::atoi(v));
        else if ((v = val("--max-run-ms"))) o.max_run_ms = std::max(100L, std::atol(v));
        else if ((v = val("--platform"))) o.platform = v;
        else if ((v = val("--csv"))) o.csv = v;
        else if ((v = val("--seed"))) o.seed = static_cast<unsigned>(std::atol(v));
        else {
            std::cerr << "Unknown option: " << a << "\n";
            return false;
        }
    }
    if (o.mode != "fixed" && o.mode != "strong" && o.mode != "both") {
        std::cerr << "--mode must be fixed, strong or both\n";
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
    std::string label, cpu, os, arch, compiler;
    int hw = 1;
    bool pinned = false;
};

static const char* kCsvHeader =
    "platform,cpu_model,os,arch,compiler,usable_cpus,pinned,series,mode,lock,threads,outside_work,N,"
    "cs_ns,iters_per_thread,total_iters,rep,status,time_ms,region_ms,finish_min_ms,finish_max_ms,"
    "completed,expected,throughput_mops";

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

    // ---- thread counts ------------------------------------------------
    std::vector<int> core_threads = opt.threads;
    if (core_threads.empty()) {
        for (int t = 1; t <= pi.hw; t *= 2) core_threads.push_back(t);
        if (core_threads.back() != pi.hw) core_threads.push_back(pi.hw);
    }
    if (opt.oversub) core_threads.push_back(2 * pi.hw);
    std::sort(core_threads.begin(), core_threads.end());
    core_threads.erase(std::unique(core_threads.begin(), core_threads.end()), core_threads.end());

    // ---- critical-section sizes (N) -----------------------------------
    std::vector<int> Ns = opt.N;
    if (Ns.empty()) {
        // cost of one loop iteration of is_prime, measured on a large prime
        const double ns_per_d = measure_cs_ns(10007) / 10005.0;
        for (double target : opt.cs_ns) {
            const long approx = std::max(5L, std::lround(target / ns_per_d));
            const int n = next_prime(approx);
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

    // ---- plan ---------------------------------------------------------
    struct Config {
        std::string mode;
        int threads, outside, N;
        double cs_ns;
        long iters, total;
        std::string series;
    };
    std::vector<Config> configs;
    const std::vector<std::string> modes =
        opt.mode == "both" ? std::vector<std::string>{"fixed", "strong"} : std::vector<std::string>{opt.mode};
    for (std::size_t ni = 0; ni < Ns.size(); ++ni)
        for (int out : opt.outside)
            for (int thr : core_threads)
                for (const auto& m : modes) {
                    Config c;
                    c.mode = m;
                    c.threads = thr;
                    c.outside = out;
                    c.N = Ns[ni];
                    c.cs_ns = cs_of[ni];
                    const double per_iter_ns = (1.0 + out) * c.cs_ns;
                    long auto_iters = static_cast<long>(opt.target_ms * 1e6 / per_iter_ns);
                    auto_iters = std::min(20'000'000L, std::max(100L, auto_iters));
                    c.iters = opt.iters > 0 ? opt.iters : auto_iters;
                    c.total = opt.total > 0 ? opt.total : auto_iters * pi.hw;
                    c.total = std::max<long>(c.total, thr);
                    c.series = thr > pi.hw ? "oversub" : "core";
                    configs.push_back(c);
                }

    const long runs_per_cfg = static_cast<long>(sel.size()) * (opt.reps + opt.warmup);
    std::cout << "============================================================\n";
    std::cout << "Mutex lock benchmark\n";
    std::cout << "============================================================\n";
    std::cout << "Platform label : " << pi.label << "\n";
    std::cout << "CPU            : " << pi.cpu << "\n";
    std::cout << "OS / arch      : " << pi.os << " / " << pi.arch << "\n";
    std::cout << "Compiler       : " << pi.compiler << "\n";
    std::cout << "Usable CPUs    : " << pi.hw << "\n";
    std::cout << "Pinning        : " << (pi.pinned ? "on (physical cores first)" : "off") << "\n";
    std::cout << "Padding        : " << kCacheLine << " bytes\n";
    std::cout << "Threads        : ";
    for (std::size_t i = 0; i < core_threads.size(); ++i)
        std::cout << core_threads[i] << (core_threads[i] > pi.hw ? "*" : "") << (i + 1 < core_threads.size() ? ", " : "\n");
    std::cout << "OUTSIDE_WORK   : ";
    for (std::size_t i = 0; i < opt.outside.size(); ++i)
        std::cout << opt.outside[i] << (i + 1 < opt.outside.size() ? ", " : "\n");
    std::cout << "N (cs length)  : ";
    for (std::size_t i = 0; i < Ns.size(); ++i)
        std::cout << Ns[i] << " (" << std::fixed << std::setprecision(0) << cs_of[i] << " ns)" << (i + 1 < Ns.size() ? ", " : "\n");
    std::cout << "Locks (" << sel.size() << ")    : ";
    for (std::size_t i = 0; i < sel.size(); ++i) std::cout << sel[i]->name << (i + 1 < sel.size() ? ", " : "\n");
    std::cout << "Mode / reps    : " << opt.mode << " / " << opt.reps << " (+" << opt.warmup << " warm-up)\n";
    std::cout << "Plan           : " << configs.size() << " configurations x " << runs_per_cfg << " runs = "
              << configs.size() * runs_per_cfg << " runs (cap " << opt.max_run_ms << " ms each)\n";
    std::cout << "(* = oversubscribed secondary series)\n";
    if (opt.dry_run) return 0;
#if defined(__linux__) || defined(__APPLE__)
    if (pi.hw >= 2 && !pi.pinned)
        std::cout << "Note: threads are not pinned; use --pin for steadier numbers (Linux/Windows).\n";
#endif

    // ---- CSV ----------------------------------------------------------
    bool need_header = true;
    {
        std::ifstream probe(opt.csv);
        need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream csv(opt.csv, std::ios::app);
    if (!csv) {
        std::cerr << "Cannot open " << opt.csv << " for writing\n";
        return 1;
    }
    if (need_header) csv << kCsvHeader << "\n";

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

        std::cout << "\n[" << cfg_no << "/" << configs.size() << "] mode=" << c.mode << " threads=" << c.threads
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
                csv << q(pi.label) << ',' << q(pi.cpu) << ',' << pi.os << ',' << pi.arch << ',' << q(pi.compiler)
                    << ',' << pi.hw << ',' << (pi.pinned ? 1 : 0) << ',' << c.series << ',' << c.mode << ','
                    << sel[li]->name << ',' << c.threads << ',' << c.outside << ',' << c.N << ',' << std::fixed
                    << std::setprecision(1) << c.cs_ns << ',' << (rs.strong ? 0 : c.iters) << ','
                    << (rs.strong ? c.total : c.iters * c.threads) << ',' << rep << ',' << status << ','
                    << std::setprecision(4) << r.total_ms << ',' << r.region_ms << ',' << r.finish_min_ms << ','
                    << r.finish_max_ms << ',' << r.completed << ',' << expected << ',' << std::setprecision(4)
                    << mops << "\n";
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
                  << std::setw(10) << "min" << std::setw(10) << "sd" << std::setw(14) << "vs std-mutex" << "\n";
        for (std::size_t li = 0; li < sel.size(); ++li) {
            std::cout << "  " << std::left << std::setw(15) << sel[li]->name << std::right;
            if (times[li].empty()) {
                std::cout << std::setw(11) << "-" << "  (" << bad[li] << " failed runs)\n";
                continue;
            }
            const double med = median(times[li]);
            std::cout << std::fixed << std::setprecision(2) << std::setw(11) << med << std::setw(10)
                      << *std::min_element(times[li].begin(), times[li].end()) << std::setw(10)
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