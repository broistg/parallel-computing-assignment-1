// ============================================================================
// bench_mpi_mutex.cpp -- throughput, scalability and fairness study of mpi_lock.h
//
//   Build : mpicxx -std=c++17 -O2 bench_mpi_mutex.cpp -o bench_mpi_mutex
//   Run   : mpirun -n 4 ./bench_mpi_mutex [--help]
//
// Timing mode (default): each rank performs a fixed number of lock/unlock rounds.
// The critical section reads and writes a counter stored on rank 0 (non-atomic
// Get/Put, completed before release), so the lock protects real remote data.
// Timer = whole parallel region (barrier ... barrier), maximum over ranks. The
// result is validated (counter == ranks * ops) before a row is emitted.
//
// Fairness mode (--mode=fairness): all ranks compete for a fixed time; every
// rank keeps a private acquisition counter and times every 32nd acquire() into a
// private log; counts and logs are gathered only after the run, so the
// measurement adds no communication or contention to the lock under test.
//
// Output: CSV on stdout (rank 0), human-readable header and summaries on stderr.
// ============================================================================
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(__linux__)
  #include <sched.h>
#endif
#if !defined(_WIN32)
  #include <sys/utsname.h>
#endif
#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#include "mpi_lock.h"

using namespace dlock;

#ifndef BENCH_FLAGS
#define BENCH_FLAGS "not recorded (build with the Makefile)"
#endif

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
struct Opt {
    std::string mode = "timing";         // timing | fairness
    std::string series = "all";          // scale,contention,cs,strong,oversub | all
    int ops = 2000;                      // lock rounds per rank (timing)
    int duration_ms = 1000;              // fairness: length of one run
    int trials = 10;
    int warmup = 1;                      // discarded pass per configuration
    int cs_work = 50;                    // busy-work iterations inside the critical section
    std::vector<std::string> locks;
    bool list = false, help = false, header = true;
    double timeout_s = 60.0;             // soft cap: skip larger rank counts of a lock once exceeded
};

static volatile std::uint64_t g_sink = 0;
static void busy(int n) {
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
    g_sink = g_sink + x;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
static double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}
static double median(const std::vector<double>& v) { return quantile(v, 0.5); }
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
static double pct_u32(std::vector<std::uint32_t>& v, double p) {
    if (v.empty()) return 0.0;
    const std::size_t k = std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return static_cast<double>(v[k]);
}

// ---------------------------------------------------------------------------
// Platform / MPI metadata (collected collectively, used by rank 0)
// ---------------------------------------------------------------------------
struct Meta {
    std::string mpi_lib, mpi_ver, cpu, kernel, compiler, flags, model;
    int world = 1, nodes = 1, rpn_max = 1, logical = 1, physical = 0, aff_min = 0, aff_max = 0;
};

static std::string trim(std::string s) {
    const char* ws = " \t\r\n\"";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
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
        if ((key == "Model" || key == "Hardware") && fallback.empty()) fallback = trim(line.substr(colon + 1));
    }
    if (!fallback.empty()) return fallback;
#elif defined(__APPLE__)
    char buf[256];
    std::size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) return trim(buf);
#endif
    return "unknown";
}

// CPUs this process may run on (affinity mask), and physical cores among them (Linux).
static void cpu_counts(int& allowed, int& physical) {
    allowed = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    physical = 0;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = 0, prim = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            ++n;
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            int first = c;
            if (f) {
                std::string s;
                std::getline(f, s);
                first = std::atoi(s.c_str());
            }
            if (first == c) ++prim;
        }
        if (n > 0) allowed = n;
        physical = prim;
    }
#elif defined(__APPLE__)
    int v = 0;
    std::size_t len = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &len, nullptr, 0) == 0) physical = v;
#endif
}

static std::string kernel_string() {
#if !defined(_WIN32)
    struct utsname u;
    if (uname(&u) == 0) return std::string(u.sysname) + " " + u.release;
#endif
    return "unknown";
}

static std::string compiler_name() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return "unknown";
#endif
}

static bool built_optimised() {
#if defined(__OPTIMIZE__) || defined(NDEBUG)
    return true;
#else
    return false;
#endif
}

static Meta collect_meta(MPI_Comm world) {
    Meta m;
    int rank;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &m.world);
    char lib[MPI_MAX_LIBRARY_VERSION_STRING];
    int len = 0;
    MPI_Get_library_version(lib, &len);
    m.mpi_lib = trim(std::string(lib, static_cast<std::size_t>(len)).substr(0, std::string(lib, static_cast<std::size_t>(len)).find('\n')));
    int ver = 0, sub = 0;
    MPI_Get_version(&ver, &sub);
    m.mpi_ver = std::to_string(ver) + "." + std::to_string(sub);

    // node layout
    char name[MPI_MAX_PROCESSOR_NAME] = {0};
    int nlen = 0;
    MPI_Get_processor_name(name, &nlen);
    std::vector<char> all(static_cast<std::size_t>(MPI_MAX_PROCESSOR_NAME) * static_cast<std::size_t>(m.world));
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, world);
    if (rank == 0) {
        std::vector<std::string> names;
        for (int i = 0; i < m.world; ++i) names.emplace_back(all.data() + static_cast<std::size_t>(i) * MPI_MAX_PROCESSOR_NAME);
        std::set<std::string> uniq(names.begin(), names.end());
        m.nodes = static_cast<int>(uniq.size());
        int best = 0;
        for (const auto& u : uniq) best = std::max(best, static_cast<int>(std::count(names.begin(), names.end(), u)));
        m.rpn_max = best;
    }
    // affinity: how many CPUs each rank may use (1 everywhere = bound to a core)
    int allowed = 1, phys = 0;
    cpu_counts(allowed, phys);
    MPI_Allreduce(&allowed, &m.aff_min, 1, MPI_INT, MPI_MIN, world);
    MPI_Allreduce(&allowed, &m.aff_max, 1, MPI_INT, MPI_MAX, world);
    m.logical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    m.physical = phys;
    m.cpu = cpu_model();
    m.kernel = kernel_string();
    m.compiler = compiler_name();
    m.flags = BENCH_FLAGS;
    return m;
}

static const char* kMetaHeader =
    "lock_params,mpi_library,mpi_version,world_ranks,nodes,ranks_per_node_max,cpu_model,logical_cpus,"
    "physical_cores,kernel,compiler,flags,affinity_cpus_min,affinity_cpus_max,rma_memory_model";

static std::string q(const std::string& s) {
    std::string out = "\"";
    for (char c : s) out += (c == '"') ? '\'' : c;
    return out + "\"";
}

static std::string meta_cols(const Meta& m, const std::string& params) {
    std::ostringstream o;
    o << q(params) << ',' << q(m.mpi_lib) << ',' << m.mpi_ver << ',' << m.world << ',' << m.nodes << ',' << m.rpn_max
      << ',' << q(m.cpu) << ',' << m.logical << ',' << m.physical << ',' << q(m.kernel) << ',' << q(m.compiler) << ','
      << q(m.flags) << ',' << m.aff_min << ',' << m.aff_max << ',' << m.model;
    return o.str();
}

// ---------------------------------------------------------------------------
// Lock parameters taken from the lock types themselves (no drift from the code)
// ---------------------------------------------------------------------------
template <int N> struct Rank : Rank<N - 1> {};
template <> struct Rank<0> {};

template <class B>
static std::string backoff_desc() {
    std::string s = std::string("backoff=") + B::kKind;
    if (B::kMax != 0) s += " " + std::to_string(B::kMin) + ".." + std::to_string(B::kMax) + " " + B::kUnit;
    return s + " (" + B::kNote + ")";
}
template <class L>
auto params_impl(Rank<2>) -> decltype((void)sizeof(typename L::backoff_type), std::string()) {
    return backoff_desc<typename L::backoff_type>();
}
template <class L>
auto params_impl(Rank<1>) -> decltype((void)L::pause_ns, std::string()) {
    return L::pause_ns == 0 ? "poll serving counter (remote read) with SpinWait(" + std::to_string(DLOCK_SPIN_LIMIT) + " polls, then yield)"
                            : "proportional backoff " + std::to_string(L::pause_ns) +
                                  " ns per position in line (cap 100000 ns)";
}
template <class L>
std::string params_impl(Rank<0>) {
    return "";
}

// ---------------------------------------------------------------------------
// Timing run
// ---------------------------------------------------------------------------
struct Result { double seconds; bool valid; };

// ops: rounds per rank (fixed work per rank) or, if strong, the total number of rounds split over the ranks.
template <class L>
Result run_one(MPI_Comm c, long ops, bool strong, int cs_work, int out_work) {
    int me, n;
    MPI_Comm_rank(c, &me);
    MPI_Comm_size(c, &n);
    const long mine = strong ? ops / n + (me < ops % n ? 1 : 0) : ops;
    const std::int64_t expected = strong ? ops : static_cast<std::int64_t>(n) * ops;
    Window data(c, 1);
    L lock(c);
    polite_barrier(c);
    MPI_Barrier(c);
    const double t0 = MPI_Wtime();
    for (long i = 0; i < mine; ++i) {
        if (out_work) busy(out_work);
        lock.acquire();
        const std::int64_t v = data.get(0, 0);
        busy(cs_work);
        data.put(0, 0, v + 1);
        lock.release();
    }
    MPI_Barrier(c);
    double dt = MPI_Wtime() - t0, mx = 0;
    MPI_Allreduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, c);
    const std::int64_t fin = data.get(0, 0);
    int ok = (fin == expected), all = 1;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, c);
    return {mx, all != 0};
}

// ---------------------------------------------------------------------------
// Fairness run: fixed duration, private counters, sampled acquire() latency
// ---------------------------------------------------------------------------
constexpr std::int64_t kSampleEvery = 32;
constexpr std::size_t kMaxSamples = 1u << 15;  // per rank

struct FairOut {            // meaningful on rank 0 of the sub-communicator
    bool valid = true;
    double region_s = 0;
    std::vector<std::int64_t> counts;
    std::vector<std::uint32_t> waits;
    double rank_p99_min = 0, rank_p99_max = 0;
};

template <class L>
FairOut run_fair(MPI_Comm c, int duration_ms, int cs_work, int out_work) {
    int me, n;
    MPI_Comm_rank(c, &me);
    MPI_Comm_size(c, &n);
    Window data(c, 1);
    L lock(c);
    std::vector<std::uint32_t> samples;
    samples.reserve(kMaxSamples);
    polite_barrier(c);
    MPI_Barrier(c);
    const double t0 = MPI_Wtime();
    const double tend = t0 + duration_ms / 1000.0;
    std::int64_t cnt = 0;
    while (MPI_Wtime() < tend) {
        if (out_work) busy(out_work);
        const bool sample = (cnt % kSampleEvery == 0) && samples.size() < kMaxSamples;
        double ts = 0;
        if (sample) ts = MPI_Wtime();
        lock.acquire();
        if (sample) {
            const double w = (MPI_Wtime() - ts) * 1e9;
            samples.push_back(static_cast<std::uint32_t>(std::min(w, 4.29e9)));
        }
        const std::int64_t v = data.get(0, 0);
        busy(cs_work);
        data.put(0, 0, v + 1);
        lock.release();
        ++cnt;
    }
    polite_barrier(c);  // ranks that finished early sleep instead of spinning
    const double dt = MPI_Wtime() - t0;
    double region = 0;
    MPI_Allreduce(&dt, &region, 1, MPI_DOUBLE, MPI_MAX, c);

    std::int64_t total = 0;
    MPI_Allreduce(&cnt, &total, 1, MPI_INT64_T, MPI_SUM, c);
    const std::int64_t fin = data.get(0, 0);
    const int ok = (fin == total);

    FairOut out;
    out.valid = ok != 0;
    out.region_s = region;
    // gather after the timed region
    std::vector<std::int64_t> counts(me == 0 ? n : 0);
    MPI_Gather(&cnt, 1, MPI_INT64_T, counts.data(), 1, MPI_INT64_T, 0, c);
    double mine_p99 = 0;
    {
        std::vector<std::uint32_t> t = samples;
        mine_p99 = pct_u32(t, 0.99);
    }
    std::vector<double> p99s(me == 0 ? n : 0);
    MPI_Gather(&mine_p99, 1, MPI_DOUBLE, p99s.data(), 1, MPI_DOUBLE, 0, c);
    int ns = static_cast<int>(samples.size());
    std::vector<int> sizes(me == 0 ? n : 0), displs(me == 0 ? n : 0);
    MPI_Gather(&ns, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, c);
    int totn = 0;
    if (me == 0)
        for (int i = 0; i < n; ++i) {
            displs[static_cast<std::size_t>(i)] = totn;
            totn += sizes[static_cast<std::size_t>(i)];
        }
    std::vector<std::uint32_t> all(me == 0 ? static_cast<std::size_t>(totn) : 0);
    // uint32 -> send as MPI_UNSIGNED
    MPI_Gatherv(samples.data(), ns, MPI_UNSIGNED, all.data(), sizes.data(), displs.data(), MPI_UNSIGNED, 0, c);
    if (me == 0) {
        out.counts = counts;
        out.waits = std::move(all);
        out.rank_p99_min = *std::min_element(p99s.begin(), p99s.end());
        out.rank_p99_max = *std::max_element(p99s.begin(), p99s.end());
    }
    return out;
}

struct FairMetrics {
    double min = 0, med = 0, max = 0, ratio = 0, jain = 0, p50 = 0, p99 = 0, pmax = 0;
    std::int64_t total = 0;
};
static FairMetrics fair_metrics(FairOut& f) {
    FairMetrics m;
    std::vector<double> c(f.counts.begin(), f.counts.end());
    double sum = 0, sq = 0;
    for (double x : c) { sum += x; sq += x * x; }
    m.total = static_cast<std::int64_t>(sum);
    m.min = *std::min_element(c.begin(), c.end());
    m.max = *std::max_element(c.begin(), c.end());
    m.med = quantile(c, 0.5);
    m.ratio = m.min > 0 ? m.max / m.min : INFINITY;
    m.jain = sq > 0 ? (sum * sum) / (static_cast<double>(c.size()) * sq) : 0.0;
    m.p50 = pct_u32(f.waits, 0.50);
    m.p99 = pct_u32(f.waits, 0.99);
    m.pmax = f.waits.empty() ? 0.0 : static_cast<double>(*std::max_element(f.waits.begin(), f.waits.end()));
    return m;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
using RunFn = Result (*)(MPI_Comm, long, bool, int, int);
using FairFn = FairOut (*)(MPI_Comm, int, int, int);
struct Entry { const char* name; RunFn run; FairFn fair; std::string params; };

template <class L>
static Entry mk(const char* name, const std::string& extra = "") {
    std::string p = params_impl<L>(Rank<2>{});
    if (p.empty()) p = extra;
    else if (!extra.empty()) p += "; " + extra;
    return Entry{name, &run_one<L>, &run_fair<L>, p};
}

static const std::vector<Entry>& registry() {
    static const std::vector<Entry> r = [] {
        const std::string home = "state in an RMA window on rank 0, passive target, flush after every operation";
        const std::string spin = "local polling with SpinWait(" + std::to_string(DLOCK_SPIN_LIMIT) +
                                 " polls, then yield), MPI progress poll every " + std::to_string(DLOCK_PROGRESS_EVERY) + " polls";
        std::vector<Entry> v;
        v.push_back(mk<lock_tas>("tas", home));
        v.push_back(mk<lock_tas_pause>("tas-pause", home));
        v.push_back(mk<lock_tas_sleep>("tas-sleep", home));
        v.push_back(mk<lock_ttas>("ttas", home));
        v.push_back(mk<lock_ttas_pause>("ttas-pause", home));
        v.push_back(mk<lock_ttas_sleep>("ttas-sleep", home));
        v.push_back(mk<lock_ticket>("ticket", home));
        v.push_back(mk<lock_ticket_prop>("ticket-prop", home));
        v.push_back(mk<lock_mcs>("mcs", "queue lock (Hoefler et al.): tail on rank 0, per-rank next/locked slots; " + spin));
        v.push_back(mk<lock_winlock>("winlock", "MPI_Win_lock(EXCLUSIVE) + flush, behaviour is implementation-defined"));
        return v;
    }();
    return r;
}

static std::vector<std::string> split(const std::string& s, char sep = ',') {
    std::vector<std::string> o;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) { if (!cur.empty()) o.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) o.push_back(cur);
    return o;
}

static void print_help() {
    std::cout <<
        "Usage: mpirun -n P ./bench_mpi_mutex [options]\n"
        "  --mode=timing|fairness  timing (default) or the fixed-duration fairness experiment\n"
        "  --series=a,b|all        timing: scale, contention, cs, strong, oversub (default all)\n"
        "  --ops=N                 rounds per rank (fixed work); strong scaling splits ops*P rounds over the ranks\n"
        "  --duration-ms=T         fairness: length of one run (default 1000)\n"
        "  --trials=N              measured repetitions per configuration (default 10)\n"
        "  --warmup=N              discarded passes per configuration (default 1)\n"
        "  --cs-work=N             medium busy-work iterations inside the critical section (default 50)\n"
        "  --lock=a,b              only these locks (see --list)\n"
        "  --timeout=SEC           skip larger rank counts of a lock once a run exceeds SEC (default 60)\n"
        "  --no-header             omit the CSV header line (to append to an existing file)\n"
        "  --list, --help\n"
        "Series (OUTSIDE_WORK k means k x the medium critical-section work; CS = busy work inside the lock):\n"
        "  scale       k = 0,2,8; ranks 1,2,4,.. up to the logical CPUs, plus the physical-core count\n"
        "  contention  k = 0,1,2,4,8,16; ranks 1, physical, logical\n"
        "  cs          CS work 0, medium, 100x medium; k = 0,8; ranks 1, physical, logical\n"
        "  strong      fixed total rounds, k = 2; same ranks as scale\n"
        "  oversub     k = 0,8; ranks = logical CPUs and 2x that (needs -n >= 2x logical, single node)\n";
}

struct MCfg {
    std::string mode;  // fixed | strong | fairness
    int ranks, cs, k, ow;
    std::string exp;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Opt o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* k) -> const char* {
            std::string p = std::string(k) + "=";
            return a.compare(0, p.size(), p) == 0 ? a.c_str() + p.size() : nullptr;
        };
        const char* v;
        if (a == "--help") o.help = true;
        else if (a == "--list") o.list = true;
        else if (a == "--no-header") o.header = false;
        else if ((v = val("--mode"))) o.mode = v;
        else if ((v = val("--series"))) o.series = v;
        else if ((v = val("--ops"))) o.ops = std::max(1, atoi(v));
        else if ((v = val("--duration-ms"))) o.duration_ms = std::max(50, atoi(v));
        else if ((v = val("--trials"))) o.trials = std::max(1, atoi(v));
        else if ((v = val("--warmup"))) o.warmup = std::max(0, atoi(v));
        else if ((v = val("--cs-work"))) o.cs_work = std::max(0, atoi(v));
        else if ((v = val("--lock"))) o.locks = split(v);
        else if ((v = val("--timeout"))) o.timeout_s = atof(v);
        else { if (!rank) std::cerr << "unknown option " << a << "\n"; MPI_Finalize(); return 64; }
    }
    if (o.mode != "timing" && o.mode != "fairness") {
        if (!rank) std::cerr << "--mode must be timing or fairness\n";
        MPI_Finalize();
        return 64;
    }
    if (o.help || o.list) {
        if (!rank) {
            if (o.list) for (auto& e : registry()) std::cout << e.name << "\n";
            else print_help();
        }
        MPI_Finalize();
        return 0;
    }
    const bool fairness = (o.mode == "fairness");

    std::vector<Entry> sel;
    for (auto& e : registry())
        if (o.locks.empty() || std::find(o.locks.begin(), o.locks.end(), e.name) != o.locks.end()) sel.push_back(e);
    if (sel.empty()) { if (!rank) std::cerr << "no matching lock (see --list)\n"; MPI_Finalize(); return 64; }

    Meta meta = collect_meta(MPI_COMM_WORLD);
    {
        Window probe(MPI_COMM_WORLD, 1);
        const int mm = probe.memory_model();
        meta.model = mm == 1 ? "unified" : mm == 0 ? "separate" : "unknown";
    }

    // ---- rank sets -------------------------------------------------------
    // "logical" = logical CPUs of this node (single node) or all ranks (several nodes);
    // physical cores are only meaningful on one node.
    const bool one_node = meta.nodes == 1;
    const int R_log = one_node ? std::min(size, meta.logical) : size;
    const int R_phys = (one_node && meta.physical > 0 && meta.physical < R_log) ? meta.physical : 0;
    std::set<int> sc{1};
    for (int t = 2; t <= R_log; t *= 2) sc.insert(t);
    sc.insert(R_log);
    if (R_phys) sc.insert(R_phys);
    std::set<int> r3{1, R_log};
    if (R_phys) r3.insert(R_phys);

    // ---- configurations ---------------------------------------------------
    std::vector<MCfg> cfgs;
    auto add = [&](const std::string& exp, const std::string& mode, int ranks, int cs, int k, int ow) {
        for (auto& c : cfgs)
            if (c.mode == mode && c.ranks == ranks && c.cs == cs && c.ow == ow) {
                if (("+" + c.exp + "+").find("+" + exp + "+") == std::string::npos) c.exp += "+" + exp;
                return;
            }
        cfgs.push_back({mode, ranks, cs, k, ow, exp});
    };
    const int unit = std::max(1, o.cs_work);  // OUTSIDE_WORK k = k x medium critical-section work
    std::string note;
    if (fairness) {
        for (int r : sc) if (r >= 2 && r <= size) add("fairness", "fairness", r, o.cs_work, 0, 0);  // sustained contention
    } else {
        std::set<std::string> want;
        for (const auto& t : split(o.series)) {
            if (t == "all") { for (const char* n : {"scale", "contention", "cs", "strong", "oversub"}) want.insert(n); }
            else if (t == "scale" || t == "contention" || t == "cs" || t == "strong" || t == "oversub") want.insert(t);
            else { if (!rank) std::cerr << "Unknown series '" << t << "'\n"; MPI_Finalize(); return 64; }
        }
        if (want.count("scale")) for (int k : {0, 2, 8}) for (int r : sc) add("scale", "fixed", r, o.cs_work, k, k * unit);
        if (want.count("contention")) for (int k : {0, 1, 2, 4, 8, 16}) for (int r : r3) add("contention", "fixed", r, o.cs_work, k, k * unit);
        if (want.count("cs")) for (int cs : {0, o.cs_work, 100 * o.cs_work}) for (int k : {0, 8}) for (int r : r3) add("cs", "fixed", r, cs, k, k * unit);
        if (want.count("strong")) for (int r : sc) add("strong", "strong", r, o.cs_work, 2, 2 * unit);
        if (want.count("oversub")) {
            if (one_node && size >= 2 * R_log && R_log >= 1) {
                for (int k : {0, 8}) for (int r : {R_log, 2 * R_log}) add("oversub", "fixed", r, o.cs_work, k, k * unit);
            } else {
                note = one_node ? "oversub series skipped: launch with -n " + std::to_string(2 * R_log) + " (and --oversubscribe) to include it"
                                : "oversub series skipped: it needs a single node";
            }
        }
    }
    std::stable_sort(cfgs.begin(), cfgs.end(), [](const MCfg& a, const MCfg& b) {
        return std::tie(a.mode, a.cs, a.ow, a.ranks) < std::tie(b.mode, b.cs, b.ow, b.ranks);
    });
    if (cfgs.empty()) { if (!rank) std::cerr << "nothing to run\n"; MPI_Finalize(); return 64; }
    for (auto& c : cfgs) c.ranks = std::min(c.ranks, size);

    if (!rank) {
        std::cerr << "============================================================\n"
                  << (fairness ? "MPI RMA mutex fairness experiment (fixed duration, per-rank counters)\n"
                               : "MPI RMA mutex benchmark\n")
                  << "============================================================\n";
        std::cerr << "MPI library    : " << meta.mpi_lib << " (MPI " << meta.mpi_ver << ")\n";
        std::cerr << "Ranks / nodes  : " << meta.world << " / " << meta.nodes << " (max " << meta.rpn_max << " per node)\n";
        std::cerr << "CPU            : " << meta.cpu << "\n";
        std::cerr << "Cores (rank 0) : " << (meta.physical > 0 ? std::to_string(meta.physical) : std::string("unknown"))
                  << " physical / " << meta.logical << " logical\n";
        std::cerr << "Affinity       : each rank may use " << meta.aff_min << ".." << meta.aff_max
                  << " CPUs (1 = bound to a core; use the launcher's binding option, see README)\n";
        std::cerr << "OS / kernel    : " << meta.kernel << "\n";
        std::cerr << "Compiler       : " << meta.compiler << "\n";
        std::cerr << "Flags          : " << meta.flags << "\n";
        std::cerr << "RMA model      : " << meta.model << "\n";
        std::cerr << "Lock parameters:\n";
        for (auto& e : sel) std::cerr << "  " << e.name << std::string(14 - std::min<std::size_t>(13, std::strlen(e.name)), ' ') << e.params << "\n";
        if (!fairness) {
            std::map<std::string, int> per;
            for (const auto& c : cfgs) for (const auto& t : split(c.exp, '+')) ++per[t];
            std::cerr << "Series         : ";
            bool first = true;
            for (const char* n : {"scale", "contention", "cs", "strong", "oversub"})
                if (per.count(n)) { std::cerr << (first ? "" : ", ") << n << " (" << per[n] << " configs)"; first = false; }
            std::cerr << "\n";
        }
        std::set<int> rs, css, ows;
        for (const auto& c : cfgs) { rs.insert(c.ranks); css.insert(c.cs); ows.insert(c.ow); }
        std::cerr << "Ranks          : ";
        for (int r : rs) std::cerr << r << " ";
        std::cerr << "(logical " << R_log << ", physical " << (R_phys ? std::to_string(R_phys) : std::string("unknown/n.a.")) << ")\n";
        std::cerr << "CS work        : ";
        for (int c : css) std::cerr << c << " ";
        std::cerr << "busy-work iterations on a remote counter (the RMA round trips dominate); outside work: ";
        for (int w : ows) std::cerr << w << " ";
        std::cerr << "\n";
        if (fairness) std::cerr << "Duration       : " << o.duration_ms << " ms per run; acquire() timed on every " << kSampleEvery << "th acquisition\n";
        else std::cerr << "Rounds         : " << o.ops << " per rank (strong scaling: " << static_cast<long>(o.ops) * R_log << " in total)\n";
        std::cerr << "Trials         : " << o.trials << " (+" << o.warmup << " warm-up pass per configuration)\n";
        std::cerr << "Plan           : " << cfgs.size() << " configurations x " << sel.size() * (o.trials + o.warmup) << " runs\n";
        if (!note.empty()) std::cerr << "Note: " << note << "\n";
        if (!built_optimised()) std::cerr << "WARNING: built without optimisation (-O2 expected for performance runs).\n";
        if (o.trials < 5) std::cerr << "Note: fewer than 5 trials: the IQR is not meaningful (smoke test only).\n";
        if (meta.nodes == 1 && meta.world > meta.logical) std::cerr << "Note: more ranks than logical CPUs: oversubscribed, expect slow spin locks.\n";
        std::cerr << std::flush;
        if (o.header) {
            if (fairness)
                std::cout << "lock,ranks,cs_work,out_work,duration_ms,trial,region_s,valid,total_acq,min_acq,median_acq,max_acq,"
                             "max_min_ratio,jain,wait_p50_ns,wait_p99_ns,wait_max_ns,rank_p99_min_ns,rank_p99_max_ns,"
                             "wait_samples,"
                          << kMetaHeader << ",experiment,per_rank_acq\n";
            else
                std::cout << "lock,ranks,mode,cs_work,out_work,outside_k,trial,total_ops,seconds,ops_per_sec,valid," << kMetaHeader
                          << ",experiment\n";
        }
    }
    if (o.help) { MPI_Finalize(); return 0; }

    std::vector<std::vector<char>> skipped;  // per lock: keyed by (mode, cs, ow)
    std::set<std::tuple<size_t, std::string, int, int>> skip;
    const long strong_total = static_cast<long>(o.ops) * std::max(1, R_log);
    std::mt19937 g(12345u);
    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        const MCfg& c = cfgs[ci];
        MPI_Comm sub;
        MPI_Comm_split(MPI_COMM_WORLD, rank < c.ranks ? 0 : MPI_UNDEFINED, rank, &sub);
        const bool strong = c.mode == "strong";
        std::vector<std::vector<double>> secs(sel.size()), rate(sel.size()), jains(sel.size()), p99s(sel.size()),
            ratios(sel.size()), minacq(sel.size()), maxacq(sel.size());

        const int passes = o.warmup + o.trials;
        for (int t = 0; t < passes; ++t) {
            const bool warm = t < o.warmup;
            const int trial = t - o.warmup + 1;
            // Block-randomised lock order (same permutation on every rank: identical seed and engine).
            std::vector<size_t> order(sel.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::shuffle(order.begin(), order.end(), g);
            for (size_t li : order) {
                const auto key = std::make_tuple(li, c.mode, c.cs, c.ow);
                const int skipv = skip.count(key) ? 1 : 0;
                polite_barrier(MPI_COMM_WORLD);
                if (skipv) continue;
                double s = 0;
                int valid = 1;
                if (!fairness) {
                    Result r{0, true};
                    const long ops = strong ? strong_total : o.ops;
                    if (sub != MPI_COMM_NULL) r = sel[li].run(sub, ops, strong, c.cs, c.ow);
                    polite_barrier(MPI_COMM_WORLD);
                    s = r.seconds;
                    valid = r.valid;
                    MPI_Bcast(&s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (!warm && !rank) {
                        const long total = strong ? strong_total : static_cast<long>(c.ranks) * o.ops;
                        const double ops_s = static_cast<double>(total) / s;
                        std::printf("%s,%d,%s,%d,%d,%d,%d,%ld,%.6f,%.1f,%d,%s,%s\n", sel[li].name, c.ranks, c.mode.c_str(), c.cs, c.ow,
                                    c.k, trial, total, s, ops_s, valid, meta_cols(meta, sel[li].params).c_str(), c.exp.c_str());
                        std::fflush(stdout);
                        if (!valid) std::fprintf(stderr, "[INVALID] %s ranks=%d: counter mismatch\n", sel[li].name, c.ranks);
                        if (valid) { secs[li].push_back(s); rate[li].push_back(ops_s); }
                    }
                } else {
                    FairOut f;
                    if (sub != MPI_COMM_NULL) f = sel[li].fair(sub, o.duration_ms, c.cs, c.ow);
                    polite_barrier(MPI_COMM_WORLD);
                    s = f.region_s;
                    valid = f.valid;
                    MPI_Bcast(&s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (!warm && !rank) {
                        const FairMetrics m = fair_metrics(f);
                        std::string per;
                        for (size_t i = 0; i < f.counts.size(); ++i) per += (i ? ";" : "") + std::to_string(f.counts[i]);
                        std::printf("%s,%d,%d,%d,%d,%d,%.4f,%d,%lld,%.0f,%.1f,%.0f,%.4f,%.4f,%.0f,%.0f,%.0f,%.0f,%.0f,%zu,%s,%s,%s\n",
                                    sel[li].name, c.ranks, c.cs, c.ow, o.duration_ms, trial, s, valid, static_cast<long long>(m.total),
                                    m.min, m.med, m.max, m.ratio, m.jain, m.p50, m.p99, m.pmax, f.rank_p99_min, f.rank_p99_max,
                                    f.waits.size(), meta_cols(meta, sel[li].params).c_str(), c.exp.c_str(), per.c_str());
                        std::fflush(stdout);
                        if (!valid) std::fprintf(stderr, "[INVALID] %s ranks=%d: counter mismatch\n", sel[li].name, c.ranks);
                        if (valid) {
                            jains[li].push_back(m.jain);
                            ratios[li].push_back(m.ratio);
                            p99s[li].push_back(m.p99);
                            minacq[li].push_back(m.min);
                            maxacq[li].push_back(m.max);
                        }
                    }
                }
                if (!warm && s > o.timeout_s) skip.insert(key);  // identical on all ranks (s was broadcast)
            }
        }

        if (!rank) {
            std::fprintf(stderr, "\n[%zu/%zu] [%s] mode=%s ranks=%d cs_work=%d out_work=%d\n", ci + 1, cfgs.size(), c.exp.c_str(),
                         c.mode.c_str(), c.ranks, c.cs, c.ow);
            if (!fairness) {
                std::fprintf(stderr, "  %-14s %12s %10s %10s %10s %14s\n", "lock", "median s", "iqr", "min", "sd", "median ops/s");
                for (size_t li = 0; li < sel.size(); ++li) {
                    if (secs[li].empty()) { std::fprintf(stderr, "  %-14s %12s\n", sel[li].name, "-"); continue; }
                    std::fprintf(stderr, "  %-14s %12.5f %10.5f %10.5f %10.5f %14.0f\n", sel[li].name, median(secs[li]),
                                 iqr(secs[li]), *std::min_element(secs[li].begin(), secs[li].end()), stddev(secs[li]),
                                 median(rate[li]));
                }
            } else {
                std::fprintf(stderr, "  %-14s %10s %10s %9s %8s %8s %13s\n", "lock", "min acq", "max acq", "max/min", "Jain",
                             "(iqr)", "p99 wait ns");
                for (size_t li = 0; li < sel.size(); ++li) {
                    if (jains[li].empty()) { std::fprintf(stderr, "  %-14s %10s\n", sel[li].name, "-"); continue; }
                    std::fprintf(stderr, "  %-14s %10.0f %10.0f %9.2f %8.3f %8.3f %13.0f\n", sel[li].name, median(minacq[li]),
                                 median(maxacq[li]), median(ratios[li]), median(jains[li]), iqr(jains[li]), median(p99s[li]));
                }
                std::cerr << "  (medians over the trials; (iqr) is the IQR of Jain's index. Observed behaviour, not a guarantee.)\n";
            }
            std::cerr << std::flush;
        }
        if (sub != MPI_COMM_NULL) MPI_Comm_free(&sub);
    }
    MPI_Finalize();
    return 0;
}