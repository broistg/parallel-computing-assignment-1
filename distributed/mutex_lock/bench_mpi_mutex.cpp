// ============================================================================
// bench_mpi_mutex.cpp -- throughput/scalability benchmark for mpi_lock.h
//
//   Build : mpicxx -std=c++17 -O2 bench_mpi_mutex.cpp -o bench_mpi_mutex
//   Run   : mpirun -n 4 ./bench_mpi_mutex [--help]
//
// Each rank performs a fixed number of lock/unlock rounds. The critical section
// reads and writes a counter stored on rank 0 (non-atomic Get/Put, completed
// before release) so that the lock protects real remote data. Timer = whole
// parallel region (barrier ... barrier), maximum over ranks. The result is
// validated (counter == ranks * ops) before a row is emitted. Output: CSV on
// rank 0 (stdout), human summary on stderr.
// ============================================================================
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "mpi_lock.h"

using namespace dlock;

struct Opt {
    int ops = 2000;                      // lock rounds per rank
    int trials = 5;
    int cs_work = 50;                    // busy-work iterations inside the critical section
    std::vector<int> out_work{0, 500};   // busy-work iterations outside
    std::vector<int> ranks;
    std::vector<std::string> locks;
    bool quick = false, list = false, help = false, header = true;
    double timeout_s = 60.0;             // soft cap: skip larger rank counts of a lock once exceeded
};

static volatile std::uint64_t g_sink = 0;
static void busy(int n) {
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
    g_sink = g_sink + x;
}

struct Result { double seconds; bool valid; };

template <class L>
Result run_one(MPI_Comm c, int ops, int cs_work, int out_work) {
    int me, n;
    MPI_Comm_rank(c, &me);
    MPI_Comm_size(c, &n);
    Window data(c, 1);
    L lock(c);
    polite_barrier(c);
    MPI_Barrier(c);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < ops; ++i) {
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
    int ok = (fin == static_cast<std::int64_t>(n) * ops), all = 1;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, c);
    return {mx, all != 0};
}

using RunFn = Result (*)(MPI_Comm, int, int, int);
struct Entry { const char* name; RunFn run; };
static const std::vector<Entry>& registry() {
    static const std::vector<Entry> r = {
        {"tas", &run_one<lock_tas>},         {"tas-pause", &run_one<lock_tas_pause>},
        {"tas-sleep", &run_one<lock_tas_sleep>}, {"ttas", &run_one<lock_ttas>},
        {"ttas-pause", &run_one<lock_ttas_pause>}, {"ttas-sleep", &run_one<lock_ttas_sleep>},
        {"ticket", &run_one<lock_ticket>},   {"ticket-prop", &run_one<lock_ticket_prop>},
        {"mcs", &run_one<lock_mcs>},         {"winlock", &run_one<lock_winlock>},
    };
    return r;
}

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> o; std::string cur;
    for (char ch : s) { if (ch == ',') { if (!cur.empty()) o.push_back(cur); cur.clear(); } else cur += ch; }
    if (!cur.empty()) o.push_back(cur);
    return o;
}

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
        else if (a == "--quick") { o.quick = true; o.ops = 300; o.trials = 2; o.out_work = {0}; }
        else if (a == "--no-header") o.header = false;
        else if ((v = val("--ops"))) o.ops = std::max(1, atoi(v));
        else if ((v = val("--trials"))) o.trials = std::max(1, atoi(v));
        else if ((v = val("--cs-work"))) o.cs_work = std::max(0, atoi(v));
        else if ((v = val("--out-work"))) { o.out_work.clear(); for (auto& s : split(v)) o.out_work.push_back(atoi(s.c_str())); }
        else if ((v = val("--ranks"))) { for (auto& s : split(v)) o.ranks.push_back(std::max(1, atoi(s.c_str()))); }
        else if ((v = val("--lock"))) o.locks = split(v);
        else if ((v = val("--timeout"))) o.timeout_s = atof(v);
        else { if (!rank) std::cerr << "unknown option " << a << "\n"; MPI_Finalize(); return 64; }
    }
    if (o.help || o.list) {
        if (!rank) {
            if (o.list) for (auto& e : registry()) std::cout << e.name << "\n";
            else std::cout << "Usage: mpirun -n P ./bench_mpi_mutex [--ops=N] [--trials=N] [--cs-work=N]\n"
                              "       [--out-work=0,500] [--ranks=1,2,4] [--lock=a,b] [--timeout=SEC] [--quick] [--list]\n";
        }
        MPI_Finalize();
        return 0;
    }
    if (o.ranks.empty()) { std::set<int> s{1}; for (int t = 2; t <= size; t *= 2) s.insert(t); s.insert(size); o.ranks.assign(s.begin(), s.end()); }
    for (int& k : o.ranks) k = std::min(k, size);
    std::sort(o.ranks.begin(), o.ranks.end());
    o.ranks.erase(std::unique(o.ranks.begin(), o.ranks.end()), o.ranks.end());

    std::vector<Entry> sel;
    for (auto& e : registry()) {
        if (o.locks.empty() || std::find(o.locks.begin(), o.locks.end(), e.name) != o.locks.end()) sel.push_back(e);
    }
    if (sel.empty()) { if (!rank) std::cerr << "no matching lock (see --list)\n"; MPI_Finalize(); return 64; }

    if (!rank && o.header)
        std::cout << "lock,ranks,cs_work,out_work,trial,ops_per_rank,seconds,ops_per_sec,valid\n";

    std::vector<char> skip(sel.size() * 1024, 0);  // (lock, outwork) pairs that exceeded the soft timeout
    for (size_t wi = 0; wi < o.out_work.size(); ++wi) {
        for (int k : o.ranks) {
            MPI_Comm sub;
            MPI_Comm_split(MPI_COMM_WORLD, rank < k ? 0 : MPI_UNDEFINED, rank, &sub);
            for (int t = 1; t <= o.trials; ++t) {
                // Block-randomised lock order (same permutation on every rank: seeded identically).
                std::vector<size_t> order(sel.size());
                for (size_t i = 0; i < order.size(); ++i) order[i] = i;
                std::mt19937 g(12345u + 977u * t + 31u * k);
                std::shuffle(order.begin(), order.end(), g);
                for (size_t li : order) {
                    int skipv = skip[li * 1024 + wi];
                    polite_barrier(MPI_COMM_WORLD);
                    if (skipv) continue;
                    Result r{0, true};
                    if (sub != MPI_COMM_NULL) r = sel[li].run(sub, o.ops, o.cs_work, o.out_work[wi]);
                    polite_barrier(MPI_COMM_WORLD);
                    double s = r.seconds;
                    int valid = r.valid;
                    MPI_Bcast(&s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (!rank) {
                        std::printf("%s,%d,%d,%d,%d,%d,%.6f,%.1f,%d\n", sel[li].name, k, o.cs_work, o.out_work[wi], t,
                                    o.ops, s, (double)k * o.ops / s, valid);
                        std::fflush(stdout);
                        if (!valid) std::fprintf(stderr, "[INVALID] %s ranks=%d: counter mismatch\n", sel[li].name, k);
                    }
                    if (s > o.timeout_s) skip[li * 1024 + wi] = 1;  // identical on all ranks (s was broadcast)
                }
            }
            if (sub != MPI_COMM_NULL) MPI_Comm_free(&sub);
        }
    }
    MPI_Finalize();
    return 0;
}
