#ifndef MPI_LOCK_H
#define MPI_LOCK_H

// ============================================================================
// mpi_lock.h -- distributed-memory versions of the Section 1 locks, written
// with MPI-3 one-sided communication (RMA, passive target).
//
//   lock_tas / lock_ttas   : test-and-set / test-and-test-and-set (+ backoff)
//   lock_ticket            : FIFO ticket lock (+ proportional backoff variant)
//   lock_mcs               : FIFO queue lock, each rank spins on LOCAL memory
//   lock_winlock           : MPI's own MPI_Win_lock(EXCLUSIVE) as a baseline
//
// Process model: one MPI process per rank, each single-threaded (the library
// may be initialised with MPI_THREAD_SINGLE or MPI_THREAD_FUNNELED).
//
// Interface (same as the shared-memory locks, plus a communicator):
//     lock_xxx lk(comm [, home]);   // COLLECTIVE over comm: every rank of comm
//                                   // must construct (and later destroy) its
//                                   // locks in the same order
//     lk.acquire(); ... lk.release();   // also lock()/unlock()/try_lock()
//
// Rules the caller must follow:
//   * A lock must be released by the rank that acquired it.
//   * RMA operations on data protected by the lock must be COMPLETED (flush /
//     unlock) before release(); release() only orders the lock's own window.
//   * Locks must not be used recursively (a rank may hold several DIFFERENT
//     locks at once, and release them in any order).
//
// Every word that is accessed concurrently lives in a window and is accessed
// only through MPI atomics (Fetch_and_op, Compare_and_swap, Accumulate), except
// for a rank's own MCS flags, which are polled locally after MPI_Win_sync.
//
// Compile-time knobs: -DDLOCK_SPIN_LIMIT=N (polls before yielding),
//                     -DDLOCK_PROGRESS_EVERY=N (local polls between MPI
//                     progress calls; 0 disables; see README).
// Requires C++17 and an MPI-3 implementation.
// ============================================================================

#if defined(_MSVC_LANG)
  #if _MSVC_LANG < 201703L
    #error "mpi_lock.h requires C++17"
  #endif
#elif __cplusplus < 201703L
  #error "mpi_lock.h requires C++17 (compile with -std=c++17)"
#endif

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <thread>
#include <utility>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

#ifndef DLOCK_SPIN_LIMIT
  #define DLOCK_SPIN_LIMIT 2048
#endif
// Waiting loops that only touch local memory still call into MPI now and then,
// because some MPI implementations apply incoming RMA operations only while
// the target process is inside the library ("target-side progress").
#ifndef DLOCK_PROGRESS_EVERY
  #define DLOCK_PROGRESS_EVERY 64
#endif

#define DLOCK_MPI(call)                                                                  \
    do {                                                                                 \
        const int dlock_rc_ = (call);                                                    \
        if (dlock_rc_ != MPI_SUCCESS) {                                                  \
            std::fprintf(stderr, "dlock: MPI call failed (%s), code %d, at %s:%d\n", #call, \
                         dlock_rc_, __FILE__, __LINE__);                                 \
            MPI_Abort(MPI_COMM_WORLD, dlock_rc_);                                        \
        }                                                                                \
    } while (0)

namespace dlock {

// ----------------------------------------------------------------------------
// Local helpers: spin hint, hybrid waiting, backoff policies
// ----------------------------------------------------------------------------
inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

namespace detail {

inline std::uint32_t jitter_rand() noexcept {
    static thread_local std::uint32_t s = 0;
    if (s == 0) {
        s = (static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&s) >> 4) *
             2654435761u) | 1u;
    }
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

template <unsigned Max>
constexpr unsigned next_delay(unsigned d) noexcept {
    return (d > Max / 2) ? Max : d * 2;
}

// Busy-wait for about `ns` nanoseconds (used by proportional ticket backoff).
inline void spin_for_ns(std::int64_t ns) noexcept {
    const auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
    while (std::chrono::steady_clock::now() < end) cpu_relax();
}

}  // namespace detail

// Spin with cpu_relax() for Limit calls, then yield to the OS scheduler. The
// yield matters when there are more ranks than cores.
template <unsigned Limit = DLOCK_SPIN_LIMIT>
class SpinWait {
public:
    void operator()() noexcept {
        if (spins_ < Limit) {
            ++spins_;
            cpu_relax();
        } else {
            std::this_thread::yield();
        }
    }

private:
    unsigned spins_ = 0;
};

// A new policy object is created for each acquire(), so every acquisition
// starts at the minimum delay.
struct NoBackoff {
    void operator()() noexcept {}
};

// Exponential backoff with CPU pause hints and random jitter in [d/2, d].
// A retry costs a network round trip, so the pauses are short in absolute time.
template <unsigned MinSpins = 4, unsigned MaxSpins = 1024>
class PauseBackoff {
    static_assert(MinSpins > 0, "MinSpins must be > 0");
    static_assert(MinSpins <= MaxSpins, "MinSpins must be <= MaxSpins");

public:
    static constexpr unsigned next_delay(unsigned d) noexcept {
        return detail::next_delay<MaxSpins>(d);
    }
    void operator()() noexcept {
        const unsigned half = delay_ / 2;
        unsigned spins = half + detail::jitter_rand() % (delay_ - half + 1);
        if (spins == 0) spins = 1;
        for (unsigned i = 0; i < spins; ++i) cpu_relax();
        if (delay_ >= MaxSpins) {
            std::this_thread::yield();
        } else {
            delay_ = next_delay(delay_);
        }
    }
    unsigned delay() const noexcept { return delay_; }

private:
    unsigned delay_ = MinSpins;
};

// The assignment's backoff: sleep_for(nanoseconds(delay)), doubling to a maximum.
template <unsigned MinNs = 1000, unsigned MaxNs = 128000>
class SleepBackoff {
    static_assert(MinNs > 0, "MinNs must be > 0");
    static_assert(MinNs <= MaxNs, "MinNs must be <= MaxNs");

public:
    static constexpr unsigned next_delay(unsigned d) noexcept {
        return detail::next_delay<MaxNs>(d);
    }
    void operator()() noexcept {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay_));
        delay_ = next_delay(delay_);
    }
    unsigned delay() const noexcept { return delay_; }

private:
    unsigned delay_ = MinNs;
};

// Barrier that does not burn CPU while waiting: ranks that are idle (for
// example the ranks outside a sub-communicator) would otherwise busy-poll
// inside MPI_Barrier and steal cores from the ranks that are measured.
inline void polite_barrier(MPI_Comm comm) {
    MPI_Request rq;
    DLOCK_MPI(MPI_Ibarrier(comm, &rq));
    int done = 0;
    while (!done) {
        DLOCK_MPI(MPI_Test(&rq, &done, MPI_STATUS_IGNORE));
        if (!done) std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

// ----------------------------------------------------------------------------
// Window: an MPI RMA window holding 64-bit "slots", plus blocking helpers.
//
// Each slot sits on its own 128-byte stride (the default), so words polled by
// different ranks do not share cache lines. All helpers complete their
// operation (MPI_Win_flush) before returning.
// ----------------------------------------------------------------------------
inline constexpr int kStrideWords = 16;  // 16 x 8 bytes = 128 bytes

class Window {
public:
    using Init = std::initializer_list<std::pair<int, std::int64_t>>;

    // COLLECTIVE over comm. Memory from MPI_Win_allocate is NOT zeroed by MPI,
    // so every slot is set to `fill` and then to its value in `init`.
    Window(MPI_Comm comm, MPI_Aint slots, Init init = {}, int stride = kStrideWords,
           std::int64_t fill = 0)
        : comm_(comm), stride_(stride), slots_(slots) {
        DLOCK_MPI(MPI_Comm_rank(comm, &rank_));
        DLOCK_MPI(MPI_Comm_size(comm, &size_));
        void* base = nullptr;
        const MPI_Aint bytes = slots * static_cast<MPI_Aint>(stride) *
                               static_cast<MPI_Aint>(sizeof(std::int64_t));
        DLOCK_MPI(MPI_Win_allocate(bytes, static_cast<int>(sizeof(std::int64_t)), MPI_INFO_NULL,
                                   comm, &base, &win_));
        base_ = static_cast<volatile std::int64_t*>(base);
        for (MPI_Aint i = 0; i < slots * stride; ++i) base_[i] = fill;
        for (const auto& kv : init) base_[static_cast<MPI_Aint>(kv.first) * stride_] = kv.second;
        // One passive-target epoch for the whole lifetime of the window.
        DLOCK_MPI(MPI_Win_lock_all(MPI_MODE_NOCHECK, win_));
        DLOCK_MPI(MPI_Win_sync(win_));
        DLOCK_MPI(MPI_Barrier(comm));  // nobody communicates before everyone initialised
    }

    ~Window() {
        MPI_Barrier(comm_);  // all ranks are done with their operations
        MPI_Win_unlock_all(win_);
        MPI_Win_free(&win_);
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    int rank() const { return rank_; }
    int size() const { return size_; }
    MPI_Comm comm() const { return comm_; }

    // ---- atomic operations (remote or local, all through MPI) -----------
    std::int64_t fetch_add(int target, int slot, std::int64_t v) {
        return fao(MPI_SUM, v, target, slot);
    }
    std::int64_t exchange(int target, int slot, std::int64_t v) {
        return fao(MPI_REPLACE, v, target, slot);
    }
    std::int64_t read(int target, int slot) { return fao(MPI_NO_OP, 0, target, slot); }

    // Returns the old value; the swap happened iff the result equals `expected`.
    std::int64_t cas(int target, int slot, std::int64_t expected, std::int64_t desired) {
        std::int64_t old = 0;
        DLOCK_MPI(MPI_Compare_and_swap(&desired, &expected, &old, MPI_INT64_T, target, disp(slot),
                                       win_));
        flush(target);
        return old;
    }

    // Atomic store.
    void write(int target, int slot, std::int64_t v) {
        DLOCK_MPI(MPI_Accumulate(&v, 1, MPI_INT64_T, target, disp(slot), 1, MPI_INT64_T,
                                 MPI_REPLACE, win_));
        flush(target);
    }

    // ---- NON-atomic transfers: only for data protected by a lock ---------
    void put(int target, int slot, std::int64_t v) {
        DLOCK_MPI(MPI_Put(&v, 1, MPI_INT64_T, target, disp(slot), 1, MPI_INT64_T, win_));
        flush(target);
    }
    std::int64_t get(int target, int slot) {
        std::int64_t v = 0;
        DLOCK_MPI(MPI_Get(&v, 1, MPI_INT64_T, target, disp(slot), 1, MPI_INT64_T, win_));
        flush(target);
        return v;
    }
    // Contiguous slots (the window must have been created with stride 1).
    void get_range(int target, int first_slot, std::int64_t* out, int count) {
        DLOCK_MPI(MPI_Get(out, count, MPI_INT64_T, target, disp(first_slot), count, MPI_INT64_T,
                          win_));
        flush(target);
    }

    // ---- local access to this rank's own memory --------------------------
    // MPI_Win_sync makes remote updates visible to local loads (and local stores
    // visible to remote operations) in both the unified and separate models.
    std::int64_t local_load(int slot) {
        DLOCK_MPI(MPI_Win_sync(win_));
        return base_[static_cast<MPI_Aint>(slot) * stride_];
    }
    void local_store(int slot, std::int64_t v) {
        base_[static_cast<MPI_Aint>(slot) * stride_] = v;
    }
    void sync() { DLOCK_MPI(MPI_Win_sync(win_)); }

    // Poll a local slot until pred(value) holds (hybrid spin, periodic MPI progress).
    template <class Pred>
    std::int64_t wait_local(int slot, Pred pred) {
        SpinWait<> wait;
        unsigned polls = 0;
        for (;;) {
            const std::int64_t v = local_load(slot);
            if (pred(v)) return v;
            wait();
            if (DLOCK_PROGRESS_EVERY > 0 && ++polls % DLOCK_PROGRESS_EVERY == 0) progress();
        }
    }

    // Give the MPI library a chance to process incoming operations.
    void progress() {
        int flag = 0;
        DLOCK_MPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &flag, MPI_STATUS_IGNORE));
    }

    void flush(int target) { DLOCK_MPI(MPI_Win_flush(target, win_)); }

    // 1 = unified memory model, 0 = separate, -1 = unknown.
    int memory_model() {
        int* model = nullptr;
        int flag = 0;
        if (MPI_Win_get_attr(win_, MPI_WIN_MODEL, &model, &flag) != MPI_SUCCESS || !flag) return -1;
        return *model == MPI_WIN_UNIFIED ? 1 : 0;
    }

private:
    MPI_Aint disp(int slot) const { return static_cast<MPI_Aint>(slot) * stride_; }

    std::int64_t fao(MPI_Op op, std::int64_t v, int target, int slot) {
        std::int64_t res = 0;
        DLOCK_MPI(MPI_Fetch_and_op(&v, &res, MPI_INT64_T, target, disp(slot), op, win_));
        flush(target);
        return res;
    }

    MPI_Comm comm_;
    int stride_;
    MPI_Aint slots_;
    int rank_ = 0, size_ = 1;
    MPI_Win win_ = MPI_WIN_NULL;
    volatile std::int64_t* base_ = nullptr;
};

// ----------------------------------------------------------------------------
// TAS: CAS(0 -> 1) on a lock word hosted by rank `home`.
// Release is CAS(1 -> 0), so the word is only touched by CAS and NO_OP reads.
// ----------------------------------------------------------------------------
template <class Backoff = NoBackoff>
class basic_lock_tas {
public:
    static constexpr bool is_fifo = false;
    static constexpr bool has_try_lock = true;

    explicit basic_lock_tas(MPI_Comm comm, int home = 0) : w_(comm, 1), home_(home) {}

    void acquire() {
        Backoff backoff;
        while (w_.cas(home_, 0, 0, 1) != 0) backoff();
    }
    bool try_lock() { return w_.cas(home_, 0, 0, 1) == 0; }
    void release() { w_.cas(home_, 0, 1, 0); }

    void lock() { acquire(); }
    void unlock() { release(); }

private:
    Window w_;
    int home_;
};

// ----------------------------------------------------------------------------
// TTAS: poll the word with an atomic read, attempt the CAS only when it looks
// free, and back off only after a failed CAS. (On distributed memory the read
// is a remote operation too, so TTAS loses its cache advantage; it still avoids
// the heavier atomic read-modify-write while the lock is held.)
// ----------------------------------------------------------------------------
template <class Backoff = NoBackoff>
class basic_lock_ttas {
public:
    static constexpr bool is_fifo = false;
    static constexpr bool has_try_lock = true;

    explicit basic_lock_ttas(MPI_Comm comm, int home = 0) : w_(comm, 1), home_(home) {}

    void acquire() {
        Backoff backoff;
        for (;;) {
            SpinWait<> wait;
            while (w_.read(home_, 0) != 0) wait();
            if (w_.cas(home_, 0, 0, 1) == 0) return;
            backoff();
        }
    }
    bool try_lock() { return w_.read(home_, 0) == 0 && w_.cas(home_, 0, 0, 1) == 0; }
    void release() { w_.cas(home_, 0, 1, 0); }

    void lock() { acquire(); }
    void unlock() { release(); }

private:
    Window w_;
    int home_;
};

// ----------------------------------------------------------------------------
// Ticket lock (FIFO). next_ticket and now_serving live on rank `home`, in
// separate cache lines. PauseNs > 0 adds proportional backoff: a waiter k places
// behind the head waits about (k-1) * PauseNs between polls (capped at 100 us),
// so only the next-in-line rank polls at full speed.
// ----------------------------------------------------------------------------
template <unsigned PauseNs = 0>
class basic_lock_ticket {
    static constexpr int kNext = 0, kServing = 1;

public:
    static constexpr bool is_fifo = true;
    static constexpr bool has_try_lock = true;

    explicit basic_lock_ticket(MPI_Comm comm, int home = 0) : w_(comm, 2), home_(home) {}

    void acquire() {
        const std::int64_t my = w_.fetch_add(home_, kNext, 1);
        SpinWait<> wait;
        for (;;) {
            const std::int64_t cur = w_.read(home_, kServing);
            if (cur == my) break;
            if constexpr (PauseNs > 0) {
                const std::int64_t ns = std::min<std::int64_t>((my - cur - 1) * PauseNs, 100000);
                if (ns > 0) detail::spin_for_ns(ns);
            }
            wait();
        }
        ticket_ = my;
    }

    bool try_lock() {
        // Free iff next == serving; claim ticket `serving` with a CAS on next.
        const std::int64_t s = w_.read(home_, kServing);
        if (w_.cas(home_, kNext, s, s + 1) == s) {
            ticket_ = s;
            return true;
        }
        return false;
    }

    void release() { w_.write(home_, kServing, ticket_ + 1); }  // only the holder writes it

    void lock() { acquire(); }
    void unlock() { release(); }

private:
    Window w_;
    int home_;
    std::int64_t ticket_ = 0;  // ticket of the current holder (this rank)
};

// ----------------------------------------------------------------------------
// MCS lock (FIFO queue lock; the MPI-3 RMA formulation of Hoefler et al.)
//
//   tail   (on rank `home`) : rank at the end of the queue, -1 if empty
//   next   (on every rank)  : successor's rank, -1 if none yet
//   locked (on every rank)  : 1 while this rank waits for its predecessor
//
// A waiter swaps itself into tail, writes its rank into its predecessor's next,
// then spins on its OWN locked flag in local memory. Release writes 0 into the
// successor's locked flag: one remote write per hand-over. Because every
// (lock, rank) pair has its own slots, a rank may hold several MCS locks.
// ----------------------------------------------------------------------------
class lock_mcs {
    static constexpr int kTail = 0, kNext = 1, kLocked = 2;

public:
    static constexpr bool is_fifo = true;
    static constexpr bool has_try_lock = true;

    explicit lock_mcs(MPI_Comm comm, int home = 0)
        : w_(comm, 3, {{kTail, -1}, {kNext, -1}, {kLocked, 0}}), home_(home), me_(w_.rank()) {}

    void acquire() {
        // Local initialisation, published before we become visible in the queue.
        w_.local_store(kNext, -1);
        w_.local_store(kLocked, 1);
        w_.sync();

        const std::int64_t pred = w_.exchange(home_, kTail, me_);
        if (pred != -1) {
            w_.write(static_cast<int>(pred), kNext, me_);          // link behind pred
            w_.wait_local(kLocked, [](std::int64_t v) { return v == 0; });  // spin locally
        }
    }

    bool try_lock() {
        w_.local_store(kNext, -1);
        w_.local_store(kLocked, 0);
        w_.sync();
        return w_.cas(home_, kTail, -1, me_) == -1;
    }

    void release() {
        std::int64_t succ = w_.local_load(kNext);
        if (succ == -1) {
            // No visible successor: try to reset the queue to empty.
            if (w_.cas(home_, kTail, me_, -1) == me_) return;
            // A successor swapped into tail but has not linked itself yet.
            succ = w_.wait_local(kNext, [](std::int64_t v) { return v != -1; });
        }
        w_.write(static_cast<int>(succ), kLocked, 0);  // pass ownership
    }

    void lock() { acquire(); }
    void unlock() { release(); }

private:
    Window w_;
    int home_;
    int me_;
};

// ----------------------------------------------------------------------------
// Baseline: MPI's own exclusive passive-target lock.
//
// MPI_Win_lock only guarantees that RMA operations of different exclusive
// epochs are serialised; an implementation may even defer taking the lock until
// the first operation. To obtain a mutex for ARBITRARY local code, acquire()
// therefore issues one (empty) atomic read and flushes it; the flush returns only
// after the lock has been granted, and unlock() ends the epoch. There is no
// try_lock.
// ----------------------------------------------------------------------------
class lock_winlock {
public:
    static constexpr bool is_fifo = false;
    static constexpr bool has_try_lock = false;

    explicit lock_winlock(MPI_Comm comm, int home = 0) : comm_(comm), home_(home) {
        void* base = nullptr;
        DLOCK_MPI(MPI_Win_allocate(static_cast<MPI_Aint>(sizeof(std::int64_t)),
                                   static_cast<int>(sizeof(std::int64_t)), MPI_INFO_NULL, comm,
                                   &base, &win_));
        *static_cast<volatile std::int64_t*>(base) = 0;
        DLOCK_MPI(MPI_Barrier(comm));
    }
    ~lock_winlock() {
        MPI_Barrier(comm_);
        MPI_Win_free(&win_);
    }
    lock_winlock(const lock_winlock&) = delete;
    lock_winlock& operator=(const lock_winlock&) = delete;

    void acquire() {
        DLOCK_MPI(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, home_, 0, win_));
        std::int64_t dummy = 0, res = 0;
        DLOCK_MPI(MPI_Fetch_and_op(&dummy, &res, MPI_INT64_T, home_, 0, MPI_NO_OP, win_));
        DLOCK_MPI(MPI_Win_flush(home_, win_));
    }
    void release() { DLOCK_MPI(MPI_Win_unlock(home_, win_)); }

    void lock() { acquire(); }
    void unlock() { release(); }

private:
    MPI_Comm comm_;
    int home_;
    MPI_Win win_ = MPI_WIN_NULL;
};

// RAII guard (std::lock_guard cannot be used: locks are constructed from a communicator,
// but any lock here is BasicLockable, so std::lock_guard<L> on an existing lock works too).
template <class L>
class Guard {
public:
    explicit Guard(L& l) : l_(l) { l_.acquire(); }
    ~Guard() { l_.release(); }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

private:
    L& l_;
};

// ----------------------------------------------------------------------------
// Named variants used by the tests and benchmarks
// ----------------------------------------------------------------------------
using lock_tas         = basic_lock_tas<NoBackoff>;
using lock_tas_pause   = basic_lock_tas<PauseBackoff<>>;
using lock_tas_sleep   = basic_lock_tas<SleepBackoff<>>;
using lock_ttas        = basic_lock_ttas<NoBackoff>;
using lock_ttas_pause  = basic_lock_ttas<PauseBackoff<>>;
using lock_ttas_sleep  = basic_lock_ttas<SleepBackoff<>>;
using lock_ticket      = basic_lock_ticket<0>;
using lock_ticket_prop = basic_lock_ticket<1000>;

}  // namespace dlock

#endif  // MPI_LOCK_H
