#ifndef LOCK_H
#define LOCK_H

// ============================================================================
// lock.h -- spinlock family for the Parallel Computing assignment (Section 1)
//
//   lock_tas / lock_ttas      : test-and-set / test-and-test-and-set
//   lock_ticket               : FIFO ticket lock
//   lock_mcs                  : FIFO queue lock (local spinning)
//
// All locks expose acquire()/release() (spec interface), lock()/unlock()/
// try_lock() (so std::lock_guard / std::unique_lock work), and a
// `static constexpr bool is_fifo` trait.
//
// Requires C++17 (guaranteed copy elision for non-movable atomics-holding
// locks, over-aligned new for MCS nodes).
//
// Compile-time knobs (all optional):
//   -DLOCK_CACHE_LINE=N              override the padding size (bytes)
//   -DLOCK_SPIN_LIMIT=N              pauses before a waiter starts yielding
//   -DLOCK_MUTATE_RELAXED_RELEASE    TEST ONLY: weaken release orderings so the
//                                    correctness harness can be mutation-tested
// ============================================================================

#if defined(_MSVC_LANG)
  #if _MSVC_LANG < 201703L
    #error "lock.h requires C++17 (compile with /std:c++17)"
  #endif
#elif __cplusplus < 201703L
  #error "lock.h requires C++17 (compile with -std=c++17)"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

// ----------------------------------------------------------------------------
// Platform constants
// ----------------------------------------------------------------------------

// Padding unit used to keep hot atomics on separate cache lines.
// std::hardware_destructive_interference_size is not reliably available, so we
// pick per platform. 128 B on Apple Silicon (128 B lines) and x86-64 (the
// adjacent-line prefetcher effectively couples pairs of 64 B lines).
#ifndef LOCK_CACHE_LINE
  #if (defined(__APPLE__) && defined(__aarch64__)) || defined(__x86_64__) || defined(_M_X64)
    #define LOCK_CACHE_LINE 128
  #else
    #define LOCK_CACHE_LINE 64
  #endif
#endif

// Number of cpu_relax() iterations a waiter performs before it starts calling
// yield(). Protects against oversubscription (waiter spinning while the lock
// holder / next-in-line thread is descheduled).
#ifndef LOCK_SPIN_LIMIT
  #define LOCK_SPIN_LIMIT 2048
#endif

// Ordering used by every "publishing" store (unlock). Mutation hook for tests.
#ifdef LOCK_MUTATE_RELAXED_RELEASE
  #define LOCK_RELEASE_ORDER std::memory_order_relaxed
#else
  #define LOCK_RELEASE_ORDER std::memory_order_release
#endif

inline constexpr std::size_t kCacheLine = LOCK_CACHE_LINE;
#ifdef LOCK_MUTATE_RELAXED_RELEASE
inline constexpr bool kMutatedRelease = true;
#else
inline constexpr bool kMutatedRelease = false;
#endif

// ----------------------------------------------------------------------------
// cpu_relax(): architecture-specific spin-wait hint
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
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
    asm volatile("or 27,27,27" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);  // compiler barrier only
#endif
}

namespace detail {

// Tiny per-thread xorshift32 used only to jitter backoff delays.
inline std::uint32_t jitter_rand() noexcept {
    static thread_local std::uint32_t s = 0;
    if (s == 0) {
        // The address of a thread_local differs per thread -> distinct seeds.
        s = (static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&s) >> 4) *
             2654435761u) | 1u;
    }
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

// Doubling that saturates at Max without overflowing unsigned.
template <unsigned Max>
constexpr unsigned next_delay(unsigned d) noexcept {
    return (d > Max / 2) ? Max : d * 2;
}

}  // namespace detail

// ----------------------------------------------------------------------------
// SpinWait: hybrid wait used by TTAS / ticket / MCS waiting loops.
// Spins with cpu_relax() for `Limit` calls, then yields to the scheduler.
// ----------------------------------------------------------------------------
template <unsigned Limit = LOCK_SPIN_LIMIT>
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
    void reset() noexcept { spins_ = 0; }

private:
    unsigned spins_ = 0;
};

// ----------------------------------------------------------------------------
// Backoff policies (compile-time parameters; zero runtime state in the lock).
// A fresh policy object is created per acquire() call, so each acquisition
// starts at the minimum delay.
// ----------------------------------------------------------------------------

// No backoff: retry immediately (textbook TAS/TTAS).
struct NoBackoff {
    void operator()() noexcept {}
};

// Exponential backoff using CPU pause hints, with random jitter in
// [delay/2, delay] so contending threads do not retry in lock-step.
// Once the delay saturates at MaxSpins the thread also yields each time, which
// keeps the lock usable under oversubscription.
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

// Exponential backoff where "one unit" is one std::this_thread::yield().
// (This is the behaviour of the original ExponentialBackoff class; the cap is
// lower because each unit is a system call.)
template <unsigned MinYields = 1, unsigned MaxYields = 64>
class YieldBackoff {
    static_assert(MinYields > 0, "MinYields must be > 0");
    static_assert(MinYields <= MaxYields, "MinYields must be <= MaxYields");

public:
    static constexpr unsigned next_delay(unsigned d) noexcept {
        return detail::next_delay<MaxYields>(d);
    }

    void operator()() noexcept {
        for (unsigned i = 0; i < delay_; ++i) std::this_thread::yield();
        delay_ = next_delay(delay_);
    }

    unsigned delay() const noexcept { return delay_; }

private:
    unsigned delay_ = MinYields;
};

// Exponential backoff exactly as written in the assignment text:
// sleep_for(nanoseconds(delay)), doubling up to a maximum.
// Note: the real sleep granularity is set by the OS timer (tens of microseconds
// on Linux, ~1 ms on macOS under load, up to ~15 ms on Windows).
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

// ----------------------------------------------------------------------------
// TAS (Test-And-Set) lock
//   test_and_set returns the OLD value: false -> we got the lock,
//   true -> someone holds it, back off (optionally) and retry.
// ----------------------------------------------------------------------------
template <class Backoff = NoBackoff>
class basic_lock_tas {
public:
    static constexpr bool is_fifo = false;

    basic_lock_tas() noexcept { flag_.clear(std::memory_order_relaxed); }
    basic_lock_tas(const basic_lock_tas&) = delete;
    basic_lock_tas& operator=(const basic_lock_tas&) = delete;

    void acquire() noexcept {
        Backoff backoff;
        while (flag_.test_and_set(std::memory_order_acquire)) {
            backoff();
        }
    }

    bool try_lock() noexcept { return !flag_.test_and_set(std::memory_order_acquire); }

    void release() noexcept { flag_.clear(LOCK_RELEASE_ORDER); }

    void lock() noexcept { acquire(); }
    void unlock() noexcept { release(); }

private:
    alignas(kCacheLine) std::atomic_flag flag_;
};

// ----------------------------------------------------------------------------
// TTAS (Test-Test-And-Set) lock
//   Waiters spin on a plain load (cache-resident, no invalidations) and only
//   attempt the expensive exchange when the lock looks free. Backoff is applied
//   ONLY after a failed exchange (real evidence of contention), not while
//   merely waiting for the holder to finish.
// ----------------------------------------------------------------------------
template <class Backoff = NoBackoff>
class basic_lock_ttas {
public:
    static constexpr bool is_fifo = false;

    basic_lock_ttas() noexcept : locked_(false) {}
    basic_lock_ttas(const basic_lock_ttas&) = delete;
    basic_lock_ttas& operator=(const basic_lock_ttas&) = delete;

    void acquire() noexcept {
        Backoff backoff;
        for (;;) {
            SpinWait<> wait;
            while (locked_.load(std::memory_order_relaxed)) {
                wait();
            }
            if (!locked_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            backoff();
        }
    }

    bool try_lock() noexcept {
        return !locked_.load(std::memory_order_relaxed) &&
               !locked_.exchange(true, std::memory_order_acquire);
    }

    void release() noexcept { locked_.store(false, LOCK_RELEASE_ORDER); }

    void lock() noexcept { acquire(); }
    void unlock() noexcept { release(); }

private:
    alignas(kCacheLine) std::atomic<bool> locked_;
};

// ----------------------------------------------------------------------------
// Ticket lock (FIFO)
//   next_    : next ticket to hand out   (fetch_add by arriving threads)
//   serving_ : ticket currently served   (written only by the lock holder)
// They live on separate cache lines: arrivals write next_, waiters read
// serving_.
//
// PauseUnit == 0 : waiters poll serving_ with the hybrid SpinWait.
// PauseUnit  > 0 : proportional backoff -- a waiter k places behind the head
//                  pauses (k-1)*PauseUnit times between polls, so only the
//                  next-in-line thread polls aggressively.
// ----------------------------------------------------------------------------
template <unsigned PauseUnit = 0>
class basic_lock_ticket {
public:
    static constexpr bool is_fifo = true;

    basic_lock_ticket() noexcept : next_(0), serving_(0) {}
    basic_lock_ticket(const basic_lock_ticket&) = delete;
    basic_lock_ticket& operator=(const basic_lock_ticket&) = delete;

    void acquire() noexcept {
        const std::uint64_t my = next_.fetch_add(1, std::memory_order_relaxed);

        if constexpr (PauseUnit == 0) {
            SpinWait<> wait;
            while (serving_.load(std::memory_order_acquire) != my) {
                wait();
            }
        } else {
            constexpr std::uint64_t kMaxPauses = 1u << 14;
            std::uint64_t spent = 0;
            for (;;) {
                const std::uint64_t cur = serving_.load(std::memory_order_acquire);
                if (cur == my) return;
                const std::uint64_t pauses =
                    std::min<std::uint64_t>((my - cur - 1) * PauseUnit, kMaxPauses);
                for (std::uint64_t i = 0; i < pauses; ++i) cpu_relax();
                cpu_relax();
                spent += pauses + 1;
                if (spent >= LOCK_SPIN_LIMIT) std::this_thread::yield();
            }
        }
    }

    bool try_lock() noexcept {
        // The lock is free iff next_ == serving_. Taking ticket `serving`
        // with a CAS succeeds only if nobody else drew a ticket meanwhile.
        const std::uint64_t serving = serving_.load(std::memory_order_acquire);
        std::uint64_t expected = serving;
        return next_.compare_exchange_strong(expected, serving + 1,
                                             std::memory_order_relaxed,
                                             std::memory_order_relaxed);
    }

    void release() noexcept {
        // Only the holder writes serving_, so a plain load + store replaces
        // the (more expensive) atomic RMW.
        serving_.store(serving_.load(std::memory_order_relaxed) + 1, LOCK_RELEASE_ORDER);
    }

    void lock() noexcept { acquire(); }
    void unlock() noexcept { release(); }

private:
    alignas(kCacheLine) std::atomic<std::uint64_t> next_;
    alignas(kCacheLine) std::atomic<std::uint64_t> serving_;
};

// ----------------------------------------------------------------------------
// MCS lock (FIFO queue lock, each waiter spins on its own node)
//
//                 tail_
//                   |
//                   v
//   [holder] -> [waiter B] -> [waiter C]
//
// Nodes come from a per-thread free list (heap-allocated on first use, reused
// afterwards), so a thread may hold several MCS locks at once, release them in
// any order (including hand-over-hand), and use many lock objects. The node of
// the current holder is remembered in owner_, which only the holder touches.
// A lock must be released by the thread that acquired it (mutex semantics).
// ----------------------------------------------------------------------------
class lock_mcs {
    struct alignas(kCacheLine) Node {
        std::atomic<Node*> next{nullptr};
        std::atomic<bool> locked{false};
        Node* pool_next = nullptr;  // free-list link (only while node is free)
    };

    // Per-thread free list of nodes.
    struct Pool {
        Node* head = nullptr;
        ~Pool() {
            while (head != nullptr) {
                Node* n = head;
                head = n->pool_next;
                delete n;
            }
        }
        Node* get() {
            if (head != nullptr) {
                Node* n = head;
                head = n->pool_next;
                return n;
            }
            return new Node();
        }
        void put(Node* n) noexcept {
            n->pool_next = head;
            head = n;
        }
    };

    static Pool& pool() noexcept {
        static thread_local Pool p;
        return p;
    }

public:
    static constexpr bool is_fifo = true;

    lock_mcs() noexcept : tail_(nullptr), owner_(nullptr) {}
    lock_mcs(const lock_mcs&) = delete;
    lock_mcs& operator=(const lock_mcs&) = delete;

    void acquire() {
        Node* node = pool().get();
        node->next.store(nullptr, std::memory_order_relaxed);
        node->locked.store(true, std::memory_order_relaxed);

        // Append ourselves to the queue. pred == nullptr -> lock was free.
        Node* pred = tail_.exchange(node, std::memory_order_acq_rel);

        if (pred != nullptr) {
            // Link behind our predecessor, then spin on OUR OWN flag.
            pred->next.store(node, std::memory_order_release);
            SpinWait<> wait;
            while (node->locked.load(std::memory_order_acquire)) {
                wait();
            }
        }
        owner_ = node;  // written by the holder only
    }

    bool try_lock() {
        Node* node = pool().get();
        node->next.store(nullptr, std::memory_order_relaxed);
        node->locked.store(false, std::memory_order_relaxed);

        Node* expected = nullptr;
        if (tail_.compare_exchange_strong(expected, node, std::memory_order_acq_rel,
                                          std::memory_order_relaxed)) {
            owner_ = node;
            return true;
        }
        pool().put(node);
        return false;
    }

    void release() {
        Node* node = owner_;
        assert(node != nullptr && "release() without a matching acquire()");
        owner_ = nullptr;  // clear before handing the lock over

        Node* successor = node->next.load(std::memory_order_acquire);
        if (successor == nullptr) {
            // No visible successor: try to swing tail_ back to empty.
            Node* expected = node;
            if (tail_.compare_exchange_strong(expected, nullptr, LOCK_RELEASE_ORDER,
                                              std::memory_order_relaxed)) {
                pool().put(node);
                return;
            }
            // A successor swapped itself into tail_ but has not linked yet.
            SpinWait<> wait;
            while ((successor = node->next.load(std::memory_order_acquire)) == nullptr) {
                wait();
            }
        }

        successor->locked.store(false, LOCK_RELEASE_ORDER);  // pass ownership
        pool().put(node);  // successor never touches our node again
    }

    void lock() { acquire(); }
    void unlock() { release(); }

private:
    alignas(kCacheLine) std::atomic<Node*> tail_;
    alignas(kCacheLine) Node* owner_;
};

// ----------------------------------------------------------------------------
// Named variants used by the tests / benchmarks
// ----------------------------------------------------------------------------
using lock_tas         = basic_lock_tas<NoBackoff>;
using lock_tas_pause   = basic_lock_tas<PauseBackoff<>>;
using lock_tas_yield   = basic_lock_tas<YieldBackoff<>>;
using lock_tas_sleep   = basic_lock_tas<SleepBackoff<>>;

using lock_ttas        = basic_lock_ttas<NoBackoff>;
using lock_ttas_pause  = basic_lock_ttas<PauseBackoff<>>;
using lock_ttas_yield  = basic_lock_ttas<YieldBackoff<>>;
using lock_ttas_sleep  = basic_lock_ttas<SleepBackoff<>>;

using lock_ticket      = basic_lock_ticket<0>;
using lock_ticket_prop = basic_lock_ticket<64>;

#endif  // LOCK_H