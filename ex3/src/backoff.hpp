#pragma once

#include <thread>

// Wait a little longer after each failed attempt.
// Used by the CAS counter so threads stop hammering the same cache line.
class ExponentialBackoff {
 public:
  void pause() {
    for (int i = 0; i < spins_; ++i) {
      cpu_pause();
    }
    if (spins_ < kMaxSpins) {
      spins_ *= 2;
    } else {
      // Cap the spin, then give the core to another thread.
      std::this_thread::yield();
    }
  }

 private:
  static void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
  }

  int spins_ = 1;
  static constexpr int kMaxSpins = 1024;
};
