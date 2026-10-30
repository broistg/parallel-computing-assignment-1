#pragma once

#include "backoff.hpp"
#include "counter.hpp"

#include <atomic>

// Lock-free increment: try to swap expected -> expected+1.
// If another thread won the race, expected is updated and we retry,
// waiting longer each time (exponential backoff).
class CasCounter : public Counter {
 public:
  void increment() override {
    long long expected = value_.load(std::memory_order_relaxed);
    ExponentialBackoff backoff;
    while (!value_.compare_exchange_weak(expected, expected + 1,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
      backoff.pause();
    }
  }

  long long get() const override {
    return value_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<long long> value_{0};
};
