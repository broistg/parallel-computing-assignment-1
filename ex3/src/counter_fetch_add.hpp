#pragma once

#include "counter.hpp"

#include <atomic>

// One hardware read-modify-write. No software retry loop.
class FetchAddCounter : public Counter {
 public:
  void increment() override {
    value_.fetch_add(1, std::memory_order_relaxed);
  }

  long long get() const override {
    return value_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<long long> value_{0};
};
