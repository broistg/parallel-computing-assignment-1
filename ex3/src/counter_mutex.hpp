#pragma once

#include "counter.hpp"

#include <mutex>

// Only one thread may touch value_ at a time.
// Everyone else waits on the mutex until the holder unlocks.
class MutexCounter : public Counter {
 public:
  void increment() override {
    std::lock_guard<std::mutex> lock(mu_);
    value_ = value_ + 1;
  }

  long long get() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return value_;
  }

 private:
  mutable std::mutex mu_;
  long long value_ = 0;
};
