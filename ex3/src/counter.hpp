#pragma once

#include <memory>
#include <string>

// One shared counter updated by many threads.
// increment() must not lose updates. get() returns the current total.
struct Counter {
  virtual void increment() = 0;
  virtual long long get() const = 0;
  virtual ~Counter() = default;
};

// impl is one of: mutex, cas, fetch_add
std::unique_ptr<Counter> make_counter(const std::string& impl);
