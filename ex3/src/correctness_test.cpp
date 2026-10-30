#include "counter.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

bool check(const std::string& impl, int threads, long long iters) {
  auto counter = make_counter(impl);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&counter, iters] {
      for (long long i = 0; i < iters; ++i) {
        counter->increment();
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  const long long expected = static_cast<long long>(threads) * iters;
  const long long got = counter->get();
  if (got != expected) {
    std::cerr << "FAIL " << impl << " threads=" << threads
              << " iters=" << iters << " got=" << got
              << " expected=" << expected << "\n";
    return false;
  }
  std::cout << "ok   " << impl << " threads=" << threads
            << " iters=" << iters << "\n";
  return true;
}

}  // namespace

int main() {
  const std::vector<std::string> impls = {"mutex", "cas", "fetch_add"};
  bool ok = true;
  for (const auto& impl : impls) {
    ok = check(impl, 1, 1000) && ok;
    ok = check(impl, 4, 50000) && ok;
    ok = check(impl, 8, 20000) && ok;
  }
  // Repeat the hot path so a lost update is unlikely to hide.
  for (int round = 0; round < 5; ++round) {
    ok = check("cas", 8, 100000) && ok;
    ok = check("mutex", 8, 20000) && ok;
    ok = check("fetch_add", 8, 100000) && ok;
  }
  if (!ok) {
    std::cerr << "correctness tests failed\n";
    return 1;
  }
  std::cout << "all correctness tests passed\n";
  return 0;
}
