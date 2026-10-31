#include "counter.hpp"

#include "counter_cas.hpp"
#include "counter_fetch_add.hpp"
#include "counter_mutex.hpp"

#include <stdexcept>

std::unique_ptr<Counter> make_counter(const std::string& impl) {
  if (impl == "mutex") {
    return std::make_unique<MutexCounter>();
  }
  if (impl == "cas") {
    return std::make_unique<CasCounter>();
  }
  if (impl == "fetch_add") {
    return std::make_unique<FetchAddCounter>();
  }
  throw std::invalid_argument(
      "unknown impl '" + impl + "' (use mutex, cas, or fetch_add)");
}
