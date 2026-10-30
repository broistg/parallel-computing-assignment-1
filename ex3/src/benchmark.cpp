#include "counter.hpp"
#include "host_info.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
  int threads = 4;
  long long iters = 100000;
  std::string impl = "fetch_add";
  std::string out;
};

void usage() {
  std::cerr
      << "usage: benchmark --threads N --iters N --impl NAME [--out FILE]\n"
      << "  impl: mutex | cas | fetch_add\n";
}

Options parse(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--threads") {
      opt.threads = std::stoi(need("--threads"));
    } else if (arg == "--iters") {
      opt.iters = std::stoll(need("--iters"));
    } else if (arg == "--impl") {
      opt.impl = need("--impl");
    } else if (arg == "--out") {
      opt.out = need("--out");
    } else if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument " + arg);
    }
  }
  if (opt.threads < 1 || opt.iters < 1) {
    throw std::invalid_argument("threads and iters must be >= 1");
  }
  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse(argc, argv);
    auto counter = make_counter(opt.impl);

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(opt.threads));
    for (int t = 0; t < opt.threads; ++t) {
      workers.emplace_back([&counter, iters = opt.iters] {
        for (long long i = 0; i < iters; ++i) {
          counter->increment();
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(stop - start).count();

    const long long expected =
        static_cast<long long>(opt.threads) * opt.iters;
    const long long got = counter->get();
    if (got != expected) {
      std::cerr << "correctness failure: got " << got << " expected "
                << expected << "\n";
      return 2;
    }

    const std::string mem = memory_bytes();
    const std::string json =
        std::string("{\n") + "  \"hostname\": \"" + json_escape(hostname()) +
        "\",\n" + "  \"cpu_model\": \"" + json_escape(cpu_model()) + "\",\n" +
        "  \"os\": \"" + json_escape(os_name()) + "\",\n" +
        "  \"logical_cpus\": " + std::to_string(logical_cpus()) + ",\n" +
        "  \"memory_bytes\": " + (mem.empty() ? "null" : mem) + ",\n" +
        "  \"threads\": " + std::to_string(opt.threads) + ",\n" +
        "  \"increments_per_thread\": " + std::to_string(opt.iters) + ",\n" +
        "  \"impl\": \"" + json_escape(opt.impl) + "\",\n" +
        "  \"seconds\": " + std::to_string(seconds) + ",\n" +
        "  \"total_increments\": " + std::to_string(got) + "\n" + "}\n";

    if (opt.out.empty()) {
      std::cout << json;
    } else {
      std::ofstream file(opt.out);
      if (!file) {
        std::cerr << "cannot write " << opt.out << "\n";
        return 1;
      }
      file << json;
      std::cout << opt.out << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    usage();
    return 1;
  }
}
