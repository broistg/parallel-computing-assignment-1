#include "host_info.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#else
#include <unistd.h>
#endif

namespace {

std::string env_or_empty(const char* key) {
  const char* v = std::getenv(key);
  return v != nullptr ? std::string(v) : std::string();
}

#if defined(__linux__)
std::string read_cpu_linux() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const auto pos = line.find("model name");
    if (pos == std::string::npos) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string model = line.substr(colon + 1);
    while (!model.empty() && model.front() == ' ') {
      model.erase(model.begin());
    }
    return model;
  }
  return "unknown";
}

std::string read_mem_linux() {
  std::ifstream in("/proc/meminfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("MemTotal:", 0) != 0) {
      continue;
    }
    std::istringstream iss(line);
    std::string key;
    long kb = 0;
    std::string unit;
    iss >> key >> kb >> unit;
    if (kb > 0) {
      return std::to_string(static_cast<long long>(kb) * 1024LL);
    }
  }
  return "";
}
#endif

#if defined(__APPLE__)
std::string popen_trim(const char* cmd) {
  FILE* pipe = popen(cmd, "r");
  if (pipe == nullptr) {
    return "";
  }
  std::ostringstream collected;
  char buf[256];
  while (fgets(buf, sizeof(buf), pipe) != nullptr) {
    collected << buf;
  }
  pclose(pipe);
  std::string out = collected.str();
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}
#endif

int cpu_count() {
#if defined(_WIN32)
  return 0;
#else
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? static_cast<int>(n) : 0;
#endif
}

}  // namespace

std::string hostname() {
  const std::string override = env_or_empty("EX3_HOSTNAME");
  if (!override.empty()) {
    return override;
  }
  char buf[256];
#if defined(_WIN32)
  const char* env = std::getenv("COMPUTERNAME");
  return env != nullptr ? env : "unknown";
#else
  if (gethostname(buf, sizeof(buf)) != 0) {
    return "unknown";
  }
  buf[sizeof(buf) - 1] = '\0';
  return buf;
#endif
}

std::string cpu_model() {
  const std::string override = env_or_empty("EX3_CPU_MODEL");
  if (!override.empty()) {
    return override;
  }
#if defined(__APPLE__)
  const std::string model = popen_trim("sysctl -n machdep.cpu.brand_string 2>/dev/null");
  return model.empty() ? "unknown" : model;
#elif defined(__linux__)
  return read_cpu_linux();
#else
  return "unknown";
#endif
}

int logical_cpus() {
  const std::string override = env_or_empty("EX3_LOGICAL_CPUS");
  if (!override.empty()) {
    try {
      return std::stoi(override);
    } catch (...) {
      // fall through
    }
  }
  return cpu_count();
}

std::string os_name() {
  const std::string override = env_or_empty("EX3_OS");
  if (!override.empty()) {
    return override;
  }
#if defined(__APPLE__)
  const std::string ver = popen_trim("sw_vers -productVersion 2>/dev/null");
  return ver.empty() ? "macOS" : ("macOS " + ver);
#elif defined(__linux__)
  return "Linux";
#else
  return "unknown";
#endif
}

std::string memory_bytes() {
  const std::string override = env_or_empty("EX3_MEMORY_BYTES");
  if (!override.empty()) {
    return override;
  }
#if defined(__APPLE__)
  return popen_trim("sysctl -n hw.memsize 2>/dev/null");
#elif defined(__linux__)
  return read_mem_linux();
#else
  return "";
#endif
}

std::string json_escape(const std::string& s) {
  std::ostringstream out;
  for (char c : s) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}
