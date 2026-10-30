#include "host_info.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

std::string env_or_empty(const char* key) {
  const char* v = std::getenv(key);
  return v != nullptr ? std::string(v) : std::string();
}

#if defined(_WIN32)
std::string read_cpu_windows() {
  // PROCESSOR_IDENTIFIER is set by Windows (e.g. "Intel64 Family 6 ...").
  const char* id = std::getenv("PROCESSOR_IDENTIFIER");
  if (id != nullptr && id[0] != '\0') {
    return id;
  }
  SYSTEM_INFO info;
  GetNativeSystemInfo(&info);
  switch (info.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64:
      return "x86_64";
    case PROCESSOR_ARCHITECTURE_ARM64:
      return "ARM64";
    case PROCESSOR_ARCHITECTURE_INTEL:
      return "x86";
    default:
      return "unknown";
  }
}

std::string read_mem_windows() {
  MEMORYSTATUSEX st;
  st.dwLength = sizeof(st);
  if (!GlobalMemoryStatusEx(&st)) {
    return "";
  }
  return std::to_string(static_cast<unsigned long long>(st.ullTotalPhys));
}

std::string read_os_windows() {
  // Prefer env for a short label; avoid deprecated GetVersionEx.
  const char* ver = std::getenv("OS");
  std::ostringstream out;
  out << (ver != nullptr ? ver : "Windows");
#if defined(_WIN64)
  out << " 64-bit";
#endif
  return out.str();
}

int cpu_count_windows() {
  SYSTEM_INFO info;
  GetNativeSystemInfo(&info);
  return info.dwNumberOfProcessors > 0
             ? static_cast<int>(info.dwNumberOfProcessors)
             : 0;
}
#endif

#if defined(__linux__)
std::string trim_front(std::string s) {
  while (!s.empty() && s.front() == ' ') {
    s.erase(s.begin());
  }
  return s;
}

std::string read_cpu_linux() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  std::string hardware;
  std::string processor;
  while (std::getline(in, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, colon);
    std::string val = trim_front(line.substr(colon + 1));
    if (key.find("model name") != std::string::npos ||
        key.find("Model name") != std::string::npos) {
      if (!val.empty()) {
        return val;
      }
    }
    if (key.rfind("Hardware", 0) == 0 && hardware.empty()) {
      hardware = val;
    }
    if (key.rfind("Processor", 0) == 0 && processor.empty()) {
      processor = val;
    }
  }
  if (!hardware.empty()) {
    return hardware;
  }
  if (!processor.empty()) {
    return processor;
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
  return cpu_count_windows();
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
#if defined(_WIN32)
  char buf[MAX_COMPUTERNAME_LENGTH + 1];
  DWORD size = static_cast<DWORD>(sizeof(buf));
  if (GetComputerNameA(buf, &size)) {
    return buf;
  }
  const char* env = std::getenv("COMPUTERNAME");
  return env != nullptr ? env : "unknown";
#else
  char buf[256];
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
#if defined(_WIN32)
  return read_cpu_windows();
#elif defined(__APPLE__)
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
    }
  }
  return cpu_count();
}

std::string os_name() {
  const std::string override = env_or_empty("EX3_OS");
  if (!override.empty()) {
    return override;
  }
#if defined(_WIN32)
  return read_os_windows();
#elif defined(__APPLE__)
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
#if defined(_WIN32)
  return read_mem_windows();
#elif defined(__APPLE__)
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
