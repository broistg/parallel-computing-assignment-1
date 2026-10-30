#pragma once

#include <string>

// Prefer EX3_* env overrides (set by run scripts from the host),
// so Docker runs still record the real machine, not the container.
std::string hostname();
std::string cpu_model();
int logical_cpus();
std::string os_name();
std::string memory_bytes();
std::string json_escape(const std::string& s);
