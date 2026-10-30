#!/usr/bin/env bash
# Portable one-command study: macOS + Linux + Windows (Git Bash / WSL).
# Windows PowerShell users: prefer .\run.ps1 <name>
#   ./run.sh dangnhh5
#   ITERS=5000 ./run.sh smoke
#   ./run.sh merge
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"

THREADS=(1 2 4 8 16 32)
IMPLS=(mutex cas fetch_add)
DEFAULT_ITERS=(5 25 100 500 2500 10000 50000 200000 500000 1000000)

die() { echo "error: $*" >&2; exit 1; }

usage() {
  echo "usage: $0 <device_name>   # cmake build + full study on this machine"
  echo "       $0 merge           # merge ≥3 results/* into results/report/"
  echo "quick: ITERS=5000 $0 smoke"
  echo ""
  echo "macOS/Linux/Windows(Git Bash). needs: cmake ≥3.16, C++17, python3"
  echo "Windows PowerShell: .\\run.ps1 <name>"
}

os_id() {
  uname -s 2>/dev/null || echo unknown
}

is_windows() {
  case "$(os_id)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT) return 0 ;;
  esac
  [[ -n "${WINDIR:-}" ]] && return 0
  return 1
}

ncpus() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif getconf _NPROCESSORS_ONLN >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  elif [[ "$(os_id)" == "Darwin" ]]; then
    sysctl -n hw.ncpu 2>/dev/null || echo 2
  elif is_windows && [[ -n "${NUMBER_OF_PROCESSORS:-}" ]]; then
    echo "${NUMBER_OF_PROCESSORS}"
  else
    echo 2
  fi
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  elif command -v py >/dev/null 2>&1; then
    echo "py -3"
  else
    return 1
  fi
}

find_exe() {
  local name="$1"
  local c
  for c in \
    "build/${name}" \
    "build/${name}.exe" \
    "build/Release/${name}.exe" \
    "build/RelWithDebInfo/${name}.exe" \
    "build/Debug/${name}.exe"; do
    if [[ -x "${c}" || -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

install_hint() {
  case "$(os_id)" in
    Darwin)
      echo "  macOS: xcode-select --install && brew install cmake python3"
      ;;
    Linux)
      echo "  Debian/Ubuntu: sudo apt-get install -y build-essential cmake python3"
      echo "  Fedora:        sudo dnf install -y gcc-c++ cmake python3"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      echo "  Windows: install VS Build Tools (C++), cmake, python"
      echo "           or use PowerShell: .\\run.ps1 <name>"
      echo "  winget install Kitware.CMake Python.Python.3.12"
      ;;
    *)
      if is_windows; then
        echo "  Windows: VS Build Tools + cmake + python, or .\\run.ps1 <name>"
      else
        echo "  install: cmake (>=3.16), C++17 compiler, python3"
      fi
      ;;
  esac
}

require_tools() {
  local missing=0
  if ! command -v cmake >/dev/null 2>&1; then
    echo "missing: cmake" >&2
    missing=1
  fi
  if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 \
    && ! command -v clang++ >/dev/null 2>&1 && ! command -v cl >/dev/null 2>&1; then
    echo "missing: C++ compiler (g++/clang++/cl)" >&2
    missing=1
  fi
  if ! PY="$(find_python)"; then
    echo "missing: python3" >&2
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    echo "install tools for $(os_id):" >&2
    install_hint >&2
    exit 1
  fi
  export PY
}

run_py() {
  # shellcheck disable=SC2086
  ${PY} "$@"
}

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

require_tools

if [[ "${1}" == "merge" ]]; then
  shift
  run_py report.py merge "$@"
  exit $?
fi

NAME="$1"
if [[ ! "${NAME}" =~ ^[A-Za-z0-9._][A-Za-z0-9._-]*$ ]]; then
  die "bad name '${NAME}' (use letters/digits/._-)"
fi

if [[ -n "${ITERS:-}" ]]; then
  ITERS_VALUES=("${ITERS}")
elif [[ -n "${ITERS_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  ITERS_VALUES=(${ITERS_LIST})
else
  ITERS_VALUES=("${DEFAULT_ITERS[@]}")
fi

RESULT_DIR="results/${NAME}"
mkdir -p "${RESULT_DIR}/plots"

echo "==> host $(os_id)/$(uname -m 2>/dev/null || echo unknown) device=${NAME}"
CAPTURE_OUT="$(mktemp "${TMPDIR:-/tmp}/ex3_capture.XXXXXX" 2>/dev/null || mktemp)"
EX3_DEVICE_FOLDER="${NAME}" run_py report.py capture "${RESULT_DIR}/machine.json" > "${CAPTURE_OUT}"
while IFS= read -r line || [[ -n "${line}" ]]; do
  case "${line}" in
    EX3_*=*) export "${line}" ;;
  esac
done < "${CAPTURE_OUT}"
rm -f "${CAPTURE_OUT}"
echo "    ${EX3_HOSTNAME:-?} | ${EX3_CPU_MODEL:-?} | ${EX3_OS:-?} | ${EX3_LOGICAL_CPUS:-?} CPUs"

JOBS="$(ncpus)"
echo "==> cmake build (-j${JOBS})"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "${JOBS}"

CORRECTNESS="$(find_exe correctness_test)" || die "correctness_test not found under build/"
BENCHMARK="$(find_exe benchmark)" || die "benchmark not found under build/"

echo "==> correctness"
"${CORRECTNESS}"

total=$((${#IMPLS[@]} * ${#THREADS[@]} * ${#ITERS_VALUES[@]}))
echo "==> matrix (${total} runs) -> ${RESULT_DIR}"
echo "    iters: ${ITERS_VALUES[*]}"

for impl in "${IMPLS[@]}"; do
  for t in "${THREADS[@]}"; do
    for i in "${ITERS_VALUES[@]}"; do
      out="${RESULT_DIR}/${impl}_t${t}_i${i}.json"
      echo "  ${impl} t=${t} i=${i}"
      "${BENCHMARK}" --threads "${t}" --iters "${i}" --impl "${impl}" --out "${out}"
    done
  done
done

echo "==> aggregate + plots"
run_py report.py aggregate "${RESULT_DIR}"
if ! run_py report.py plot "${RESULT_DIR}"; then
  echo "warning: plots skipped (${PY} -m pip install matplotlib)" >&2
fi

echo "done: ${RESULT_DIR}/"
echo "copy this folder to your merge machine, then: ./run.sh merge  (or .\\run.ps1 merge)"
