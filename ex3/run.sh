#!/usr/bin/env bash
# One command per machine. Usage:
#   ./run.sh macbook_air
#   ITERS=5000 ./run.sh smoke
#   ./run.sh merge
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

THREADS=(1 2 4 8 16 32)
IMPLS=(mutex cas fetch_add)
# 10 increments-per-thread sizes from 5 → 1,000,000
DEFAULT_ITERS=(5 25 100 500 2500 10000 50000 200000 500000 1000000)

usage() {
  echo "usage: $0 <device_name>   # full study on this machine"
  echo "       $0 merge           # merge ≥3 results/* into results/report/"
  echo "quick: ITERS=5000 $0 smoke"
}

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

# --- merge 3 devices ---
if [[ "${1}" == "merge" ]]; then
  shift
  exec python3 report.py merge "$@"
fi

NAME="$1"
if [[ ! "${NAME}" =~ ^[A-Za-z0-9._][A-Za-z0-9._-]*$ ]]; then
  echo "error: bad name '${NAME}'" >&2
  exit 1
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
abs_result="$(cd "${RESULT_DIR}" && pwd)"
export RESULTS_DIR="${abs_result}"

echo "==> machine (${NAME})"
CAPTURE_OUT="$(mktemp)"
EX3_DEVICE_FOLDER="${NAME}" python3 report.py capture "${RESULT_DIR}/machine.json" > "${CAPTURE_OUT}"
while IFS= read -r line; do
  case "${line}" in
    EX3_*=*) export "${line}" ;;
  esac
done < "${CAPTURE_OUT}"
rm -f "${CAPTURE_OUT}"
echo "    ${EX3_HOSTNAME} | ${EX3_CPU_MODEL} | ${EX3_OS} | ${EX3_LOGICAL_CPUS} CPUs"

echo "==> build"
docker compose build

echo "==> correctness"
docker compose run --rm correctness

host="$(printf '%s' "${EX3_HOSTNAME}" | tr -c 'A-Za-z0-9_.-' '_')"
total=$((${#IMPLS[@]} * ${#THREADS[@]} * ${#ITERS_VALUES[@]}))
echo "==> matrix (${total} runs) -> ${RESULT_DIR}"
echo "    iters: ${ITERS_VALUES[*]}"

for impl in "${IMPLS[@]}"; do
  for threads in "${THREADS[@]}"; do
    for iters in "${ITERS_VALUES[@]}"; do
      out="/results/${host}_${impl}_t${threads}_i${iters}.json"
      echo "  ${impl} t=${threads} i=${iters}"
      docker compose run --rm \
        -e EX3_HOSTNAME -e EX3_CPU_MODEL -e EX3_LOGICAL_CPUS \
        -e EX3_OS -e EX3_MEMORY_BYTES \
        benchmark \
        --threads "${threads}" --iters "${iters}" --impl "${impl}" --out "${out}"
    done
  done
done

echo "==> aggregate + plots"
python3 report.py aggregate "${RESULT_DIR}"
python3 report.py plot "${RESULT_DIR}" || echo "warning: pip install matplotlib" >&2

echo "done: ${RESULT_DIR}/"
