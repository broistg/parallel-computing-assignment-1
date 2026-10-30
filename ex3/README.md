# Exercise 3 — concurrent counters

C++ concurrent counters with **mutex**, **compare-and-swap** (+ exponential backoff), and **fetch-and-add**.

Native one-command study on **macOS, Linux, and Windows**: correctness tests, execution-time matrix, machine info, CSV, and plots.

## Quick start

| OS | Full study on this machine |
| --- | --- |
| macOS / Linux | `./run.sh <device_name>` |
| Windows PowerShell | `.\run.ps1 <device_name>` |

Example:

```bash
./run.sh dangnhh5
```

Smoke (one iters size):

```bash
ITERS=5000 ./run.sh smoke
# Windows:  $env:ITERS=5000; .\run.ps1 smoke
```

## What each run does

1. Records host info → `results/<name>/machine.json`
2. `cmake` Release build
3. Correctness tests (`get() == threads × iters`)
4. Benchmark matrix → JSON under `results/<name>/`
5. `summary.csv` + plots under `results/<name>/plots/`

### Matrix

| Axis | Values |
| --- | --- |
| impl | `mutex`, `cas`, `fetch_add` |
| threads | `1 2 4 8 16 32` |
| iters / thread | `5 25 100 500 2500 10000 50000 200000 500000 1000000` (10 sizes) |

= **180** timed runs per machine.

## Setup once per machine

**macOS**

```bash
xcode-select --install
brew install cmake python3
python3 -m pip install --user matplotlib
```

**Linux (Ubuntu/Debian)**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3 python3-pip
python3 -m pip install --user matplotlib
```

**Windows**

1. [VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — **Desktop development with C++**
2. `winget install Kitware.CMake Python.Python.3.12`
3. Ensure `cmake` and `python` are on PATH
4. `python -m pip install matplotlib`
5. `.\run.ps1 dangnhh5`

## Three platforms → report

Assignment needs results on enough shared-memory machines (≥ group size).

1. On each machine: `./run.sh <name>` or `.\run.ps1 <name>`
2. Copy all `results/<name>/` folders onto one PC
3. Merge:

```bash
./run.sh merge
# Windows: .\run.ps1 merge
```

Output: `results/report/` — `machines.md`, combined `summary.csv`, per-device and cross-device plots.

## Layout

| Path | Role |
| --- | --- |
| `src/` | counters, benchmark, correctness, host info |
| `run.sh` / `run.ps1` | one-command study / merge |
| `report.py` | capture, aggregate, plot, merge |
| `results/<name>/` | per-machine official data |
| `CMakeLists.txt` | native C++17 build |
