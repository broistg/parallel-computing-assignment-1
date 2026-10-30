# Exercise 3 — concurrent counters

`mutex` · `cas` (+ backoff) · `fetch_add`

Same study on **macOS, Linux, and Windows**.

## One command

| OS | Command |
| --- | --- |
| macOS / Linux | `./run.sh dangnhh5` |
| Windows PowerShell | `.\run.ps1 dangnhh5` |
| Windows Git Bash | `./run.sh dangnhh5` |

Smoke: `ITERS=5000 ./run.sh smoke` or `$env:ITERS=5000; .\run.ps1 smoke`

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
1. [VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — workload **Desktop development with C++**
2. `winget install Kitware.CMake Python.Python.3.12`
3. Open **x64 Native Tools** or a normal PowerShell (cmake must be on PATH)
4. `python -m pip install matplotlib`
5. `.\run.ps1 dangnhh5`

## 3 machines → one report

On each PC:
```text
./run.sh lab1     or     .\run.ps1 lab1
```

Copy the three `results/<name>/` folders onto one machine, then:
```text
./run.sh merge     or     .\run.ps1 merge
```

→ `results/report/` (`machines.md`, CSV, compare plots).
