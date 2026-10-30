# Exercise 3 — concurrent counters

`mutex` · `cas` (+ backoff) · `fetch_add`

## Per machine (do this on 3 devices)

```bash
./run.sh macbook_air
# quick smoke:
ITERS=5000 ./run.sh smoke
```

Runs correctness + **10 iters** (`5 … 1_000_000`) × threads `{1,2,4,8,16,32}` × 3 impls → `results/<name>/`.

## Merge for the report

Copy the three `results/<name>/` folders onto one PC, then:

```bash
./run.sh merge
```

→ `results/report/` (`machines.md`, combined CSV, compare plots). Needs matplotlib: `pip install matplotlib`.

## Layout

| Path | What |
| --- | --- |
| `src/` | C++ counters + benchmark |
| `run.sh` | one-command study / merge |
| `report.py` | machine capture, CSV, plots, merge |
| `results/` | per-device + report output |
| `Dockerfile` / `docker-compose.yml` / `CMakeLists.txt` | build |
