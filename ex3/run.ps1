# Portable one-command study for Windows (PowerShell 5+ / 7+).
#   .\run.ps1 dangnhh
#   $env:ITERS=5000; .\run.ps1 smoke
#   .\run.ps1 merge
param(
    [Parameter(Position = 0)]
    [string]$Name
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Threads = @(1, 2, 4, 8, 16, 32)
$Impls = @("mutex", "cas", "fetch_add")
$DefaultIters = @(5, 25, 100, 500, 2500, 10000, 50000, 200000, 500000, 1000000)

function Show-Usage {
    Write-Host "usage: .\run.ps1 <device_name>   # cmake build + full study"
    Write-Host "       .\run.ps1 merge           # merge >=3 results\* into results\report\"
    Write-Host "quick: `$env:ITERS=5000; .\run.ps1 smoke"
    Write-Host ""
    Write-Host "Windows needs: cmake >= 3.16, VS Build Tools (C++), Python 3"
}

function Find-Python {
    foreach ($c in @("python", "python3", "py")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

function Require-Tools {
    $missing = $false
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Error "missing: cmake"
        $missing = $true
    }
    $py = Find-Python
    if (-not $py) {
        Write-Error "missing: python (install Python 3 and enable 'Add to PATH')"
        $missing = $true
    }
    if ($missing) {
        Write-Host @"
Install on Windows:
  1) Visual Studio Build Tools with 'Desktop development with C++'
     https://visualstudio.microsoft.com/visual-cpp-build-tools/
  2) cmake: winget install Kitware.CMake
  3) python: winget install Python.Python.3.12
  4) plots:  python -m pip install matplotlib
"@
        exit 1
    }
    return $py
}

function Find-Exe([string]$baseName) {
    $candidates = @(
        (Join-Path "build" "$baseName.exe"),
        (Join-Path "build" $baseName),
        (Join-Path "build\Release" "$baseName.exe"),
        (Join-Path "build\RelWithDebInfo" "$baseName.exe"),
        (Join-Path "build\Debug" "$baseName.exe")
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }
    throw "built binary not found: $baseName (looked under build/)"
}

if ([string]::IsNullOrWhiteSpace($Name) -or $Name -in @("-h", "--help", "/?")) {
    Show-Usage
    if ([string]::IsNullOrWhiteSpace($Name)) { exit 1 }
    exit 0
}

$Py = Require-Tools

if ($Name -eq "merge") {
    $extra = @()
    if ($args) { $extra = $args }
    & $Py report.py merge @extra
    exit $LASTEXITCODE
}

if ($Name -notmatch '^[A-Za-z0-9._][A-Za-z0-9._-]*$') {
    throw "bad name '$Name' (use letters/digits/._-)"
}

if ($env:ITERS) {
    $ItersValues = @([int64]$env:ITERS)
} elseif ($env:ITERS_LIST) {
    $ItersValues = @($env:ITERS_LIST -split '\s+' | Where-Object { $_ } | ForEach-Object { [int64]$_ })
} else {
    $ItersValues = $DefaultIters
}

$ResultDir = Join-Path "results" $Name
New-Item -ItemType Directory -Force -Path (Join-Path $ResultDir "plots") | Out-Null

Write-Host "==> host Windows device=$Name"
$env:EX3_DEVICE_FOLDER = $Name
$capture = & $Py report.py capture (Join-Path $ResultDir "machine.json")
foreach ($line in $capture) {
    if ($line -match '^(EX3_[A-Z0-9_]+)=(.*)$') {
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
}
Write-Host "    $env:EX3_HOSTNAME | $env:EX3_CPU_MODEL | $env:EX3_OS | $env:EX3_LOGICAL_CPUS CPUs"

$jobs = [Environment]::ProcessorCount
if ($jobs -lt 1) { $jobs = 2 }
Write-Host "==> cmake build (-j$jobs)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build build --config Release -j $jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Correctness = Find-Exe "correctness_test"
$Benchmark = Find-Exe "benchmark"

Write-Host "==> correctness"
& $Correctness
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$total = $Impls.Count * $Threads.Count * $ItersValues.Count
Write-Host "==> matrix ($total runs) -> $ResultDir"
Write-Host "    iters: $($ItersValues -join ' ')"

foreach ($impl in $Impls) {
    foreach ($t in $Threads) {
        foreach ($i in $ItersValues) {
            $out = Join-Path $ResultDir "${impl}_t${t}_i${i}.json"
            Write-Host "  $impl t=$t i=$i"
            & $Benchmark --threads $t --iters $i --impl $impl --out $out
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}

Write-Host "==> aggregate + plots"
& $Py report.py aggregate $ResultDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py report.py plot $ResultDir
if ($LASTEXITCODE -ne 0) {
    Write-Warning "plots skipped (python -m pip install matplotlib)"
}

Write-Host "done: $ResultDir\"
Write-Host "copy this folder to your merge machine, then: .\run.ps1 merge"
