#Requires -Version 5.1
<#
.SYNOPSIS
    Benchmark LSTM architecture configurations on enwik7.

.DESCRIPTION
    Runs compress + decompress for each (cells, layers) config, measures:
      - Wall time  : total elapsed time including I/O and preprocessing
      - Log time   : time reported by cmix inside "N bytes -> M bytes in T s."
      - Overhead   : wall - log  (I/O, preprocessing, vocab, linking)
      - Output size: compressed bytes
      - BPC        : bits per character  (output_bytes * 8 / input_bytes)

    Results are written to benchmark_results.csv and printed as a table.

.PARAMETER Input
    Input file to compress (default: .\prof_input\enwik7).

.PARAMETER Dict
    Dictionary file (default: .\dictionary\words_enwik8_opt.dic).
    Pass "" or $null to run without dictionary.

.PARAMETER Configs
    Hashtable array of @{cells=...; layers=...} to test.
    Default: the full test matrix.

.PARAMETER OutFile
    CSV output path (default: benchmark_results.csv).

.EXAMPLE
    .\benchmark_lstm.ps1
    .\benchmark_lstm.ps1 -InputFile .\prof_input\enwik7 -Dict ""
    .\benchmark_lstm.ps1 -Configs @(@{cells=200;layers=1},@{cells=256;layers=1})
#>
param(
    [string]$InputFile = ".\prof_input\enwik7",
    [string]$Dict    = ".\dictionary\words_enwik8_opt.dic",
    [object[]]$Configs = @(
        @{ cells = 128; layers = 1 },   # small / speed baseline
        @{ cells = 200; layers = 1 },   # current default
        @{ cells = 256; layers = 1 },   # wider (+28% params)
        @{ cells = 320; layers = 1 },   # wider (+60% params)
        @{ cells = 128; layers = 2 },   # two-layer narrow
        @{ cells = 200; layers = 2 },   # two-layer default-width
        @{ cells = 256; layers = 2 }    # two-layer wide (richest / slowest)
    ),
    [string]$OutFile = "benchmark_results.csv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve cmix binary
$cmix = Join-Path $PSScriptRoot "cmix.exe"
if (-not (Test-Path $cmix)) {
    Write-Error "cmix.exe not found at $cmix"
    exit 1
}

# Validate input
if (-not (Test-Path $InputFile)) {
    Write-Error "Input file not found: $InputFile"
    exit 1
}
$inputBytes = (Get-Item $InputFile).Length

# Dict args
$dictArgs = @()
if ($Dict -and (Test-Path $Dict)) {
    $dictArgs = @("-d", $Dict)
    Write-Host "Dictionary : $Dict"
} else {
    Write-Host "Dictionary : (none)"
}

Write-Host "Input      : $InputFile  ($inputBytes bytes)"
Write-Host "Configs    : $($Configs.Count)"
Write-Host ""

$results = [System.Collections.Generic.List[PSCustomObject]]::new()

$colW = 8   # column width for table

# Header
$hdr = "{0,-$colW} {1,-$colW} {2,10} {3,10} {4,10} {5,8} {6,8}" -f `
    "cells", "layers", "wall(s)", "log(s)", "ovhd(s)", "outKB", "bpc"
Write-Host $hdr
Write-Host ("-" * $hdr.Length)

foreach ($cfg in $Configs) {
    $cells  = [int]$cfg.cells
    $layers = [int]$cfg.layers

    $outBin = Join-Path $PSScriptRoot "bench_${cells}_${layers}.bin"

    # --- Compress ---
    $wallStart = [System.Diagnostics.Stopwatch]::StartNew()
    $rawOutput = & $cmix --lstm-num-cells $cells --lstm-num-layers $layers `
        compress @dictArgs $InputFile $outBin 2>&1
    $wallStart.Stop()
    $wallSec = $wallStart.Elapsed.TotalSeconds

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  cells=$cells layers=$layers  cmix FAILED (exit $LASTEXITCODE)"
        continue
    }

    # Parse cmix summary line: "N bytes -> M bytes in T s."
    $logSec    = $null
    $outBytes  = $null
    $summaryLine = ($rawOutput | Select-String "bytes -> \d+ bytes in") | Select-Object -Last 1
    if ($summaryLine) {
        if ($summaryLine.Line -match '(\d+) bytes -> (\d+) bytes in ([\d.]+) s') {
            $outBytes = [long]$Matches[2]
            $logSec   = [double]$Matches[3]
        }
    }

    # Fallback: get output file size
    if ($null -eq $outBytes -and (Test-Path $outBin)) {
        $outBytes = (Get-Item $outBin).Length
    }

    $overheadSec = if ($null -ne $logSec) { $wallSec - $logSec } else { $null }
    $bpc         = if ($null -ne $outBytes -and $inputBytes -gt 0) {
                       [math]::Round($outBytes * 8.0 / $inputBytes, 4) } else { $null }
    $outKB       = if ($null -ne $outBytes) { [math]::Round($outBytes / 1024.0, 1) } else { $null }

    # Cleanup
    if (Test-Path $outBin) { Remove-Item $outBin -Force }

    $row = [PSCustomObject]@{
        cells       = $cells
        layers      = $layers
        wall_s      = [math]::Round($wallSec, 2)
        log_s       = if ($null -ne $logSec) { [math]::Round($logSec, 2) } else { "n/a" }
        overhead_s  = if ($null -ne $overheadSec) { [math]::Round($overheadSec, 2) } else { "n/a" }
        out_kb      = if ($null -ne $outKB) { $outKB } else { "n/a" }
        bpc         = if ($null -ne $bpc) { $bpc } else { "n/a" }
    }
    $results.Add($row)

    $line = "{0,-$colW} {1,-$colW} {2,10} {3,10} {4,10} {5,8} {6,8}" -f `
        $row.cells, $row.layers, $row.wall_s, $row.log_s, $row.overhead_s, $row.out_kb, $row.bpc
    Write-Host $line
}

Write-Host ""

# Export CSV
$results | Export-Csv -Path $OutFile -NoTypeInformation -Encoding UTF8
Write-Host "Results saved to $OutFile"
