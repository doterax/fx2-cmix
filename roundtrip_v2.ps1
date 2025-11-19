# Roundtrip test script for chunked preprocessor v2
# Usage: Run from repo root

$ErrorActionPreference = 'Stop'

function Find-Binary {
    param([Parameter(Mandatory=$true)][string]$Name)
    $candidates = @(
        (Join-Path $PSScriptRoot "src\preprocess_chunked_v2\$Name"),
        (Join-Path $PSScriptRoot "bin\$Name"),
        (Join-Path $PSScriptRoot $Name)
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

function Test-Roundtrip {
    param(
        [Parameter(Mandatory=$true)][string]$InputPath,
        [Parameter(Mandatory=$true)][string]$ContainerPath,
        [Parameter(Mandatory=$true)][string]$DecodedPath
    )
    $chunker = Find-Binary -Name 'chunked_v2.exe'
    if (-not $chunker) { Write-Error "chunked_v2.exe not found in expected locations" }

    & $chunker c $InputPath $ContainerPath
    & $chunker d $ContainerPath $DecodedPath

    $h1 = (Get-FileHash -Algorithm SHA256 $InputPath).Hash
    $h2 = (Get-FileHash -Algorithm SHA256 $DecodedPath).Hash
    if ($h1 -eq $h2) {
        Write-Host "ROUNDTRIP OK: $InputPath" -ForegroundColor Green
    } else {
        Write-Host "ROUNDTRIP MISMATCH: $InputPath" -ForegroundColor Red
        Write-Host "  src: $h1"
        Write-Host "  dec: $h2"
        exit 1
    }
}

function Maybe-Roundtrip {
    param([string]$InputPath)
    if (Test-Path $InputPath) {
        $container = "$InputPath.v2"
        $decoded = "$InputPath.v2.dec"
        Test-Roundtrip -InputPath $InputPath -ContainerPath $container -DecodedPath $decoded
        $statsBin = Find-Binary -Name 'stats_v2.exe'
        if ($statsBin -and (Test-Path $container)) { & $statsBin $container | Select-Object -First 25 }
        $dumpBin = Find-Binary -Name 'dump_v2.exe'
        if ($dumpBin -and (Test-Path $container)) { & $dumpBin $container | Select-Object -First 25 }
    } else {
        Write-Warning "Input not found, skipping: $InputPath"
    }
}

# Small sample
Maybe-Roundtrip -InputPath '.\prof_input\input'

# enwik8 (if present)
Maybe-Roundtrip -InputPath '.\prof_input\enwik8'
