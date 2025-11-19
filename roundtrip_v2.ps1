# Experimental roundtrip script for chunked preprocessor (v3 NUMBER + external lengths)
# Usage: pwsh -File .\roundtrip_v2.ps1

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

function Test-RoundtripV3 {
    param(
        [Parameter(Mandatory=$true)][string]$InputPath,
        [Parameter(Mandatory=$true)][string]$ContainerPath,
        [Parameter(Mandatory=$true)][string]$DecodedPath
    )
    $chunker = Find-Binary -Name 'chunked_main.exe'
    if (-not $chunker) { Write-Error "chunked_main.exe not found (build with: make chunked-tools)" }

    Write-Host "Compress: $InputPath -> $ContainerPath" -ForegroundColor Cyan
    & $chunker c $InputPath $ContainerPath | Write-Verbose
    if (-not (Test-Path "$InputPath.numbers")) { } # legacy ignore
    $sidecar = "$InputPath.numbers"
    if (-not (Test-Path $sidecar)) { $sidecar = "$ContainerPath.numbers" }
    if (-not (Test-Path $sidecar)) { Write-Warning "Numbers sidecar missing after compression" }

    Write-Host "Decompress: $ContainerPath -> $DecodedPath" -ForegroundColor Cyan
    & $chunker d $ContainerPath $DecodedPath | Write-Verbose

    if (-not (Test-Path $DecodedPath)) { Write-Error "Decoded output missing" }

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

    $statsBin = Find-Binary -Name 'stats_v2.exe'
    if ($statsBin -and (Test-Path $ContainerPath)) {
        Write-Host "Stats:" -ForegroundColor Yellow
        & $statsBin $ContainerPath | Select-Object -First 50
    }
    $dumpBin = Find-Binary -Name 'dump_v2.exe'
    if ($dumpBin -and (Test-Path $ContainerPath)) {
        Write-Host "Dump (first chunks):" -ForegroundColor Yellow
        & $dumpBin $ContainerPath | Select-Object -First 50
    }
}

function Maybe-RoundtripV3 {
    param([string]$InputPath)
    if (Test-Path $InputPath) {
        $container = "$InputPath.v3"
        $decoded = "$InputPath.v3.dec"
        Test-RoundtripV3 -InputPath $InputPath -ContainerPath $container -DecodedPath $decoded
    } else { Write-Warning "Input not found, skipping: $InputPath" }
}

# Ensure a small synthetic sample exists for NUMBER, wrappers, XML
$sample = '.\prof_input\sample.txt'
if (-not (Test-Path $sample)) {
    @(
        'Title: Example12345',
        '[[BRACKETS WRAP TEST]]',
        '==Header Section==',
        '{{Curly Block Data}}',
        '<note attr="x">XMLContentHere</note>',
        'Numbers: 7 42 1000 00012 999999',
        'Mix: ABCDEFG end.'
    ) | Set-Content -NoNewline:$false -Encoding UTF8 $sample
}

Write-Host "--- ROUNDTRIP SMALL SAMPLE (v3) ---" -ForegroundColor Magenta
Maybe-RoundtripV3 -InputPath $sample

Write-Host "--- ROUNDTRIP ENWIK8 (if present) ---" -ForegroundColor Magenta
Maybe-RoundtripV3 -InputPath '.\prof_input\enwik8'
