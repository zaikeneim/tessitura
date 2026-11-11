<#
.SYNOPSIS
    Avvia Tessitura senza dover attivare l'ambiente conda a mano.

.PARAMETER DataDir
    Cartella dati da usare (registrazioni e sessioni). Utile per tenerle su una
    cartella sincronizzata e ritrovarle su un altro PC.

.EXAMPLE
    .\scripts\start.ps1
    .\scripts\start.ps1 -DataDir D:\tessitura-dati
#>

[CmdletBinding()]
param(
    [string] $DataDir,
    [string] $EnvName = "tessitura"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

if ($EnvName -eq 'tessitura' -and ((Test-Path -LiteralPath (Join-Path $ProjectRoot '.runtime.json')) -or
    (Test-Path -LiteralPath (Join-Path $ProjectRoot 'env\python.exe')))) {
    & "$PSScriptRoot\launch.ps1" -DataDir $DataDir -Console
    exit $LASTEXITCODE
}

$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) {
    $condaExe = $conda.Source
} else {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )
    $condaExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $condaExe) {
    throw "Conda non trovata. Lancia prima:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1"
}

if ($DataDir) {
    $env:TESSITURA_DATA_DIR = $DataDir
    Write-Host "Cartella dati: $DataDir" -ForegroundColor DarkGray
}

& $condaExe run -n $EnvName --no-capture-output python "$ProjectRoot\main.py"
