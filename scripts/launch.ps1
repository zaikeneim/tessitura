[CmdletBinding()]
param([string] $DataDir, [switch] $Console)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\runtime.ps1"
$root = Split-Path -Parent $PSScriptRoot
try {
    $state = Get-RuntimeState $root
    $runtime = Get-RuntimePath $root $state.active
    if ($DataDir) { $env:TESSITURA_DATA_DIR = $DataDir }
    $python = Join-Path $runtime $(if ($Console) { 'python.exe' } else { 'pythonw.exe' })
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Ambiente non trovato: $runtime. Eseguire scripts\make_portable.ps1."
    }
    if ($Console) {
        & $python (Join-Path $root 'main.py')
        exit $LASTEXITCODE
    }
    # Niente -WindowStyle Hidden: Windows propaga SW_HIDE dello STARTUPINFO alla
    # prima finestra del processo, quindi la finestra di Qt nascerebbe nascosta
    # (app avviata, nessuna finestra). pythonw.exe non apre console comunque.
    Start-Process -FilePath $python -ArgumentList ('"' + (Join-Path $root 'main.py') + '"') -WorkingDirectory $root
} catch {
    Write-Error "$($_.Exception.Message) Per il ripristino: scripts\update.ps1 -Rollback"
    exit 1
}
