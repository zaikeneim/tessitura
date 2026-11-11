<#
.SYNOPSIS
    Aggiornamento isolato con attivazione atomica e rollback immediato.
#>
[CmdletBinding()]
param([switch] $Check, [switch] $Rollback, [switch] $Cpu)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\runtime.ps1"
$root = Split-Path -Parent $PSScriptRoot
if ($Cpu) { throw 'Il cambio del backend non fa parte di un aggiornamento sicuro. Il backend esistente viene conservato.' }
if ($Check -and $Rollback) { throw 'Scegliere -Check oppure -Rollback.' }
if ($Check) {
    $state = Get-RuntimeState $root
    $source = Get-RuntimePath $root $state.active
    Invoke-Checked (Join-Path $source 'python.exe') @('-m','pip','list','--outdated','--timeout','15','--retries','1')
    Write-Host 'Sola diagnosi. Le versioni pubblicate non sono necessariamente compatibili.'
    exit 0
}
# FileShare.None is an OS lock: automatically released even if the process dies.
$lock = [IO.File]::Open((Join-Path $root '.update.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    try { $state = Get-RuntimeState $root } catch {
        if (-not $Rollback) { throw }
        $backup = Join-Path $root '.runtime.previous.json'
        $previous = 'env'
        if (Test-Path -LiteralPath $backup) {
            $previous = (Get-Content -LiteralPath $backup -Raw | ConvertFrom-Json).active
        }
        $state = [pscustomobject]@{ active = $null; previous = $previous }
    }
    if ($Rollback) {
        if (-not $state.previous) { throw 'Nessun ambiente precedente registrato.' }
        Set-RuntimeState $root $state.previous $state.active
        Write-Host 'Ripristino completato. Riavviare app.'
        exit 0
    }
    $source = Get-RuntimePath $root $state.active
    $conda = Find-TessituraConda
    if (-not (Test-Path -LiteralPath (Join-Path $source 'python.exe'))) {
        throw 'Creare prima ambiente portabile con scripts\make_portable.ps1.'
    }
    $relative = '.runtimes/' + [guid]::NewGuid().ToString('N')
    $candidate = Get-RuntimePath $root $relative
    $null = New-Item -ItemType Directory -Path (Join-Path $root '.runtimes') -Force
    $log = Join-Path $root ('.runtimes/update-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    Start-Transcript -LiteralPath $log | Out-Null
    try {
        # No source changes; a network error, Ctrl+C or power loss leaves it usable.
        Write-Host 'Creazione di una copia indipendente. Sono necessari diversi GB liberi.'
        Invoke-Checked $conda @('create','--prefix',$candidate,'--clone',$source,'--copy','--yes')
        $python = Join-Path $candidate 'python.exe'
        $baseline = Join-Path $candidate 'tessitura-baseline.json'
        Invoke-Checked $python @("$PSScriptRoot\runtime_health.py",'--snapshot',$baseline)
        $constraints = Join-Path $candidate 'tessitura-constraints.txt'
        Invoke-Checked $python @("$PSScriptRoot\runtime_health.py",'--constraints',$constraints)
        Invoke-Checked $conda @('env','update','--prefix',$candidate,'--file',"$root\environment.yml")
        Invoke-Checked $python @('-m','pip','install','--upgrade','--timeout','30','--retries','2','--constraint',$constraints,'-r',"$root\requirements.txt")
        Invoke-Checked $python @('-m','pip','check')
        Invoke-Checked $python @("$PSScriptRoot\runtime_health.py",'--verify',$baseline)
        Push-Location $root
        try {
            Invoke-Checked $python @('-m','unittest','discover','-s','tests','-v')
            Invoke-Checked $python @('tests/test_merge.py')
            Invoke-Checked $python @('tests/test_paths.py')
        } finally { Pop-Location }
        # This is the ONLY operation that changes which environment gets launched.
        Set-RuntimeState $root $relative $state.active
        Write-Host 'Aggiornamento verificato e attivato. Riavviare app.'
        Write-Host 'Il precedente ambiente resta intatto: scripts\update.ps1 -Rollback'
    } finally { Stop-Transcript -ErrorAction SilentlyContinue | Out-Null }
} catch {
    Write-Error "Aggiornamento non attivato: $($_.Exception.Message). Ambiente precedente non è stato modificato."
    exit 1
} finally { $lock.Dispose() }
