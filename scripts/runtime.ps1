# Shared runtime selection. Candidates stay at their original conda prefix;
# switching a tiny JSON pointer never renames or edits a working environment.
Set-StrictMode -Version Latest

function Get-RuntimePath([string] $Root, [string] $Relative) {
    $base = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    if ([IO.Path]::IsPathRooted($Relative)) { throw 'Runtime path must be relative.' }
    $path = [IO.Path]::GetFullPath((Join-Path $base $Relative))
    $default = Join-Path $base 'env'
    $stages = (Join-Path $base '.runtimes') + [IO.Path]::DirectorySeparatorChar
    if ($path -ne $default -and -not $path.StartsWith($stages, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Runtime path outside the permitted directories.'
    }
    # A junction could bypass the lexical check above.
    $cursor = $path
    while ($cursor -and $cursor -ne $base) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Runtime junction/symlink is not allowed: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $path
}

function Get-RuntimeState([string] $Root) {
    $pointer = Join-Path $Root '.runtime.json'
    if (-not (Test-Path -LiteralPath $pointer)) {
        return [pscustomobject]@{ active = 'env'; previous = $null }
    }
    # An invalid pointer fails closed: do not silently start a different runtime.
    $state = Get-Content -LiteralPath $pointer -Raw | ConvertFrom-Json
    $null = Get-RuntimePath $Root $state.active
    if ($state.previous) { $null = Get-RuntimePath $Root $state.previous }
    return $state
}

function Set-RuntimeState([string] $Root, [string] $Active, $Previous) {
    $path = Get-RuntimePath $Root $Active
    if (-not (Test-Path -LiteralPath (Join-Path $path 'python.exe'))) {
        throw "Runtime incomplete: $path"
    }
    if ($Previous) { $null = Get-RuntimePath $Root $Previous }
    $pointer = Join-Path $Root '.runtime.json'
    $temp = Join-Path $Root ('.runtime-' + [guid]::NewGuid().ToString('N') + '.tmp')
    $json = @{ active = $Active; previous = $Previous } | ConvertTo-Json
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    $file = [IO.File]::Open($temp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $file.Write($bytes, 0, $bytes.Length); $file.Flush($true) } finally { $file.Dispose() }
    try {
        if (Test-Path -LiteralPath $pointer) {
            [IO.File]::Replace($temp, $pointer, (Join-Path $Root '.runtime.previous.json'))
        } else {
            [IO.File]::Move($temp, $pointer)
        }
    } finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
    }
}

function Find-TessituraConda {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($path in @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe")) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    throw 'Conda non trovata. Eseguire scripts\setup.ps1.'
}

function Invoke-Checked([string] $Exe, [string[]] $Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Comando fallito ($LASTEXITCODE): $Exe" }
}
