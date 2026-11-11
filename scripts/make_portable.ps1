<#
.SYNOPSIS
    Rende la cartella dell'app autosufficiente: ambiente Python e modelli dentro.

.DESCRIPTION
    Dopo questo script la cartella di Tessitura contiene tutto il necessario:

        env\      ambiente Python con torch, Whisper, pyannote, Qt, ffmpeg
        models\   modelli scaricati (Whisper large-v3, pyannote)
        data\     registrazioni e sessioni

    Copiandola su un altro PC o su un disco esterno l'app parte senza installare
    niente. Se conda ha gia' un ambiente `tessitura` lo clona (in pochi secondi, con
    hardlink, quindi quasi senza occupare spazio in piu'); altrimenti lo crea da
    zero con setup.ps1.

.PARAMETER FromScratch
    Ignora l'ambiente `tessitura` esistente e ricrea tutto da environment.yml.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\make_portable.ps1
#>

[CmdletBinding()]
param(
    [switch] $FromScratch,
    [string] $HfToken,
    [string] $EnvName = "tessitura"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvDir = Join-Path $ProjectRoot "env"
$ModelsDir = Join-Path $ProjectRoot "models"

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}
function Write-Note($text) { Write-Host "    $text" -ForegroundColor DarkGray }

function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($c in @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

# --- 1. ambiente dentro la cartella ------------------------------------------

Write-Step "Ambiente Python in env\"
if (Test-Path (Join-Path $EnvDir "python.exe")) {
    Write-Note "C'e' gia': lo lascio com'e'. Cancella env\ per rifarlo da capo."
} else {
    $conda = Find-Conda
    $canClone = $false
    if ($conda -and -not $FromScratch) {
        $canClone = (& $conda env list) | Select-String -Pattern "^\s*$EnvName\s" -Quiet
    }
    if ($canClone) {
        Write-Note "Clono l'ambiente '$EnvName' gia' presente (usa hardlink: veloce)"
        & $conda create --prefix $EnvDir --clone $EnvName --yes
        if ($LASTEXITCODE -ne 0) { throw "Clonazione dell'ambiente fallita." }
    } else {
        Write-Note "Nessun ambiente '$EnvName' da clonare: lo creo da zero"
        $setup = Join-Path $PSScriptRoot "setup.ps1"
        $args = @("-ExecutionPolicy", "Bypass", "-File", $setup, "-Prefix", $EnvDir)
        if ($HfToken) { $args += @("-HfToken", $HfToken) }
        & powershell @args
        if ($LASTEXITCODE -ne 0) { throw "setup.ps1 non e' riuscito a creare l'ambiente." }
    }
}

# --- 2. modelli dentro la cartella -------------------------------------------

Write-Step "Modelli in models\"
$hfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
$already = @(Get-ChildItem -Path $ModelsDir -Recurse -File -ErrorAction SilentlyContinue).Count
if ($already -gt 0) {
    Write-Note "Gia' presenti ($already file): non tocco niente."
} elseif (Test-Path $hfCache) {
    $size = [math]::Round((Get-ChildItem $hfCache -Recurse -File | Measure-Object Length -Sum).Sum / 1GB, 1)
    Write-Note "Copio la cache HuggingFace gia' scaricata ($size GB): evita di riscaricarla"
    # Non Copy-Item: risolverebbe i collegamenti interni della cache, raddoppiando
    # lo spazio (i pesi stanno una volta in blobs/ e sono linkati da snapshots/).
    & (Join-Path $EnvDir "python.exe") (Join-Path $PSScriptRoot "copy_model_cache.py") $hfCache $ModelsDir
    if ($LASTEXITCODE -ne 0) { throw "Copia della cache dei modelli fallita." }
    Write-Note "Fatto. L'originale in $hfCache resta: puoi cancellarlo se non serve ad altro."
} else {
    Write-Note "Nessuna cache da copiare: i modelli si scaricheranno al primo uso."
}

# --- 3. collegamento con icona -----------------------------------------------

Write-Step "Collegamento 'Avvia Tessitura'"
$icon = Join-Path $ProjectRoot "assets\tessitura.ico"
$target = Join-Path $ProjectRoot "Avvia Tessitura.cmd"
$shortcut = Join-Path $ProjectRoot "Avvia Tessitura.lnk"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($shortcut)
$lnk.TargetPath = $target
$lnk.WorkingDirectory = $ProjectRoot
$lnk.Description = "Tessitura - registra, trascrivi e separa le voci"
$lnk.WindowStyle = 7          # avvio minimizzato: la finestra e' quella dell'app
if (Test-Path $icon) { $lnk.IconLocation = $icon }
$lnk.Save()
Write-Note "Creato: $shortcut"

$desktop = [Environment]::GetFolderPath("Desktop")
if ($desktop -and (Test-Path $desktop)) {
    $lnk2 = $ws.CreateShortcut((Join-Path $desktop "Tessitura.lnk"))
    $lnk2.TargetPath = $target
    $lnk2.WorkingDirectory = $ProjectRoot
    $lnk2.WindowStyle = 7
    if (Test-Path $icon) { $lnk2.IconLocation = $icon }
    $lnk2.Save()
    Write-Note "Creato anche sul Desktop: Tessitura"
}

# --- 4. verifica --------------------------------------------------------------

Write-Step "Verifica"
& (Join-Path $EnvDir "python.exe") (Join-Path $ProjectRoot "scripts\check_env.py")

Write-Host ""
Write-Host "Cartella pronta e autosufficiente." -ForegroundColor Green
Write-Host "Avvio: doppio clic su 'Avvia Tessitura'."
Write-Host "Per portarla altrove: copia l'intera cartella. Su un PC senza ambiente,"
Write-Host "il primo avvio lo ricrea da solo."
