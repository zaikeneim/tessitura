<#
.SYNOPSIS
    Prepara Tessitura su un PC Windows nuovo, in un colpo solo.

.DESCRIPTION
    Installa Miniconda se manca, crea (o aggiorna) l'ambiente conda `tessitura`,
    installa le dipendenze e sceglie da solo il build di PyTorch giusto: CUDA se
    trova una GPU NVIDIA, altrimenti CPU. Alla fine lancia la verifica ambiente.

    Lo script e' ripetibile: rilanciarlo su un ambiente gia' pronto aggiorna
    quello che serve senza rifare tutto.

.PARAMETER HfToken
    Token HuggingFace da salvare in .hf_token (serve alla diarizzazione).
    Senza questo parametro lo script spiega come ottenerlo e prosegue.

.PARAMETER Cpu
    Forza il build CPU di PyTorch anche se c'e' una GPU NVIDIA.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -HfToken hf_xxx
#>

[CmdletBinding()]
param(
    [string] $HfToken,
    [switch] $Cpu,
    [string] $EnvName = "tessitura",
    # Crea l'ambiente in questa cartella invece che fra quelli di conda: e' cio'
    # che rende l'app portabile (vedi make_portable.ps1).
    [string] $Prefix
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Come indicare l'ambiente a conda: per nome, o per percorso se e' portabile.
if ($Prefix) {
    $EnvRef = @("--prefix", $Prefix)
    $EnvLabel = $Prefix
} else {
    $EnvRef = @("--name", $EnvName)
    $EnvLabel = $EnvName
}

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Write-Note($text) {
    Write-Host "    $text" -ForegroundColor DarkGray
}

# --- 1. conda -----------------------------------------------------------------

function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

Write-Step "Cerco Miniconda"
$conda = Find-Conda
if (-not $conda) {
    Write-Note "Non trovata: la installo con winget (Anaconda.Miniconda3)."
    winget install --id Anaconda.Miniconda3 --source winget --accept-package-agreements --accept-source-agreements
    $conda = Find-Conda
    if (-not $conda) {
        throw "Miniconda risulta installata ma non la trovo. Chiudi e riapri il terminale, poi rilancia questo script."
    }
}
Write-Note "conda: $conda"

# AVG e altri antivirus intercettano HTTPS e rifirmano i certificati: senza
# truststore i download di conda/pip falliscono con CERTIFICATE_VERIFY_FAILED.
Write-Step "Configuro i certificati di sistema (antivirus che intercettano HTTPS)"
& $conda config --set ssl_verify truststore
Write-Note "ssl_verify = truststore"

# --- 2. ambiente --------------------------------------------------------------

Write-Step "Ambiente conda '$EnvLabel'"
if ($Prefix) {
    $exists = Test-Path (Join-Path $Prefix "python.exe")
} else {
    $exists = (& $conda env list) | Select-String -Pattern "^\s*$EnvName\s" -Quiet
}
if ($exists) {
    Write-Note "Esiste gia': lo aggiorno da environment.yml"
    & $conda env update @EnvRef --file "$ProjectRoot\environment.yml" --prune
} else {
    Write-Note "Lo creo da environment.yml (Python 3.11, ffmpeg, PySide6 da conda-forge)"
    & $conda env create @EnvRef --file "$ProjectRoot\environment.yml"
}
if ($LASTEXITCODE -ne 0) { throw "conda non e' riuscito a preparare l'ambiente '$EnvLabel'." }

$envPython = & $conda run @EnvRef python -c "import sys; print(sys.executable)"
if (-not $envPython) { throw "Non riesco a usare Python dentro l'ambiente '$EnvLabel'." }
Write-Note "python: $envPython"

# --- 3. dipendenze pip --------------------------------------------------------

Write-Step "Dipendenze Python (faster-whisper, pyannote, cattura audio, ...)"
& $conda run @EnvRef --no-capture-output python -m pip install --upgrade pip
& $conda run @EnvRef --no-capture-output python -m pip install -r "$ProjectRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "Installazione delle dipendenze pip fallita." }

# --- 4. PyTorch: CUDA o CPU a seconda del PC ---------------------------------

Write-Step "PyTorch adatto a questo PC"
$hasNvidia = $false
if (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader
            if ($gpu) {
                $hasNvidia = $true
                Write-Note "GPU trovata: $($gpu -join ', ')"
            }
        } catch {
            $hasNvidia = $false
        }
    }
}

if ($hasNvidia) {
    # pyannote.audio tira il torch CPU da PyPI: va forzato il build CUDA.
    # cu126 (non cu130) perche' ctranslate2/faster-whisper vuole le librerie CUDA 12.
    # Serve --force-reinstall --no-deps: "torch==2.13.0" risulta gia' soddisfatto
    # dal build +cpu, dato che "+cu126" e' solo una local version.
    Write-Note "Installo i build CUDA (cu126) di torch e torchaudio"
    & $conda run @EnvRef --no-capture-output python -m pip install `
        --force-reinstall --no-deps torch==2.13.0 torchaudio==2.11.0 `
        --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "Installazione di PyTorch CUDA fallita." }
} else {
    Write-Note "Nessuna GPU NVIDIA: resta il build CPU (l'app scegliera' da sola un modello piu' leggero)."
}

# --- 5. token HuggingFace -----------------------------------------------------

Write-Step "Token HuggingFace (serve alla diarizzazione)"
$tokenFile = Join-Path $ProjectRoot ".hf_token"
if ($HfToken) {
    Set-Content -Path $tokenFile -Value $HfToken.Trim() -Encoding utf8 -NoNewline
    Write-Note "Salvato in .hf_token"
} elseif (Test-Path $tokenFile) {
    Write-Note "Gia' presente in .hf_token"
} else {
    Write-Host "    Manca. Per averlo:" -ForegroundColor Yellow
    Write-Host "      1. crea un account su https://huggingface.co"
    Write-Host "      2. accetta le condizioni di pyannote/speaker-diarization-community-1"
    Write-Host "      3. crea un token in Settings -> Access Tokens"
    Write-Host "      4. rilancia:  .\scripts\setup.ps1 -HfToken hf_xxxxx"
    Write-Host "    Senza token la trascrizione funziona, la separazione delle voci no."
}

# --- 6. verifica --------------------------------------------------------------

Write-Step "Verifica ambiente"
& $conda run @EnvRef --no-capture-output python "$ProjectRoot\scripts\check_env.py"

Write-Host ""
Write-Host "Pronto. Per avviare l'app:" -ForegroundColor Green
if ($Prefix) {
    Write-Host "    doppio clic su 'Avvia Tessitura' nella cartella dell'app"
} else {
    Write-Host "    conda activate $EnvName"
    Write-Host "    python main.py"
    Write-Host "oppure, senza attivare nulla:" -ForegroundColor Green
    Write-Host "    .\scripts\start.ps1"
}
Write-Host ""
Write-Host "Per condividere registrazioni e sessioni fra piu' PC, punta la cartella dati" -ForegroundColor Green
Write-Host "a una cartella sincronizzata:  `$env:TESSITURA_DATA_DIR = 'D:\tessitura-dati'"
