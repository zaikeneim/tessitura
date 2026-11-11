#Requires -Version 5.1
<#
.SYNOPSIS
    Reproducible Windows build and installer script for Tessitura.
.DESCRIPTION
    1. Validates SemVer version and writes build_config.json.
    2. Optionally runs automated verification tests.
    3. Builds dist\Tessitura\Tessitura.exe via PyInstaller (onedir).
    4. Smoke-tests dist\Tessitura\Tessitura.exe (--version and --smoke-test).
    5. Compiles dist\Tessitura-Setup-{Version}.exe via Inno Setup 6 (ISCC.exe).
    6. Optionally signs binaries with Authenticode if a certificate is configured.
    7. Generates dist\SHA256SUMS.txt.
#>
[CmdletBinding()]
param(
    [string]$Version = "",
    [ValidateSet("stable", "beta")]
    [string]$Channel = "stable",
    [string]$GitHubOwner = "aless",
    [string]$GitHubRepo = "tessitura",
    [ValidateSet("strict", "checksum_only")]
    [string]$TrustPolicy = "strict",
    [switch]$LightBundle,
    [switch]$SkipTests,
    [switch]$Clean,
    [switch]$SkipPyInstaller,
    [string]$SignCertThumbprint = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot "env\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCmd) {
        throw "Python executable not found in .\env\python.exe or PATH."
    }
    $PythonExe = $PythonCmd.Source
}
$EnvRoot = Split-Path -Parent $PythonExe
$env:PATH = "$EnvRoot\Library\bin;$EnvRoot\Scripts;$EnvRoot;$env:PATH"

# 1. Resolve and validate SemVer version
if ([string]::IsNullOrWhiteSpace($Version)) {
    if (-not [string]::IsNullOrWhiteSpace($env:TESSITURA_VERSION)) {
        $Version = $env:TESSITURA_VERSION
    } else {
        $Version = (& $PythonExe -c "from src.version import APP_VERSION; print(APP_VERSION)").Trim()
    }
}

$NormalizedVersion = (& $PythonExe -c "import sys; from src.version import SemVer; v = SemVer.parse(sys.argv[1]); print(v.version_without_build)" $Version).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($NormalizedVersion)) {
    throw "Invalid SemVer version: '$Version'"
}
$BaseNumericVersion = (& $PythonExe -c "import sys; from src.version import SemVer; v = SemVer.parse(sys.argv[1]); print(f'{v.major}.{v.minor}.{v.patch}.0')" $NormalizedVersion).Trim()

Write-Host "==> Building Tessitura v$NormalizedVersion (channel=$Channel, trust_policy=$TrustPolicy)" -ForegroundColor Cyan

# 2. Run automated tests unless skipped
if (-not $SkipTests) {
    Write-Host "==> Running unit test suite..." -ForegroundColor Cyan
    & $PythonExe -m unittest discover -s tests -p "test_*.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed! Aborting Windows release build."
    }
}

# 3. Ensure icon exists
$IconPath = Join-Path $ProjectRoot "assets\tessitura.ico"
if (-not (Test-Path $IconPath)) {
    Write-Host "==> Generating assets\tessitura.ico..." -ForegroundColor Cyan
    & $PythonExe -m scripts.make_icon
}

# 4. Write build_config.json
$CommitSha = ""
try {
    $CommitSha = (git rev-parse --short HEAD 2>$null).Trim()
} catch {}
$BuildTimestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

$BuildConfig = [ordered]@{
    app_name           = "Tessitura"
    app_id             = "com.tessitura.desktop"
    version            = $NormalizedVersion
    github_owner       = $GitHubOwner
    github_repo        = $GitHubRepo
    default_channel    = $Channel
    trust_policy       = $TrustPolicy
    trusted_publishers = @()
    build_timestamp    = $BuildTimestamp
    commit_sha         = $CommitSha
}
$BuildConfigPath = Join-Path $ProjectRoot "build_config.json"
$BuildConfigJson = $BuildConfig | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($BuildConfigPath, $BuildConfigJson, (New-Object System.Text.UTF8Encoding($false)))

# 5. Run PyInstaller (onedir)
$PyInstallerExe = Join-Path (Split-Path -Parent $PythonExe) "Scripts\pyinstaller.exe"
if (-not (Test-Path $PyInstallerExe)) {
    $PyInstallerExe = Join-Path (Split-Path -Parent $PythonExe) "pyinstaller.exe"
}

$PrevLight = $env:TESSITURA_LIGHT_BUNDLE
try {
    if ($LightBundle) {
        $env:TESSITURA_LIGHT_BUNDLE = "1"
    } else {
        $env:TESSITURA_LIGHT_BUNDLE = "0"
    }

    if (-not $SkipPyInstaller) {
        Write-Host "==> Packaging Tessitura.exe with PyInstaller (LightBundle=$LightBundle)..." -ForegroundColor Cyan
        $PyiArgs = @("--noconfirm")
        if ($Clean) { $PyiArgs += "--clean" }
        $PyiArgs += "packaging/windows/tessitura.spec"
        if (Test-Path $PyInstallerExe) {
            & $PyInstallerExe @PyiArgs
        } else {
            & $PythonExe -m PyInstaller @PyiArgs
        }
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "==> Reusing existing dist\Tessitura bundle..." -ForegroundColor Cyan
    }
} finally {
    $env:TESSITURA_LIGHT_BUNDLE = $PrevLight
}

$DistAppDir = Join-Path $ProjectRoot "dist\Tessitura"
$TessituraExe = Join-Path $DistAppDir "Tessitura.exe"
if (-not (Test-Path $TessituraExe)) {
    throw "Expected executable not found at $TessituraExe"
}
Copy-Item -Path $BuildConfigPath -Destination (Join-Path $DistAppDir "build_config.json") -Force
if (Test-Path (Join-Path $DistAppDir "_internal")) {
    Copy-Item -Path $BuildConfigPath -Destination (Join-Path $DistAppDir "_internal\build_config.json") -Force
}

# 6. Smoke-test Tessitura.exe before building installer
Write-Host "==> Running smoke test on packaged Tessitura.exe..." -ForegroundColor Cyan
$SmokeOut = Join-Path $ProjectRoot "dist\smoke_test_result.json"
if (Test-Path $SmokeOut) { Remove-Item $SmokeOut -Force }
$env:TESSITURA_SMOKE_OUT = $SmokeOut
try {
    $Proc = Start-Process -FilePath $TessituraExe -ArgumentList "--smoke-test" -Wait -PassThru -NoNewWindow
    if ($Proc.ExitCode -ne 0) {
        throw "Tessitura.exe --smoke-test exited with code $($Proc.ExitCode)"
    }
    if (-not (Test-Path $SmokeOut)) {
        throw "Tessitura.exe --smoke-test did not write output file $SmokeOut"
    }
    $SmokeData = Get-Content $SmokeOut -Raw | ConvertFrom-Json
    Write-Host "    Smoke test OK: $($SmokeData.app_name) v$($SmokeData.version) (frozen=$($SmokeData.frozen))" -ForegroundColor Green
} finally {
    Remove-Item Env:\TESSITURA_SMOKE_OUT -ErrorAction SilentlyContinue
    if (Test-Path $SmokeOut) { Remove-Item $SmokeOut -Force }
}

# 7. Optional Authenticode signing of Tessitura.exe
if (-not [string]::IsNullOrWhiteSpace($SignCertThumbprint)) {
    Write-Host "==> Signing Tessitura.exe with certificate $SignCertThumbprint..." -ForegroundColor Cyan
    & signtool sign /sha1 $SignCertThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $TessituraExe
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to sign Tessitura.exe"
    }
}

# 8. Compile Windows Installer via Inno Setup 6 (ISCC.exe)
$IsccCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$IsccExe = $null
foreach ($Cand in $IsccCandidates) {
    if (Test-Path $Cand) {
        $IsccExe = $Cand
        break
    }
}
if ($null -eq $IsccExe) {
    $IsccCmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($null -ne $IsccCmd) {
        $IsccExe = $IsccCmd.Source
    }
}
if ($null -eq $IsccExe) {
    throw "Inno Setup 6 compiler (ISCC.exe) not found. Install JRSoftware.InnoSetup."
}

$IssPath = Join-Path $ProjectRoot "packaging\windows\tessitura.iss"
$OutDir = Join-Path $ProjectRoot "dist"
Write-Host "==> Compiling Windows installer with Inno Setup ($IsccExe)..." -ForegroundColor Cyan
& $IsccExe "/Q" "/DMyAppVersion=$NormalizedVersion" "/DMyAppNumericVersion=$BaseNumericVersion" "/DMyDistDir=$DistAppDir" "/DMyOutputDir=$OutDir" $IssPath
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE"
}

$SetupExe = Join-Path $ProjectRoot "dist\Tessitura-Setup-$NormalizedVersion.exe"
if (-not (Test-Path $SetupExe)) {
    throw "Expected installer not found at $SetupExe"
}

if (-not [string]::IsNullOrWhiteSpace($SignCertThumbprint)) {
    Write-Host "==> Signing installer $SetupExe..." -ForegroundColor Cyan
    & signtool sign /sha1 $SignCertThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $SetupExe
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to sign $SetupExe"
    }
}

# 9. Generate dist\SHA256SUMS.txt
$ShaFile = Join-Path $ProjectRoot "dist\SHA256SUMS.txt"
$SetupHash = (Get-FileHash -Path $SetupExe -Algorithm SHA256).Hash.ToLowerInvariant()
$SetupName = Split-Path -Leaf $SetupExe
$ShaLines = @(
    "$SetupHash  $SetupName"
)
[System.IO.File]::WriteAllLines($ShaFile, $ShaLines, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "==> Build completed successfully!" -ForegroundColor Green
Write-Host "    Executable : $TessituraExe"
Write-Host "    Installer  : $SetupExe"
Write-Host "    Checksums  : $ShaFile ($SetupHash)"
