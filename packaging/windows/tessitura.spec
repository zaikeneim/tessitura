# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir specification for Tessitura (Windows x64).

Usage:
    pyinstaller --noconfirm --clean packaging/windows/tessitura.spec
"""

from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

PROJECT_ROOT = Path(SPECPATH).resolve().parent.parent
ICON_PATH = PROJECT_ROOT / "assets" / "tessitura.ico"
BUILD_CONFIG_PATH = PROJECT_ROOT / "build_config.json"

datas = [
    (str(PROJECT_ROOT / "assets"), "assets"),
]
if BUILD_CONFIG_PATH.is_file():
    datas.append((str(BUILD_CONFIG_PATH), "."))

binaries = []
hiddenimports = [
    "src",
    "src.app",
    "src.bootstrap",
    "src.config",
    "src.hardware",
    "src.migrations",
    "src.paths",
    "src.update_manager",
    "src.updates",
    "src.version",
    "truststore",
    "sounddevice",
    "soundfile",
    "pyaudiowpatch",
    "faster_whisper",
    "ctranslate2",
]

for pkg in ("faster_whisper", "ctranslate2", "sounddevice", "soundfile", "truststore"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Fast/smoke build mode allows packaging without bundling 4+ GB of optional torch/pyannote wheels
# when TESSITURA_LIGHT_BUNDLE=1 is set during local smoke validation; full release builds bundle all.
light_bundle = os.environ.get("TESSITURA_LIGHT_BUNDLE", "0") == "1"
excludes = [
    "pytest",
    "unittest",
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
]
if light_bundle:
    excludes.extend([
        "torch",
        "torchaudio",
        "torchvision",
        "pyannote",
        "pyannote.audio",
        "lightning",
        "pytorch_lightning",
        "onnxruntime",
        "scipy",
        "sklearn",
    ])

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Tessitura",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_PATH) if ICON_PATH.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Tessitura",
)

