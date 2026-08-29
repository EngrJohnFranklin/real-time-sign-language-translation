# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the Windows FSL Translator distribution."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH)

RESOURCE_DIRECTORIES = (
    "assets",
    "config",
    "data",
    "model",
    "model-tl",
)
RESOURCE_FILES = (
    "LICENSE",
    "LICENSE-THIRD-PARTY.md",
)

_datas = [
    (str(PROJECT_ROOT / resource), resource)
    for resource in RESOURCE_DIRECTORIES
]
_datas.extend(
    (str(PROJECT_ROOT / resource), resource)
    for resource in RESOURCE_FILES
)
_binaries = []
_hiddenimports = []

for package in (
    "customtkinter",
    "mediapipe",
    "pyaudio",
    "pyttsx3",
    "vosk",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    _datas.extend(package_datas)
    _binaries.extend(package_binaries)
    _hiddenimports.extend(package_hiddenimports)

_datas.extend(collect_data_files("xgboost"))
_binaries.extend(collect_dynamic_libs("xgboost"))
_hiddenimports.extend(
    collect_submodules(
        "xgboost",
        filter=lambda name: not name.startswith("xgboost.testing"),
    )
)
_hiddenimports.extend(
    [
        "sklearn",
        "sklearn.preprocessing",
        "sklearn.preprocessing._label",
    ]
)

analysis = Analysis(
    [str(PROJECT_ROOT / "src" / "main.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SignLanguageTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SignLanguageTranslator",
)
