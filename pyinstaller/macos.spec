# -*- mode: python ; coding: utf-8 -*-
import os
import sys

if len(sys.argv) and os.path.exists(sys.argv[-1]):
    spec_path = os.path.abspath(sys.argv[-1])
else:
    spec_path = os.getcwd()

project_root = os.path.abspath(os.path.join(os.path.dirname(spec_path), '..'))
pathex = [project_root]

analysis = Analysis(
    [os.path.join(project_root, 'gui.py')],
    pathex=pathex,
    binaries=[],
    datas=[(os.path.join(project_root, 'model', 'model.h5'), 'model')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    [],
    name='MinecraftAutoFisherBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
