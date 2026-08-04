# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'), ('icon.png', '.'), ('.venv\\Lib\\site-packages\\faster_whisper\\assets\\silero_vad_v6.onnx', 'faster_whisper\\assets'), ('.venv\\Lib\\site-packages\\_sounddevice_data\\portaudio-binaries', '_sounddevice_data\\portaudio-binaries')],
    hiddenimports=['faster_whisper', 'sounddevice', 'soundcard', 'qasync', 'aiosqlite', 'boto3', 'pydantic_settings', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Notepadder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Notepadder',
)
