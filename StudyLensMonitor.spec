# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['app_main.py'],
    pathex=[],
    binaries=[],
    datas=[('shape_predictor_68_face_landmarks.dat', '.')], # <-- We will modify this
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StudyLensMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Already set by --windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StudyLensMonitor',
)
app = BUNDLE(
    coll,
    name='StudyLensMonitor.app',
    icon=None, # Optional: set path to .icns file
    bundle_identifier='com.yourdomain.studylensmonitor', # <-- Set your unique ID
    info_plist={ # <-- Add this dictionary
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSMinimumSystemVersion': '11.0', # Example minimum OS
        'NSHumanReadableCopyright': 'Copyright ©️ 2024 Your Name. All rights reserved.', # Optional
        'CFBundlePackageType': 'APPL',
        # --- IMPORTANT ---
        'NSCameraUsageDescription': 'StudyLens needs access to your camera to monitor facial landmarks for focus, drowsiness, and emotion analysis during your study session.'
        # --- You can add other keys from the Info.plist XML here ---
        }
)