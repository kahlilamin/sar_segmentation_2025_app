# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = []
hiddenimports += collect_submodules('rasterio')
hiddenimports += collect_submodules('tensorflow')

# Rasterio GDAL data files
rasterio_gdal_data = collect_data_files('rasterio', subdir='gdal_data')

# Rasterio pyproj data files
rasterio_proj_data = collect_data_files('rasterio', subdir='proj_data')

# Your model data folder
model_data = [('data/models', 'data/models')]

a = Analysis(
    ['gui_prediction_app.py'],
    pathex=[],
    binaries=[],
    datas=rasterio_gdal_data + rasterio_proj_data + model_data,
    hiddenimports=hiddenimports,
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
    name='veg_prediction_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='veg_prediction_app',
)
