# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SegOid Windows executable.

Build with:
    pyinstaller --clean --noconfirm segoid_gui.spec

Prerequisites:
    1. Export ONNX model first:
       export_onnx --checkpoint models/production_v2.0/checkpoints/best_model.pth \
                   --output assets/segoid_model.onnx

    2. Install PyInstaller:
       pip install pyinstaller

Output:
    dist/SegOid.exe (~100-200MB)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect all submodules for packages that have complex import structures
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all('pandas')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all('skimage')

# Collect imagecodecs with all its binary extensions (critical for TIFF LZW support)
imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all('imagecodecs')
imagecodecs_dynlibs = collect_dynamic_libs('imagecodecs')

# Collect tifffile
tifffile_datas, tifffile_binaries, tifffile_hiddenimports = collect_all('tifffile')

# Collect PIL/Pillow
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')

# Get the project root directory
project_root = Path(SPECPATH)

# Data files to bundle
datas = []

# Bundle the ONNX model if it exists
onnx_model = project_root / 'assets' / 'segoid_model.onnx'
if onnx_model.exists():
    datas.append((str(onnx_model), 'assets'))
else:
    print(f"WARNING: ONNX model not found at {onnx_model}")
    print("         The executable will require a custom model to be selected.")

# Bundle icon if it exists (must be .ico format for Windows)
icon_file = project_root / 'assets' / 'segoid.ico'
if icon_file.exists():
    datas.append((str(icon_file), 'assets'))
else:
    print(f"NOTE: Icon not found at {icon_file}")
    print("      Windows icons must be .ico format (not .png)")
    print("      Convert with: convert segoid.png -define icon:auto-resize segoid.ico")

# Analysis
a = Analysis(
    [str(project_root / 'src' / 'gui' / 'app.py')],
    pathex=[str(project_root)],
    binaries=pandas_binaries + scipy_binaries + skimage_binaries + imagecodecs_binaries + imagecodecs_dynlibs + tifffile_binaries + pil_binaries,
    datas=datas + pandas_datas + scipy_datas + skimage_datas + imagecodecs_datas + tifffile_datas + pil_datas,
    hiddenimports=[
        # ONNX Runtime
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',

        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.TiffImagePlugin',
        'tifffile',
        'tifffile.tifffile',
        'imagecodecs',
        'imagecodecs._imagecodecs',
        'skimage',
        'skimage.measure',
        'skimage.measure._regionprops',
        'skimage.morphology',
        'skimage.morphology._skeletonize',

        # Numerical
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'scipy',
        'scipy.ndimage',
        'scipy.ndimage._interpolation',
        'scipy.optimize',
        'scipy.optimize._lsq',

        # Data handling - pandas and its dependencies
        'pandas',
        'pandas.core',
        'pandas.core.frame',
        'pandas.core.series',
        'pandas._libs',
        'pandas._libs.lib',
        'pandas._libs.hashtable',
        'pandas._libs.tslibs',
        'pandas.io',
        'pandas.io.formats',
        'pandas.io.parsers',

        # Video export
        'cv2',

        # GUI (should be included but explicit)
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.scrolledtext',
        'tkinter.messagebox',

        # Our modules
        'src',
        'src.gui',
        'src.gui.app',
        'src.gui.jobs',
        'src.gui.widgets',
        'src.gui.logging_handler',
        'src.data',
        'src.data.manifest',
        'src.data.filename_schema',
        'src.inference',
        'src.inference.predict',
        'src.analysis',
        'src.analysis.quantify',
        'src.visualization',
        'src.visualization.review',
    ] + pandas_hiddenimports + scipy_hiddenimports + skimage_hiddenimports + imagecodecs_hiddenimports + tifffile_hiddenimports + pil_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Explicitly exclude PyTorch and training dependencies
        # to keep bundle size reasonable
        'torch',
        'torchvision',
        'tensorflow',
        'keras',
        'tensorboard',
        'albumentations',
        'segmentation_models_pytorch',

        # Development tools
        'pytest',
        'black',
        'flake8',

        # Jupyter/IPython
        'IPython',
        'jupyter',
        'notebook',

        # Documentation
        'sphinx',
        'docutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create PYZ archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SegOid.exe' if sys.platform == 'win32' else 'SegOid',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No terminal window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_file) if icon_file.exists() else None,
)
