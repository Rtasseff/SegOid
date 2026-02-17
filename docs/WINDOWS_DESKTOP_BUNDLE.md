# Windows Desktop Bundle

This document describes how to build a standalone Windows executable for SegOid that can be distributed to users without requiring Python installation.

**Bundle size:** ~150-200MB (using ONNX Runtime instead of PyTorch)

---

## Overview

The Windows executable provides a GUI for running spheroid segmentation inference. It:

- Loads ONNX models (bundled default or custom)
- Processes folders of TIFF images
- Outputs binary masks and morphology metrics
- Exports prediction review videos

**Key Design Decisions:**

- **ONNX Runtime** instead of PyTorch (reduces bundle from 2-4GB to ~150MB)
- **Tkinter** for GUI (included with Python, no extra dependencies)
- **PIL/Pillow** for TIFF reading (better Windows compatibility than tifffile+imagecodecs alone)
- **PyInstaller** for bundling

---

## Build Process

Building the Windows executable requires two environments:

1. **WSL/Linux** - For exporting the ONNX model (requires PyTorch)
2. **Windows** - For running PyInstaller (creates Windows-native executable)

### Step 1: Export ONNX Model (WSL/Linux)

```bash
# In WSL or Linux terminal
cd ~/projects/SegOid
source .venv/bin/activate

# Export PyTorch model to ONNX format
export_onnx \
    --checkpoint runs/train_20260216_173233/checkpoints/best_model.pth \
    --output assets/segoid_model.onnx
```

This creates `assets/segoid_model.onnx` (~55MB) which will be bundled into the executable.

### Step 2: Create Icon (Optional)

Convert a PNG icon to Windows ICO format:

```bash
# Using ImageMagick (install with: sudo apt install imagemagick)
convert assets/SegOid_icon.png -define icon:auto-resize=256,128,64,48,32,16 assets/segoid.ico

# Or use an online converter like https://convertico.com/
```

### Step 3: Install Dependencies in Windows Python

Open **Windows PowerShell** (not WSL) and install the required packages:

```powershell
pip install pandas numpy scipy scikit-image opencv-python tifffile imagecodecs onnxruntime pillow pyinstaller
```

### Step 4: Build Executable (Windows PowerShell)

```powershell
# Navigate to the project (WSL files accessible via \\wsl$\)
cd \\wsl$\Ubuntu\home\<username>\projects\SegOid

# Build the executable
python -m PyInstaller --clean --noconfirm segoid_gui.spec
```

**Output:** `dist/SegOid.exe` (~150-200MB)

### Step 5: Test

1. Copy `dist/SegOid.exe` to a Windows machine (can be a different machine)
2. Double-click to run
3. Select input folder with TIFF images
4. Select output folder
5. Click "Run Inference"

---

## Module Structure

```
src/
├── gui/
│   ├── __init__.py
│   ├── app.py                # Main Tkinter application
│   ├── widgets.py            # File/folder picker widgets
│   ├── jobs.py               # Background job orchestration
│   └── logging_handler.py    # Thread-safe logging to GUI
├── data/
│   ├── manifest.py           # Build manifest from folder
│   └── filename_schema.py    # Parse filenames into metadata
├── inference/
│   ├── predict.py            # ONNX + PyTorch inference
│   └── onnx_export.py        # PyTorch to ONNX conversion
```

---

## CLI Commands

Two new CLI commands were added:

### `export_onnx` - Convert PyTorch to ONNX

```bash
export_onnx \
    --checkpoint <model.pth> \
    --output <model.onnx> \
    --input-size 256 \
    --opset-version 17
```

### `segoid_gui` - Launch GUI (development)

```bash
segoid_gui
```

---

## PyInstaller Spec File

The `segoid_gui.spec` file handles:

- **Bundled ONNX model** - Included in executable if present in `assets/`
- **Icon** - Windows application icon (`assets/segoid.ico`)
- **Hidden imports** - Complex packages like pandas, scipy, imagecodecs
- **Binary collection** - Native extensions for imagecodecs (LZW, etc.)
- **Exclusions** - PyTorch, TensorFlow (not needed for ONNX inference)

Key features:
```python
# Collect all submodules for complex packages
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all('pandas')
imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all('imagecodecs')
# ... etc
```

---

## Troubleshooting

### "This app cannot run on your PC"

PyInstaller creates executables for the OS it runs on. You must run PyInstaller on **Windows** to create a Windows executable.

### ModuleNotFoundError (pandas, scipy, etc.)

Install the missing package in your **Windows Python** environment:
```powershell
pip install <package_name>
```

### "could not import 'lzw_decode' from imagecodecs"

This means imagecodecs binary extensions weren't bundled. The spec file now uses `collect_all('imagecodecs')` to include all codecs. Rebuild with the latest spec file.

### Icon not showing

- Icon must be `.ico` format (not `.png`)
- Must be named `assets/segoid.ico`
- Rebuild after adding the icon

---

## Future Enhancements

- GPU inference (ONNX Runtime CUDA/DirectML provider)
- Auto-update mechanism
- Progress bar with percentage
- Settings persistence (remember folders)

---

## Files Added/Modified

### New Files
- `src/gui/__init__.py`
- `src/gui/app.py`
- `src/gui/widgets.py`
- `src/gui/jobs.py`
- `src/gui/logging_handler.py`
- `src/data/manifest.py`
- `src/data/filename_schema.py`
- `src/inference/onnx_export.py`
- `segoid_gui.spec`
- `assets/segoid_model.onnx` (generated)
- `assets/segoid.ico` (optional)

### Modified Files
- `src/inference/predict.py` - Added ONNX inference path, PIL fallback for TIFF reading
- `src/cli.py` - Added `export_onnx` and `run_gui` commands
- `pyproject.toml` - Added onnx, onnxruntime dependencies and new entry points
