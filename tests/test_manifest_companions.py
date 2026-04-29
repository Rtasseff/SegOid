"""Tests for fluorescence companion image detection in src.data.manifest."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from src.data.manifest import build_manifest, classify_companions


def _write_blank_tiff(path: Path):
    tifffile.imwrite(path, np.zeros((10, 10), dtype=np.uint8))


@pytest.fixture
def companion_folder(tmp_path):
    """Folder with 2 base images, A1 has both green+red, A2 has green only."""
    for name in ["well_A1.tif", "well_A2.tif",
                 "well_A1_green.tif", "well_A1_red.tif",
                 "well_A2_green.tif"]:
        _write_blank_tiff(tmp_path / name)
    return tmp_path


# ---------- classify_companions ----------


def test_classify_no_markers_returns_all_files(companion_folder):
    files = sorted(companion_folder.glob("*.tif"))
    base, comp = classify_companions(files, [])
    assert len(base) == len(files)
    assert comp == {}


def test_classify_single_marker(companion_folder):
    files = sorted(companion_folder.glob("*.tif"))
    base, comp = classify_companions(files, ["green"])
    base_names = sorted(b.stem for b in base)
    # well_A1, well_A2 are bases; well_A1_red is also a base (red not declared).
    # well_A1_green and well_A2_green are companions.
    assert "well_A1" in base_names
    assert "well_A2" in base_names
    assert "well_A1_red" in base_names
    assert "well_A1_green" not in base_names
    assert "well_A2_green" not in base_names
    assert "well_A1" in comp["green"]
    assert "well_A2" in comp["green"]


def test_classify_two_markers(companion_folder):
    files = sorted(companion_folder.glob("*.tif"))
    base, comp = classify_companions(files, ["green", "red"])
    base_names = sorted(b.stem for b in base)
    assert base_names == ["well_A1", "well_A2"]
    assert "well_A1" in comp["green"]
    assert "well_A2" in comp["green"]
    assert "well_A1" in comp["red"]
    assert "well_A2" not in comp["red"]  # missing red companion


def test_classify_marker_case_insensitive(tmp_path):
    """Markers should match case-insensitively on the suffix."""
    _write_blank_tiff(tmp_path / "well_A1.tif")
    _write_blank_tiff(tmp_path / "well_A1_GREEN.tif")
    base, comp = classify_companions(
        sorted(tmp_path.glob("*.tif")), ["green"]
    )
    assert [b.stem for b in base] == ["well_A1"]
    assert "well_A1" in comp["green"]


# ---------- build_manifest ----------


def test_build_manifest_no_markers_unchanged_columns(companion_folder):
    """Without markers, output must be the legacy column set."""
    df = build_manifest(companion_folder)
    assert list(df.columns) == ["basename", "image_path", "mask_path"]
    # All 5 files become base rows
    assert len(df) == 5


def test_build_manifest_with_markers_adds_companion_columns(companion_folder):
    df = build_manifest(companion_folder, markers=["green", "red"])
    assert list(df.columns) == [
        "basename", "image_path", "mask_path", "companion_green", "companion_red",
    ]
    assert len(df) == 2
    a1 = df[df.basename == "well_A1"].iloc[0]
    a2 = df[df.basename == "well_A2"].iloc[0]
    assert a1.companion_green and a1.companion_red
    assert a2.companion_green and not a2.companion_red


def test_build_manifest_missing_companion_warning(companion_folder, caplog):
    """When markers are declared and some bases lack a companion, log a warning."""
    caplog.set_level(logging.WARNING)
    build_manifest(companion_folder, markers=["green", "red"])
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    # Only red is missing for one base image.
    assert any("Marker 'red'" in w and "1 / 2" in w for w in warnings)
    # Green is present for all bases — no warning expected.
    assert not any("Marker 'green'" in w for w in warnings)


def test_build_manifest_empty_folder(tmp_path):
    df = build_manifest(tmp_path, markers=["green"])
    # Empty folder + markers → empty DataFrame with companion column
    assert "companion_green" in df.columns
    assert len(df) == 0
