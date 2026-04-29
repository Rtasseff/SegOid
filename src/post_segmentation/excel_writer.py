"""Multi-tab Excel writer for post-segmentation analysis output."""

import re
from pathlib import Path
from typing import Dict

import pandas as pd

_INVALID_SHEET_CHARS = re.compile(r"[\\/?*\[\]:]")
_MAX_SHEET_NAME_LEN = 31


def _safe_sheet_name(name: str, used: set) -> str:
    r"""Return an Excel-legal, unique sheet name.

    Excel forbids ``\ / ? * [ ] :`` and limits names to 31 characters.
    """
    cleaned = _INVALID_SHEET_CHARS.sub("_", name)[:_MAX_SHEET_NAME_LEN] or "sheet"
    candidate = cleaned
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = (cleaned[: _MAX_SHEET_NAME_LEN - len(tail)] + tail)
        suffix += 1
    used.add(candidate)
    return candidate


def write_multitab_excel(output_path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """Write each (sheet_name → DataFrame) pair as a tab in one .xlsx file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    used: set = set()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for raw_name, df in sheets.items():
            sheet = _safe_sheet_name(str(raw_name), used)
            df.to_excel(writer, sheet_name=sheet, index=False)
