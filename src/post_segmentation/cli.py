"""CLI wrapper for post-segmentation analysis (registered as `post_seg_analyze`)."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.post_segmentation.analyze import run
from src.post_segmentation.config import PostSegConfig


def _parse_optional_float(value: str) -> Optional[float]:
    """Parse '' or 'none'/'null'/'na' as None, otherwise float."""
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() in {"none", "null", "na", "nan"}:
        return None
    return float(s)


def _parse_columns(value: str) -> List[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="post_seg_analyze",
        description=(
            "Post-segmentation analysis on a SegOid metrics CSV. Filters, groups, "
            "and writes a multi-tab Excel. Group-by columns must already exist in "
            "the CSV — enable 'Parse filename into fields' in the SegOid GUI to "
            "populate metadata columns."
        ),
    )
    parser.add_argument(
        "--metrics", required=True, type=Path, help="Input SegOid metrics CSV."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output .xlsx path."
    )
    parser.add_argument(
        "--group-by",
        type=_parse_columns,
        default=["image"],
        help="Comma-separated existing column names to group by (default: image).",
    )
    parser.add_argument(
        "--circularity-min", type=_parse_optional_float, default=None,
        help="Minimum circularity (or 'none' to disable).",
    )
    parser.add_argument(
        "--area-min-px", type=_parse_optional_float, default=None,
        help="Minimum area in pixels (or 'none' to disable).",
    )
    parser.add_argument(
        "--area-max-px", type=_parse_optional_float, default=None,
        help="Maximum area in pixels (or 'none' to disable).",
    )
    parser.add_argument(
        "--area-min-um2", type=_parse_optional_float, default=None,
        help="Minimum area in µm² (or 'none' to disable).",
    )
    parser.add_argument(
        "--area-max-um2", type=_parse_optional_float, default=None,
        help="Maximum area in µm² (or 'none' to disable).",
    )
    parser.add_argument(
        "--mad-max", type=_parse_optional_float, default=None,
        help="Maximum |area - median| / MAD per group (or 'none' to disable).",
    )

    args = parser.parse_args(argv)

    config = PostSegConfig(
        metrics_csv=args.metrics,
        output_xlsx=args.output,
        group_by=args.group_by,
        circularity_min=args.circularity_min,
        area_min_px=args.area_min_px,
        area_max_px=args.area_max_px,
        area_min_um2=args.area_min_um2,
        area_max_um2=args.area_max_um2,
        mad_max=args.mad_max,
    )

    output = run(config)
    print(f"Post-segmentation analysis written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
