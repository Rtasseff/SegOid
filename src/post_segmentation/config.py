"""User configuration for post-segmentation analysis.

Every filter is either a numeric threshold or None (= filter off).
Group-by columns must already exist in the input metrics CSV — this script
does not re-parse filenames. Enable "Parse filename into fields" in the
SegOid GUI when running inference to populate metadata columns.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PostSegConfig:
    metrics_csv: Path
    output_xlsx: Path

    # Grouping: list of existing column names in the metrics CSV.
    # Every column listed here must be present, or analyze.run() raises.
    group_by: List[str] = field(default_factory=lambda: ["image"])

    # Filters: numeric threshold or None.
    circularity_min: Optional[float] = None
    area_min_px: Optional[float] = None
    area_max_px: Optional[float] = None
    area_min_um2: Optional[float] = None
    area_max_um2: Optional[float] = None
    mad_max: Optional[float] = None  # filter on flag_robust_outlier_mad

    def __post_init__(self):
        self.metrics_csv = Path(self.metrics_csv)
        self.output_xlsx = Path(self.output_xlsx)
