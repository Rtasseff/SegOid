"""
Filename schema parsing for extracting metadata from image filenames.

This module provides functionality to split filenames into structured
metadata fields based on delimiter-separated naming conventions.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilenameSchema:
    """
    Schema definition for parsing filenames.

    Attributes:
        delimiter: Character used to separate fields (default: "_")
        field_names: List of names for each field position
        expected_fields: Expected number of fields (inferred from field_names if not set)
    """

    delimiter: str = "_"
    field_names: List[str] = None
    expected_fields: Optional[int] = None

    def __post_init__(self):
        if self.field_names is None:
            self.field_names = []
        if self.expected_fields is None:
            self.expected_fields = len(self.field_names)


def detect_schema(
    sample_filenames: List[str],
    delimiter: str = "_",
) -> Tuple[int, List[str]]:
    """
    Detect the filename schema from sample filenames.

    Analyzes sample filenames to determine the number of fields
    when split by the delimiter.

    Args:
        sample_filenames: List of sample filename basenames (without extension)
        delimiter: Character used to separate fields

    Returns:
        Tuple of (field_count, sample_chunks) where sample_chunks
        are the fields from the first filename
    """
    if not sample_filenames:
        return 0, []

    # Analyze first filename
    first_name = sample_filenames[0]
    chunks = first_name.split(delimiter)
    field_count = len(chunks)

    # Verify consistency across samples
    inconsistent = []
    for name in sample_filenames[1:10]:  # Check first 10
        name_chunks = name.split(delimiter)
        if len(name_chunks) != field_count:
            inconsistent.append((name, len(name_chunks)))

    if inconsistent:
        logger.warning(
            f"Inconsistent field counts detected. Expected {field_count} fields, "
            f"but found: {inconsistent}"
        )

    return field_count, chunks


def parse_filename(
    basename: str,
    schema: FilenameSchema,
) -> Dict[str, str]:
    """
    Parse a filename according to the schema.

    Args:
        basename: Filename basename (without extension)
        schema: FilenameSchema defining how to parse

    Returns:
        Dictionary mapping field names to values
    """
    chunks = basename.split(schema.delimiter)

    result = {}

    # Map chunks to field names
    for i, chunk in enumerate(chunks):
        if i < len(schema.field_names):
            field_name = schema.field_names[i]
        else:
            field_name = f"field_{i}"
        result[field_name] = chunk

    # Check for field count mismatch
    if len(chunks) != schema.expected_fields:
        logger.debug(
            f"Field count mismatch for '{basename}': "
            f"expected {schema.expected_fields}, got {len(chunks)}"
        )

    return result


def parse_filenames_batch(
    basenames: List[str],
    schema: FilenameSchema,
) -> pd.DataFrame:
    """
    Parse multiple filenames according to the schema.

    Args:
        basenames: List of filename basenames
        schema: FilenameSchema defining how to parse

    Returns:
        DataFrame with columns for each field plus 'basename'
    """
    records = []

    for basename in basenames:
        record = parse_filename(basename, schema)
        record["basename"] = basename
        records.append(record)

    df = pd.DataFrame(records)

    # Reorder columns to put basename first
    cols = ["basename"] + [c for c in df.columns if c != "basename"]
    df = df[cols]

    return df


def validate_schema(
    filenames: List[str],
    schema: FilenameSchema,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate that filenames conform to the schema.

    Args:
        filenames: List of filename basenames to validate
        schema: FilenameSchema to validate against
        strict: If True, all filenames must match; if False, warns on mismatches

    Returns:
        Tuple of (is_valid, list_of_invalid_filenames)
    """
    invalid = []

    for name in filenames:
        chunks = name.split(schema.delimiter)
        if len(chunks) != schema.expected_fields:
            invalid.append(name)

    is_valid = len(invalid) == 0

    if invalid:
        msg = f"{len(invalid)} filenames do not match schema (expected {schema.expected_fields} fields)"
        if strict:
            logger.error(msg)
        else:
            logger.warning(msg)

    return is_valid, invalid


def create_schema_from_labels(
    labels: List[str],
    delimiter: str = "_",
) -> FilenameSchema:
    """
    Create a FilenameSchema from user-provided labels.

    Args:
        labels: List of field names in order
        delimiter: Delimiter character

    Returns:
        FilenameSchema instance
    """
    # Clean up labels
    cleaned_labels = []
    for label in labels:
        # Remove special characters, convert to lowercase
        clean = re.sub(r"[^a-zA-Z0-9_]", "", label.strip().lower())
        if not clean:
            clean = f"field_{len(cleaned_labels)}"
        cleaned_labels.append(clean)

    return FilenameSchema(
        delimiter=delimiter,
        field_names=cleaned_labels,
        expected_fields=len(cleaned_labels),
    )


def add_metadata_to_dataframe(
    df: pd.DataFrame,
    schema: FilenameSchema,
    basename_column: str = "basename",
) -> pd.DataFrame:
    """
    Add parsed metadata columns to an existing DataFrame.

    Args:
        df: DataFrame with a basename column
        schema: FilenameSchema for parsing
        basename_column: Name of column containing basenames

    Returns:
        DataFrame with additional metadata columns
    """
    if basename_column not in df.columns:
        raise ValueError(f"Column '{basename_column}' not found in DataFrame")

    # Parse all basenames
    metadata_df = parse_filenames_batch(
        df[basename_column].tolist(),
        schema,
    )

    # Drop the duplicate basename column from metadata
    metadata_df = metadata_df.drop(columns=["basename"])

    # Concatenate
    result = pd.concat([df.reset_index(drop=True), metadata_df.reset_index(drop=True)], axis=1)

    return result
