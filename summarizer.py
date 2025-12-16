#!/usr/bin/env python3
"""Summarizer script for HistoPLUS WSI output.

Scans a folder (like wsi_output) and creates a CSV summary with:
- Total area (mm²)
- Total cells
- Total tiles
- % Tumor Area
- % Tumor Cells
- Time taken (if available from logs)
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from histoplus.helpers.data.slide_segmentation_data import SlideSegmentationData
from calculate_cancer_area import calculate_cell_areas, calculate_cancer_ratio


def extract_time_from_log(log_file: Path) -> Optional[float]:
    """Extract processing time from log file.

    Parameters
    ----------
    log_file : Path
        Path to the log file.

    Returns
    -------
    Optional[float]
        Time in seconds, or None if not found.
    """
    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Look for pattern: "Finished processing of ... in X.X seconds"
        pattern = r"Finished processing of .+? in ([\d.]+) seconds"
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))

        # Alternative pattern: "--- Finished processing of ... in X.X seconds"
        pattern = r"--- Finished processing of .+? in ([\d.]+) seconds"
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))

    except Exception as e:
        logger.warning(f"Error reading log file {log_file}: {e}")

    return None


def find_matching_log(slide_name: str, log_dir: Path) -> Optional[Path]:
    """Find the log file that matches a slide name.

    Parameters
    ----------
    slide_name : str
        Name of the slide (without extension).
    log_dir : Path
        Directory containing log files.

    Returns
    -------
    Optional[Path]
        Path to matching log file, or None if not found.
    """
    # Try to find log files that mention this slide
    for log_file in log_dir.glob("*.log"):
        try:
            with open(log_file, "r") as f:
                content = f.read()
                if slide_name in content:
                    return log_file
        except Exception:
            continue

    return None


def process_cell_masks_file(
    cell_masks_path: Path, log_dir: Optional[Path] = None
) -> dict:
    """Process a cell_masks.json file and extract statistics.

    Uses calculate_cancer_area.py functions for accurate tumor area calculation.

    Parameters
    ----------
    cell_masks_path : Path
        Path to the cell_masks.json file.
    log_dir : Optional[Path]
        Directory containing log files for time extraction.

    Returns
    -------
    dict
        Dictionary with statistics.
    """
    try:
        # Load using SlideSegmentationData for proper parsing
        slide_data = SlideSegmentationData.load(str(cell_masks_path))
    except Exception as e:
        logger.error(f"Error reading {cell_masks_path}: {e}")
        return {}

    # Extract metadata
    model_name = slide_data.model_name
    inference_mpp = slide_data.inference_mpp
    total_tiles = len(slide_data.cell_masks)

    # Use calculate_cancer_area.py functions for accurate tumor area calculations
    # This uses the same logic as calculate_cancer_area.py:
    # - Detects cancer cells by checking if "cancer" is in the cell type name (case-insensitive)
    # - Calculates areas using the shoelace formula for polygon areas
    cell_type_areas, total_area_pixels, total_cells = calculate_cell_areas(
        slide_data, mpp=inference_mpp
    )
    cancer_ratio = calculate_cancer_ratio(cell_type_areas, total_area_pixels)

    # Extract tumor/cancer statistics
    # Note: "tumor_cells" refers to cancer cells detected by calculate_cancer_ratio
    tumor_cells = cancer_ratio["cancer_count"]
    tumor_area_pixels = cancer_ratio["cancer_area_pixels"]
    pct_tumor_cells = (tumor_cells / total_cells * 100) if total_cells > 0 else 0.0
    pct_tumor_area = cancer_ratio["cancer_percentage"]

    # Calculate total tissue area in mm² from tile dimensions
    # Sum up all tile areas
    total_tile_area_pixels = 0.0
    for tile in slide_data.cell_masks:
        total_tile_area_pixels += tile.width * tile.height

    # Calculate area in mm²
    total_area_mm2 = None
    if inference_mpp is not None:
        # Convert pixels to mm²
        # MPP = microns per pixel
        # 1 mm = 1000 microns
        # Area in mm² = (pixels * MPP / 1000)²
        total_area_mm2 = (total_tile_area_pixels * (inference_mpp / 1000.0) ** 2)

    # Extract slide name
    slide_name = cell_masks_path.parent.name
    if slide_name.endswith(".svs"):
        slide_name = slide_name[:-4]

    # Try to extract time from log
    time_taken = None
    if log_dir and log_dir.exists():
        log_file = find_matching_log(slide_name, log_dir)
        if log_file:
            time_taken = extract_time_from_log(log_file)

    return {
        "slide_name": slide_name,
        "model_name": model_name,
        "mpp": inference_mpp,
        "total_area_mm2": total_area_mm2,
        "total_cells": total_cells,
        "total_tiles": total_tiles,
        "tumor_cells": tumor_cells,
        "pct_tumor_cells": round(pct_tumor_cells, 2),
        "pct_tumor_area": round(pct_tumor_area, 2),
        "time_taken_seconds": time_taken,
    }


def scan_output_folder(output_folder: Path, log_dir: Optional[Path] = None) -> list[dict]:
    """Scan output folder and process all cell_masks.json files.

    Parameters
    ----------
    output_folder : Path
        Path to the output folder (e.g., wsi_output).
    log_dir : Optional[Path]
        Directory containing log files. If None, searches in output_folder.

    Returns
    -------
    list[dict]
        List of statistics dictionaries.
    """
    if log_dir is None:
        log_dir = output_folder

    results = []

    # Find all cell_masks.json files
    cell_masks_files = list(output_folder.rglob("cell_masks.json"))

    if not cell_masks_files:
        logger.warning(f"No cell_masks.json files found in {output_folder}")
        return results

    logger.info(f"Found {len(cell_masks_files)} cell_masks.json file(s)")

    for cell_masks_file in cell_masks_files:
        logger.info(f"Processing {cell_masks_file.parent.name}...")
        stats = process_cell_masks_file(cell_masks_file, log_dir)
        if stats:
            results.append(stats)

    return results


def write_csv(results: list[dict], output_path: Path) -> None:
    """Write results to CSV file.

    Parameters
    ----------
    results : list[dict]
        List of statistics dictionaries.
    output_path : Path
        Path to output CSV file.
    """
    if not results:
        logger.warning("No results to write")
        return

    # Define column order
    columns = [
        "slide_name",
        "model_name",
        "mpp",
        "total_area_mm2",
        "total_cells",
        "total_tiles",
        "tumor_cells",
        "pct_tumor_cells",
        "pct_tumor_area",
        "time_taken_seconds",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for result in results:
            # Format values for CSV
            row = {}
            for col in columns:
                value = result.get(col)
                if value is None:
                    row[col] = ""
                elif isinstance(value, float):
                    # Format floats appropriately
                    if col in ["pct_tumor_cells", "pct_tumor_area"]:
                        row[col] = f"{value:.2f}"
                    elif col == "total_area_mm2":
                        row[col] = f"{value:.4f}" if value is not None else ""
                    else:
                        row[col] = f"{value:.2f}"
                else:
                    row[col] = str(value)
            writer.writerow(row)

    logger.info(f"Wrote {len(results)} rows to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize HistoPLUS WSI output into CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize wsi_output folder
  python summarizer.py --input wsi_output --output summary.csv
  
  # Specify log directory separately
  python summarizer.py --input wsi_output --output summary.csv --log_dir wsi_output
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input folder containing WSI output (e.g., wsi_output)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory containing log files (default: same as input folder)",
    )

    args = parser.parse_args()

    input_folder = Path(args.input)
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return

    log_dir = Path(args.log_dir) if args.log_dir else None
    output_path = Path(args.output)

    # Scan and process
    logger.info(f"Scanning {input_folder}...")
    results = scan_output_folder(input_folder, log_dir)

    if not results:
        logger.error("No results found")
        return

    # Write CSV
    write_csv(results, output_path)

    # Print summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total slides processed: {len(results)}")
    total_cells = sum(r.get("total_cells", 0) for r in results)
    total_tiles = sum(r.get("total_tiles", 0) for r in results)
    logger.info(f"  Total cells: {total_cells:,}")
    logger.info(f"  Total tiles: {total_tiles:,}")


if __name__ == "__main__":
    main()

