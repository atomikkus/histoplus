#!/usr/bin/env python3
"""Calculate the area covered by cancer cells compared to all other tissue."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from histoplus.helpers.data.slide_segmentation_data import SlideSegmentationData


def calculate_polygon_area(coordinates: list) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Args:
        coordinates: List of [x, y] coordinates defining the polygon
        
    Returns:
        Area of the polygon in square pixels
    """
    if not coordinates or len(coordinates) < 3:
        return 0.0
    
    coords = np.array(coordinates, dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def calculate_cell_areas(slide_data: SlideSegmentationData, mpp: float = None) -> dict:
    """
    Calculate the total area covered by each cell type.
    
    Args:
        slide_data: SlideSegmentationData containing masks
        mpp: Microns per pixel for converting to physical units (optional)
        
    Returns:
        Dictionary with area statistics for each cell type
    """
    cell_type_areas = {}
    total_area = 0.0
    total_cells = 0
    
    print("Calculating areas for all cells...")
    
    for tile_idx, tile in enumerate(slide_data.cell_masks):
        if (tile_idx + 1) % 100 == 0:
            print(f"  Processed {tile_idx + 1}/{len(slide_data.cell_masks)} tiles...")
        
        for mask in tile.masks:
            cell_type = mask.cell_type
            
            # Calculate area of this mask
            area = calculate_polygon_area(mask.coordinates)
            
            if cell_type not in cell_type_areas:
                cell_type_areas[cell_type] = {
                    "area_pixels": 0.0,
                    "count": 0,
                }
            
            cell_type_areas[cell_type]["area_pixels"] += area
            cell_type_areas[cell_type]["count"] += 1
            total_area += area
            total_cells += 1
    
    print(f"  Completed processing {len(slide_data.cell_masks)} tiles")
    print(f"  Total cells: {total_cells:,}")
    
    # Calculate percentages and convert to microns if mpp provided
    for cell_type, data in cell_type_areas.items():
        data["percentage"] = (data["area_pixels"] / total_area * 100) if total_area > 0 else 0.0
        data["avg_area_pixels"] = data["area_pixels"] / data["count"] if data["count"] > 0 else 0.0
        
        if mpp is not None:
            # Convert from pixels^2 to microns^2
            data["area_um2"] = data["area_pixels"] * (mpp ** 2)
            data["avg_area_um2"] = data["avg_area_pixels"] * (mpp ** 2)
    
    # Sort by area (descending)
    cell_type_areas = dict(sorted(cell_type_areas.items(), 
                                  key=lambda x: x[1]["area_pixels"], 
                                  reverse=True))
    
    return cell_type_areas, total_area, total_cells


def calculate_cancer_ratio(cell_type_areas: dict, total_area: float) -> dict:
    """
    Calculate the ratio of cancer cell area to total tissue area.
    
    Args:
        cell_type_areas: Dictionary with area statistics per cell type
        total_area: Total area of all cells
        
    Returns:
        Dictionary with cancer vs non-cancer statistics
    """
    cancer_area = 0.0
    cancer_count = 0
    non_cancer_area = 0.0
    non_cancer_count = 0
    
    # Sum up cancer and non-cancer areas
    for cell_type, data in cell_type_areas.items():
        if "cancer" in cell_type.lower():
            cancer_area += data["area_pixels"]
            cancer_count += data["count"]
        else:
            non_cancer_area += data["area_pixels"]
            non_cancer_count += data["count"]
    
    cancer_percentage = (cancer_area / total_area * 100) if total_area > 0 else 0.0
    non_cancer_percentage = (non_cancer_area / total_area * 100) if total_area > 0 else 0.0
    
    # Calculate ratio
    cancer_to_total_ratio = cancer_area / total_area if total_area > 0 else 0.0
    cancer_to_non_cancer_ratio = cancer_area / non_cancer_area if non_cancer_area > 0 else float('inf')
    
    return {
        "cancer_area_pixels": cancer_area,
        "cancer_count": cancer_count,
        "cancer_percentage": cancer_percentage,
        "non_cancer_area_pixels": non_cancer_area,
        "non_cancer_count": non_cancer_count,
        "non_cancer_percentage": non_cancer_percentage,
        "total_area_pixels": total_area,
        "cancer_to_total_ratio": cancer_to_total_ratio,
        "cancer_to_non_cancer_ratio": cancer_to_non_cancer_ratio,
    }


def print_results(cell_type_areas: dict, cancer_ratio: dict, mpp: float = None):
    """Print the results in a formatted table."""
    print("\n" + "=" * 80)
    print("CELL AREA ANALYSIS")
    print("=" * 80)
    
    # Print cancer vs non-cancer summary
    print("\nCANCER vs NON-CANCER SUMMARY:")
    print("-" * 80)
    print(f"Cancer cells:     {cancer_ratio['cancer_count']:>10,} cells  |  "
          f"{cancer_ratio['cancer_area_pixels']:>15,.0f} px²  |  "
          f"{cancer_ratio['cancer_percentage']:>6.2f}%")
    print(f"Non-cancer cells: {cancer_ratio['non_cancer_count']:>10,} cells  |  "
          f"{cancer_ratio['non_cancer_area_pixels']:>15,.0f} px²  |  "
          f"{cancer_ratio['non_cancer_percentage']:>6.2f}%")
    print(f"Total:            {cancer_ratio['cancer_count'] + cancer_ratio['non_cancer_count']:>10,} cells  |  "
          f"{cancer_ratio['total_area_pixels']:>15,.0f} px²  |  100.00%")
    print("-" * 80)
    print(f"Cancer to Total Ratio:       {cancer_ratio['cancer_to_total_ratio']:.4f}")
    if cancer_ratio['cancer_to_non_cancer_ratio'] != float('inf'):
        print(f"Cancer to Non-Cancer Ratio:  {cancer_ratio['cancer_to_non_cancer_ratio']:.4f}")
    else:
        print(f"Cancer to Non-Cancer Ratio:  ∞ (no non-cancer cells)")
    
    # Print detailed breakdown by cell type
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN BY CELL TYPE:")
    print("=" * 80)
    
    if mpp is not None:
        print(f"{'Cell Type':<25} {'Count':>10}  {'Area (px²)':>15}  {'Area (μm²)':>15}  {'% of Total':>10}  {'Avg Area (μm²)':>15}")
    else:
        print(f"{'Cell Type':<25} {'Count':>10}  {'Area (px²)':>15}  {'% of Total':>10}  {'Avg Area (px²)':>15}")
    print("-" * 80)
    
    for cell_type, data in cell_type_areas.items():
        if mpp is not None:
            print(f"{cell_type:<25} {data['count']:>10,}  {data['area_pixels']:>15,.0f}  "
                  f"{data['area_um2']:>15,.0f}  {data['percentage']:>9.2f}%  {data['avg_area_um2']:>15,.1f}")
        else:
            print(f"{cell_type:<25} {data['count']:>10,}  {data['area_pixels']:>15,.0f}  "
                  f"{data['percentage']:>9.2f}%  {data['avg_area_pixels']:>15,.1f}")
    
    print("=" * 80 + "\n")
    
    if mpp is not None:
        print(f"Note: Assuming {mpp} microns per pixel (mpp)")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Calculate the area covered by cancer cells compared to all other tissue"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the cell_masks.json file",
    )
    parser.add_argument(
        "-m", "--mpp",
        type=float,
        default=None,
        help="Microns per pixel (mpp) for converting to physical units",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path for saving results (optional)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load data
        print(f"Loading masks: {json_path}")
        slide_data = SlideSegmentationData.load(str(json_path))
        
        # Calculate areas
        cell_type_areas, total_area, total_cells = calculate_cell_areas(slide_data, args.mpp)
        
        # Calculate cancer ratio
        cancer_ratio = calculate_cancer_ratio(cell_type_areas, total_area)
        
        # Print results
        print_results(cell_type_areas, cancer_ratio, args.mpp)
        
        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare output data
            output_data = {
                "cancer_ratio": cancer_ratio,
                "cell_type_areas": cell_type_areas,
                "total_cells": total_cells,
                "mpp": args.mpp,
            }
            
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {output_path}")
    
    except Exception as e:
        print(f"Error analyzing areas: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

