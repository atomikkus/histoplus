#!/usr/bin/env python3
"""Generate overview images of WSI with and without cell masks."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator

from collections import Counter

from histoplus.helpers.data.slide_segmentation_data import SlideSegmentationData


def get_overview_level(slide: openslide.OpenSlide, target_size: int = 2000) -> int:
    """
    Get the appropriate level for overview image.
    
    Args:
        slide: OpenSlide object
        target_size: Target maximum dimension for overview
        
    Returns:
        Level index to use
    """
    level_count = slide.level_count
    for level in range(level_count):
        dims = slide.level_dimensions[level]
        if max(dims) <= target_size:
            return level
    return level_count - 1


def get_level_downsample(slide: openslide.OpenSlide, level: int) -> float:
    """Get the downsample factor for a given level."""
    # Validate level
    if level < 0 or level >= slide.level_count:
        raise ValueError(f"Invalid level {level}. Slide has {slide.level_count} levels.")
    
    # Try to use level_downsamples if available
    if hasattr(slide, "level_downsamples") and slide.level_downsamples is not None:
        try:
            return float(slide.level_downsamples[level])
        except (IndexError, TypeError):
            pass
    
    # Fallback calculation
    level0_dims = slide.level_dimensions[0]
    level_dims = slide.level_dimensions[level]
    return float(level0_dims[0]) / float(level_dims[0])


def load_overview_image(slide: openslide.OpenSlide, level: int) -> np.ndarray:
    """
    Load overview image from WSI at specified level.
    
    Args:
        slide: OpenSlide object
        level: Level to load
        
    Returns:
        RGB image as numpy array
    """
    dims = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, dims)
    img = np.array(img)
    # Convert RGBA to RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def calculate_class_statistics(slide_data: SlideSegmentationData) -> dict[str, dict[str, float]]:
    """
    Calculate statistics for each cell class.
    
    Args:
        slide_data: SlideSegmentationData containing masks
        
    Returns:
        Dictionary with class statistics (count, percentage)
    """
    cell_type_counts = Counter()
    total_cells = 0
    
    # Count cells by type
    for tile in slide_data.cell_masks:
        for mask in tile.masks:
            cell_type_counts[mask.cell_type] += 1
            total_cells += 1
    
    # Calculate percentages
    statistics = {}
    for cell_type, count in cell_type_counts.items():
        percentage = (count / total_cells * 100) if total_cells > 0 else 0.0
        statistics[cell_type] = {
            "count": count,
            "percentage": percentage,
        }
    
    # Sort by count (descending)
    statistics = dict(sorted(statistics.items(), key=lambda x: x[1]["count"], reverse=True))
    
    return statistics, total_cells


def print_class_statistics(slide_data: SlideSegmentationData, statistics: dict = None, total_cells: int = None) -> None:
    """Print class statistics to console."""
    if statistics is None or total_cells is None:
        statistics, total_cells = calculate_class_statistics(slide_data)
    
    print("\n" + "=" * 60)
    print("CELL CLASS STATISTICS")
    print("=" * 60)
    print(f"Total cells detected: {total_cells:,}")
    print("-" * 60)
    print(f"{'Cell Type':<25} {'Count':<12} {'Percentage':<12}")
    print("-" * 60)
    
    for cell_type, stats in statistics.items():
        print(f"{cell_type:<25} {stats['count']:<12,} {stats['percentage']:>10.2f}%")
    
    print("=" * 60 + "\n")


def get_deepzoom_level_downsample(
    deepzoom: DeepZoomGenerator,
    dz_level: int,
) -> float:
    """
    Get the downsample factor for a DeepZoom level relative to the highest resolution level.
    
    In DeepZoom, level 0 is the LOWEST resolution (most downsampled),
    and level N-1 is the HIGHEST resolution (base level, no downsampling).
    
    Args:
        deepzoom: DeepZoomGenerator object
        dz_level: DeepZoom level
        
    Returns:
        Downsample factor relative to the highest resolution level (level N-1)
    """
    n_levels = deepzoom.level_count
    # DeepZoom: level 0 = most downsampled, level N-1 = base (no downsampling)
    # Level k is downsampled by 2^(N-1-k) from the base
    return 2.0 ** (n_levels - 1 - dz_level)


def draw_masks_on_image(
    img: np.ndarray,
    slide_data: SlideSegmentationData,
    overview_level: int,
    slide: openslide.OpenSlide,
    deepzoom: DeepZoomGenerator,
    tile_size: int = 224,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw cell masks on the overview image.
    
    Args:
        img: Overview image
        slide_data: SlideSegmentationData containing masks
        overview_level: OpenSlide level of the overview image
        slide: OpenSlide object
        deepzoom: DeepZoomGenerator for coordinate conversion
        tile_size: Tile size used during extraction (default: 224)
        alpha: Transparency of mask overlay
        
    Returns:
        Image with masks drawn
    """
    # Create a copy for overlay
    overlay = img.copy()
    
    # Get downsample factor for overview level (OpenSlide level)
    overview_downsample = get_level_downsample(slide, overview_level)
    
    # Color mapping for different cell types
    cell_type_colors = {
        "Cancer cell": (255, 0, 0),      # Red
        "Apoptotic Body": (0, 255, 0),   # Green
        "T cell": (0, 0, 255),           # Blue
        "B cell": (255, 255, 0),         # Yellow
        "Macrophage": (255, 0, 255),     # Magenta
        "Neutrophil": (0, 255, 255),     # Cyan
        "Stromal cell": (128, 128, 128), # Gray
    }
    
    # Default color for unknown cell types
    default_color = (255, 165, 0)  # Orange
    
    # Get image dimensions for bounds checking
    img_height, img_width = img.shape[:2]
    
    tiles_processed = 0
    masks_drawn = 0
    
    for tile in slide_data.cell_masks:
        dz_level = int(tile.level)
        
        # Validate DeepZoom level
        if dz_level < 0 or dz_level >= deepzoom.level_count:
            continue
        
        # Get DeepZoom level dimensions
        dz_level_dims = deepzoom.level_dimensions[dz_level]
        dz_level0_dims = deepzoom.level_dimensions[0]  # Lowest resolution level
        
        # Convert tile coordinates to pixel coordinates at the DeepZoom level
        # Tile coordinates are in tile units at the DeepZoom level
        tile_x_dz = tile.x * tile_size
        tile_y_dz = tile.y * tile_size
        
        # Convert from DeepZoom level pixels to OpenSlide level 0 pixels
        # Use the ratio of level 0 dimensions (which represents the base resolution)
        # DeepZoom level 0 is the most downsampled, so we need to scale up
        dz_level0_to_base_ratio = dz_level0_dims[0] / slide.level_dimensions[0][0]
        dz_to_base_ratio = dz_level_dims[0] / slide.level_dimensions[0][0]
        
        # Actually, let's use a simpler approach: convert via DeepZoom level dimensions
        # The ratio between DeepZoom level dimensions tells us the scale
        # Level 0 (lowest res) to level N-1 (highest res) ratio
        highest_dz_level = deepzoom.level_count - 1
        highest_dz_dims = deepzoom.level_dimensions[highest_dz_level]
        
        # Scale from current DeepZoom level to highest DeepZoom level (which is close to OpenSlide level 0)
        scale_to_highest_dz = highest_dz_dims[0] / dz_level_dims[0]
        
        # Convert tile position to highest DeepZoom level (approximately OpenSlide level 0)
        tile_x_level0 = int(tile_x_dz * scale_to_highest_dz)
        tile_y_level0 = int(tile_y_dz * scale_to_highest_dz)
        
        # Now scale from level 0 to overview level
        tile_x_overview = int(tile_x_level0 / overview_downsample)
        tile_y_overview = int(tile_y_level0 / overview_downsample)
        
        # Skip if tile is completely outside the overview image bounds
        if (tile_x_overview + tile_size < 0 or 
            tile_y_overview + tile_size < 0 or
            tile_x_overview >= img_width or
            tile_y_overview >= img_height):
            continue
        
        tiles_processed += 1
        
        # Scale factor for mask coordinates (from tile space to overview space)
        # Mask coordinates are relative to the tile, in pixels at the DeepZoom level
        # Need to scale to overview level
        mask_scale_factor = scale_to_highest_dz / overview_downsample
        
        # Draw each mask in the tile
        for mask in tile.masks:
            # Get color for cell type
            color = cell_type_colors.get(mask.cell_type, default_color)
            
            # Convert mask coordinates from tile space to overview space
            if not mask.coordinates or len(mask.coordinates) < 3:
                continue  # Skip invalid polygons
            
            mask_coords = np.array(mask.coordinates, dtype=np.float32)
            
            # Scale mask coordinates from tile space to overview space
            scaled_coords = mask_coords * mask_scale_factor
            
            # Translate to tile position in overview
            scaled_coords[:, 0] += tile_x_overview
            scaled_coords[:, 1] += tile_y_overview
            
            # Clip coordinates to image bounds
            scaled_coords[:, 0] = np.clip(scaled_coords[:, 0], 0, img_width - 1)
            scaled_coords[:, 1] = np.clip(scaled_coords[:, 1], 0, img_height - 1)
            
            # Convert to integer coordinates
            coords = scaled_coords.astype(np.int32)
            
            # Draw filled polygon
            try:
                cv2.fillPoly(overlay, [coords], color)
                masks_drawn += 1
            except Exception as e:
                continue
    
    print(f"Processed {tiles_processed} tiles, drew {masks_drawn} masks")
    
    # Blend overlay with original image
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return result


def create_overview_images(
    svs_path: str,
    json_path: str,
    output_path: str,
    target_size: int = 2000,
    alpha: float = 0.5,
) -> None:
    """
    Create overview images with and without masks.
    
    Args:
        svs_path: Path to .svs file
        json_path: Path to cell_masks.json file
        output_path: Path to save output PNG file
        target_size: Target maximum dimension for overview
        alpha: Transparency of mask overlay
    """
    # Validate inputs
    svs_path = Path(svs_path)
    json_path = Path(json_path)
    
    if not svs_path.exists():
        raise FileNotFoundError(f"SVS file not found: {svs_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    print(f"Loading slide: {svs_path}")
    slide = openslide.open_slide(str(svs_path))
    
    print(f"Loading masks: {json_path}")
    slide_data = SlideSegmentationData.load(str(json_path))
    
    # Calculate and print class statistics
    statistics, total_cells = calculate_class_statistics(slide_data)
    print_class_statistics(slide_data, statistics, total_cells)
    
    # Create DeepZoomGenerator for coordinate conversion
    # Use the same tile size as was used during extraction (typically 224)
    # We need to infer this from the data - check first tile width
    tile_size = int(slide_data.cell_masks[0].width) if slide_data.cell_masks else 224
    print(f"Using tile size: {tile_size}")
    
    deepzoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    
    # Get appropriate level for overview
    overview_level = get_overview_level(slide, target_size)
    print(f"Using OpenSlide level {overview_level} for overview (dimensions: {slide.level_dimensions[overview_level]})")
    
    # Load overview image
    print("Loading overview image...")
    overview_img = load_overview_image(slide, overview_level)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot image without masks
    axes[0].imshow(overview_img)
    axes[0].set_title("WSI Overview (Without Masks)", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # Plot image with masks
    print("Drawing masks on overview...")
    img_with_masks = draw_masks_on_image(
        overview_img, slide_data, overview_level, slide, deepzoom, tile_size=tile_size, alpha=alpha
    )
    axes[1].imshow(img_with_masks)
    axes[1].set_title("WSI Overview (With Cell Masks)", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    
    # Add overall title
    fig.suptitle(
        f"Cell Segmentation Overview\n{svs_path.name}",
        fontsize=16,
        fontweight="bold",
    )
    
    # Add statistics text box
    stats_text = f"Total Cells: {total_cells:,}\n\n"
    stats_text += "Cell Type Distribution:\n"
    for cell_type, stats in list(statistics.items())[:8]:  # Show top 8 classes
        stats_text += f"  {cell_type}: {stats['percentage']:.1f}%\n"
    if len(statistics) > 8:
        stats_text += f"  ... and {len(statistics) - 8} more"
    
    # Add text box with statistics
    fig.text(
        0.02, 0.02, stats_text,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving overview to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save statistics to JSON file
    stats_output_path = output_path.parent / f"{output_path.stem}_statistics.json"
    stats_dict = {
        "total_cells": total_cells,
        "classes": statistics,
    }
    with open(stats_output_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Statistics saved to: {stats_output_path}")
    
    print(f"Overview image saved successfully!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate overview images of WSI with and without cell masks"
    )
    parser.add_argument(
        "svs_file",
        type=str,
        help="Path to the .svs file",
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the cell_masks.json file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PNG file path (default: <svs_name>_overview.png)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=2000,
        help="Target maximum dimension for overview (default: 2000)",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency of mask overlay (0.0-1.0, default: 0.5)",
    )
    
    args = parser.parse_args()
    
    # Generate default output path if not provided
    if args.output is None:
        svs_path = Path(args.svs_file)
        args.output = svs_path.parent / f"{svs_path.stem}_overview.png"
    
    try:
        create_overview_images(
            args.svs_file,
            args.json_file,
            args.output,
            target_size=args.size,
            alpha=args.alpha,
        )
    except Exception as e:
        print(f"Error creating overview: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

