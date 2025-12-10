#!/usr/bin/env python3
"""Process any .svs whole slide image using histoplus."""

import argparse
import os
import sys
from pathlib import Path

import openslide
from histoplus.extract import extract
from histoplus.helpers.segmentor import CellViTSegmentor
from histoplus.helpers.tissue_detection import detect_tissue_on_wsi

MPP = 0.25  # If available, otherwise set to 0.5
INFERENCE_IMAGE_SIZE = 784


def process_svs(svs_path: str, output_dir: str = "output", batch_size: int = 8) -> None:
    """
    Process a .svs whole slide image and extract cell segmentation masks.

    Args:
        svs_path: Path to the .svs file
        output_dir: Directory to save the results
        batch_size: Batch size for processing
    """
    # Validate input file
    svs_path = Path(svs_path)
    if not svs_path.exists():
        raise FileNotFoundError(f"File not found: {svs_path}")
    
    if not svs_path.suffix.lower() == ".svs":
        raise ValueError(f"File must be a .svs file: {svs_path}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening slide: {svs_path}")
    slide = openslide.open_slide(str(svs_path))
    
    print("Detecting tissue on WSI...")
    tissue_coords, dz_level = detect_tissue_on_wsi(slide)
    
    print("Initializing CellViT segmentor...")
    segmentor = CellViTSegmentor.from_histoplus(
        mpp=MPP,
        mixed_precision=True,
        inference_image_size=INFERENCE_IMAGE_SIZE,
    )
    
    print("Processing whole slide image...")
    # Process a whole slide image
    results = extract(
        slide=slide,
        coords=tissue_coords,
        deepzoom_level=dz_level,
        segmentor=segmentor,
        batch_size=batch_size,
    )
    
    # Generate output filename based on input filename
    output_filename = svs_path.stem + "_results.json"
    output_file = output_path / output_filename
    
    print(f"Saving results to: {output_file}")
    results.save(str(output_file))
    
    print(f"Processing complete! Results saved to {output_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process a .svs whole slide image and extract cell segmentation masks"
    )
    parser.add_argument(
        "svs_file",
        type=str,
        help="Path to the .svs file to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    
    args = parser.parse_args()
    
    try:
        process_svs(args.svs_file, args.output, args.batch_size)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

