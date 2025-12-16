#!/usr/bin/env python3
"""Tile Predictor CLI.

A script to run cell segmentation on individual tiles (images) instead of WSIs.
Supports optional Otsu thresholding to filter out background tiles.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from histoplus.helpers.constants import INFERENCE_TILE_SIZE
from histoplus.helpers.segmentor import CellViTSegmentor


class TileDataset(Dataset):
    """Dataset for loading individual tile images."""

    def __init__(
        self,
        tile_paths: list[Path],
        transform=None,
        apply_otsu: bool = False,
        otsu_threshold: float = 0.8,
    ):
        """Initialize the tile dataset.

        Parameters
        ----------
        tile_paths : list[Path]
            List of paths to tile images.
        transform : callable, optional
            Transform to apply to images.
        apply_otsu : bool
            Whether to apply Otsu thresholding to filter tiles.
        otsu_threshold : float
            Threshold for Otsu filtering (0-1). Tiles with more than this
            fraction of background pixels are filtered out.
        """
        self.tile_paths = tile_paths
        self.transform = transform
        self.apply_otsu = apply_otsu
        self.otsu_threshold = otsu_threshold

        # Filter tiles based on Otsu if enabled
        if self.apply_otsu:
            self.valid_indices, self.filtered_paths = self._filter_tiles_with_otsu()
        else:
            self.valid_indices = list(range(len(tile_paths)))
            self.filtered_paths = tile_paths

    def _filter_tiles_with_otsu(self) -> tuple[list[int], list[Path]]:
        """Filter tiles using Otsu thresholding.

        Returns
        -------
        tuple[list[int], list[Path]]
            Valid indices and filtered paths.
        """
        valid_indices = []
        filtered_paths = []

        logger.info("Filtering tiles with Otsu thresholding...")
        for idx, path in enumerate(tqdm(self.tile_paths, desc="Otsu filtering")):
            try:
                img = Image.open(path).convert("RGB")
                img_arr = np.array(img)
                gray = rgb2gray(img_arr)

                # Apply Otsu threshold
                try:
                    thresh = threshold_otsu(gray)
                    binary = gray > thresh
                    background_ratio = np.mean(binary)

                    # Keep tile if background ratio is below threshold
                    if background_ratio < self.otsu_threshold:
                        valid_indices.append(idx)
                        filtered_paths.append(path)
                except ValueError:
                    # Handle edge cases (e.g., uniform images)
                    valid_indices.append(idx)
                    filtered_paths.append(path)

            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")
                continue

        logger.info(
            f"Kept {len(valid_indices)}/{len(self.tile_paths)} tiles after Otsu filtering"
        )
        return valid_indices, filtered_paths

    def __len__(self) -> int:
        return len(self.filtered_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Get a tile and its path.

        Parameters
        ----------
        idx : int
            Index of the tile.

        Returns
        -------
        tuple[torch.Tensor, str]
            Transformed image tensor and the file path.
        """
        path = self.filtered_paths[idx]
        img = Image.open(path).convert("RGB")

        # Resize to model's expected size if needed
        img = img.resize((INFERENCE_TILE_SIZE, INFERENCE_TILE_SIZE), Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)

        return img, str(path)


def collate_fn(batch):
    """Custom collate function to handle paths."""
    images, paths = zip(*batch)
    images = torch.stack(images)
    return images, list(paths)


def get_segmentor(mpp: float = 0.5, mixed_precision: bool = True) -> CellViTSegmentor:
    """Get the segmentor model.

    Parameters
    ----------
    mpp : float
        Microns per pixel (0.25 for 40x, 0.5 for 20x).
    mixed_precision : bool
        Whether to use mixed precision.

    Returns
    -------
    CellViTSegmentor
        The segmentor model.
    """
    return CellViTSegmentor.from_histoplus(
        mpp=mpp,
        mixed_precision=mixed_precision,
        inference_image_size=INFERENCE_TILE_SIZE,
    )


def predict_tiles(
    segmentor: CellViTSegmentor,
    dataloader: DataLoader,
    verbose: int = 1,
) -> dict[str, dict]:
    """Run prediction on tiles.

    Parameters
    ----------
    segmentor : CellViTSegmentor
        The segmentation model.
    dataloader : DataLoader
        DataLoader for tiles.
    verbose : int
        Verbosity level.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping tile paths to predictions.
    """
    segmentor.model.eval()
    predictions = {}

    postprocess_fn = segmentor.get_postprocess_fn()

    iterator = tqdm(dataloader, desc="Predicting") if verbose else dataloader

    with torch.no_grad():
        for images, paths in iterator:
            # Move to same device as model
            device = next(segmentor.model.parameters()).device
            images = images.to(device)

            # Forward pass
            outputs = segmentor.forward(images)

            # Convert outputs to numpy for postprocessing
            outputs_np = {
                key: val.cpu().numpy() for key, val in outputs.items()
            }

            # Postprocess
            tile_predictions = postprocess_fn(outputs_np)

            # Store predictions
            for path, pred in zip(paths, tile_predictions):
                # Convert numpy arrays to lists for JSON serialization
                bboxes = pred.bounding_boxes
                if isinstance(bboxes, np.ndarray):
                    bboxes = bboxes.tolist()
                elif bboxes and isinstance(bboxes[0], np.ndarray):
                    bboxes = [b.tolist() for b in bboxes]

                pred_dict = {
                    "contours": [c.tolist() if isinstance(c, np.ndarray) else c for c in pred.contours],
                    "bounding_boxes": bboxes,
                    "centroids": pred.centroids,
                    "cell_types": pred.cell_types,
                    "cell_type_probabilities": pred.cell_type_probabilities,
                    "n_cells": len(pred.cell_types),
                }
                predictions[path] = pred_dict

    return predictions


def collect_tile_paths(
    input_paths: list[str],
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> list[Path]:
    """Collect tile paths from input paths.

    Parameters
    ----------
    input_paths : list[str]
        List of paths (files or directories).
    extensions : tuple[str, ...]
        Valid file extensions.

    Returns
    -------
    list[Path]
        List of tile paths.
    """
    tile_paths = []

    for input_path in input_paths:
        path = Path(input_path)

        if path.is_file():
            if path.suffix.lower() in extensions:
                tile_paths.append(path)
        elif path.is_dir():
            for ext in extensions:
                tile_paths.extend(sorted(path.glob(f"*{ext}")))
                tile_paths.extend(sorted(path.glob(f"*{ext.upper()}")))
        else:
            # Handle glob patterns
            import glob
            matched = glob.glob(str(path))
            for m in matched:
                m_path = Path(m)
                if m_path.is_file() and m_path.suffix.lower() in extensions:
                    tile_paths.append(m_path)

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for p in tile_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return unique_paths


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run cell segmentation on individual tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tiles in a directory
  python tile_predictor.py --tiles ./tiles/ --export_dir ./output/
  
  # Process specific files
  python tile_predictor.py --tiles tile1.png tile2.png --export_dir ./output/
  
  # Process with Otsu filtering
  python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --otsu 1
  
  # Process at 40x magnification
  python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --mpp 0.25
        """,
    )

    # Required arguments
    parser.add_argument(
        "--tiles",
        nargs="+",
        required=True,
        help="Path(s) to tiles. Can be files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--export_dir",
        "-e",
        type=str,
        required=True,
        help="Directory where results will be saved.",
    )

    # Otsu filtering
    parser.add_argument(
        "--otsu",
        type=int,
        choices=[0, 1],
        default=0,
        help="Apply Otsu thresholding to filter background tiles (0=off, 1=on). Default: 0",
    )
    parser.add_argument(
        "--otsu_threshold",
        type=float,
        default=0.8,
        help="Background ratio threshold for Otsu filtering (0-1). Default: 0.8",
    )

    # Processing parameters
    parser.add_argument(
        "--mpp",
        type=float,
        choices=[0.25, 0.5],
        default=0.5,
        help="Microns per pixel (0.25 for 40x, 0.5 for 20x). Default: 0.5",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing. Default: 8",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of parallel workers for data loading. Default: 4",
    )
    parser.add_argument(
        "--mixed_precision",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use mixed precision for inference (0=off, 1=on). Default: 1",
    )

    # Output parameters
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=progress). Default: 1",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "per_tile"],
        default="json",
        help="Output format: 'json' (single file) or 'per_tile' (one file per tile). Default: json",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.add(sys.stderr, level="INFO")
    else:
        logger.remove()

    # Create export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Collect tile paths
    logger.info("Collecting tile paths...")
    tile_paths = collect_tile_paths(args.tiles)

    if len(tile_paths) == 0:
        logger.error("No valid tile files found!")
        sys.exit(1)

    logger.info(f"Found {len(tile_paths)} tile(s)")

    # Load segmentor
    logger.info(f"Loading segmentor (MPP={args.mpp})...")
    segmentor = get_segmentor(
        mpp=args.mpp,
        mixed_precision=bool(args.mixed_precision),
    )

    # Create dataset
    dataset = TileDataset(
        tile_paths=tile_paths,
        transform=segmentor.transform,
        apply_otsu=bool(args.otsu),
        otsu_threshold=args.otsu_threshold,
    )

    if len(dataset) == 0:
        logger.error("No tiles remaining after filtering!")
        sys.exit(1)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Run prediction
    logger.info("Running predictions...")
    predictions = predict_tiles(
        segmentor=segmentor,
        dataloader=dataloader,
        verbose=args.verbose,
    )

    # Save results
    if args.output_format == "json":
        output_path = export_dir / "tile_predictions.json"
        
        # Add metadata
        output_data = {
            "metadata": {
                "mpp": args.mpp,
                "model": segmentor.segmentor_name,
                "otsu_filtering": bool(args.otsu),
                "otsu_threshold": args.otsu_threshold if args.otsu else None,
                "n_tiles_input": len(tile_paths),
                "n_tiles_processed": len(predictions),
            },
            "predictions": predictions,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved predictions to {output_path}")

    else:  # per_tile
        per_tile_dir = export_dir / "per_tile"
        per_tile_dir.mkdir(exist_ok=True)

        for tile_path, pred in predictions.items():
            tile_name = Path(tile_path).stem
            output_path = per_tile_dir / f"{tile_name}_predictions.json"
            with open(output_path, "w") as f:
                json.dump(pred, f, indent=2)

        logger.info(f"Saved per-tile predictions to {per_tile_dir}")

    # Summary
    total_cells = sum(pred["n_cells"] for pred in predictions.values())
    logger.info(f"Processing complete! Found {total_cells} cells across {len(predictions)} tiles.")


if __name__ == "__main__":
    main()

