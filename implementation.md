# Tile Predictor Implementation

## Overview

`tile_predictor.py` is a command-line tool for running cell segmentation on individual tile images (PNG, JPG, TIFF) instead of whole slide images (WSIs). It leverages the HistoPLUS CellViT segmentation models to detect and classify cells in histopathology tile images.

## Features

- **Individual Tile Processing**: Process single images or batches of tile images
- **Otsu Thresholding**: Optional background filtering using Otsu's method to skip tiles with too much background
- **Multiple Magnification Levels**: Support for both 20x (MPP 0.5) and 40x (MPP 0.25) models
- **Flexible Input**: Accepts files, directories, or glob patterns
- **Batch Processing**: Efficient batch inference with configurable batch size
- **Parallel Data Loading**: Multi-worker data loading for faster processing
- **Multiple Output Formats**: Single JSON file or per-tile JSON files
- **Comprehensive Metadata**: Output includes model information, processing parameters, and statistics

## Requirements

The script uses the HistoPLUS package and its dependencies:
- Python 3.8+
- PyTorch
- HistoPLUS package (installed in the project)
- PIL/Pillow
- scikit-image
- numpy
- loguru
- tqdm

All dependencies should be available if the HistoPLUS environment is properly set up.

## Installation

No additional installation is required beyond the HistoPLUS package setup. The script can be run directly:

```bash
python tile_predictor.py --help
```

## Usage

### Basic Usage

```bash
# Process a single tile
python tile_predictor.py --tiles tile.png --export_dir ./output/

# Process multiple tiles
python tile_predictor.py --tiles tile1.png tile2.png tile3.png --export_dir ./output/

# Process all tiles in a directory
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/

# Process tiles using glob patterns
python tile_predictor.py --tiles "./tiles/*.png" --export_dir ./output/
```

### With Otsu Filtering

Otsu thresholding can be used to automatically filter out tiles that contain too much background:

```bash
# Enable Otsu filtering (default threshold: 0.8)
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --otsu 1

# Custom Otsu threshold (0-1, lower = more strict)
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --otsu 1 --otsu_threshold 0.7
```

**How Otsu Filtering Works:**
1. Converts each tile to grayscale
2. Applies Otsu's thresholding algorithm to separate foreground from background
3. Calculates the ratio of background pixels
4. Filters out tiles where background ratio exceeds the threshold
5. Only processes tiles that pass the filter

### Advanced Options

```bash
# Use 40x magnification model (MPP 0.25)
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --mpp 0.25

# Custom batch size and workers
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --batch_size 16 --n_workers 8

# Disable mixed precision (for compatibility)
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --mixed_precision 0

# Per-tile output format
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --output_format per_tile

# Silent mode (no progress bars)
python tile_predictor.py --tiles ./tiles/ --export_dir ./output/ --verbose 0
```

## Command-Line Arguments

### Required Arguments

| Argument | Short | Type | Description |
|----------|-------|------|-------------|
| `--tiles` | - | paths | Path(s) to tiles. Can be files, directories, or glob patterns. Multiple paths can be specified. |
| `--export_dir` | `-e` | path | Directory where results will be saved. Will be created if it doesn't exist. |

### Optional Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--otsu` | int | 0 | 0, 1 | Apply Otsu thresholding to filter background tiles. 0=off, 1=on. |
| `--otsu_threshold` | float | 0.8 | 0.0-1.0 | Background ratio threshold for Otsu filtering. Tiles with background ratio above this value are filtered out. |
| `--mpp` | float | 0.5 | 0.25, 0.5 | Microns per pixel. 0.25 for 40x magnification, 0.5 for 20x magnification. |
| `--batch_size` | int | 8 | > 0 | Batch size for processing. Larger batches use more memory but may be faster. |
| `--n_workers` | int | 4 | ≥ 0 | Number of parallel workers for data loading. 0 disables multiprocessing. |
| `--mixed_precision` | int | 1 | 0, 1 | Use mixed precision for inference. 0=off, 1=on. Can improve speed on modern GPUs. |
| `--verbose` | int | 1 | 0, 1 | Verbosity level. 0=silent, 1=show progress bars and logs. |
| `--output_format` | str | json | json, per_tile | Output format. 'json' creates a single file, 'per_tile' creates one file per tile. |

## Output Format

### JSON Format (Default)

When `--output_format json` is used (default), a single JSON file `tile_predictions.json` is created with the following structure:

```json
{
  "metadata": {
    "mpp": 0.5,
    "model": "histoplus_cellvit_segmentor_20x",
    "otsu_filtering": false,
    "otsu_threshold": null,
    "n_tiles_input": 10,
    "n_tiles_processed": 8
  },
  "predictions": {
    "path/to/tile1.png": {
      "contours": [
        [[x1, y1], [x2, y2], ...],
        [[x3, y3], [x4, y4], ...]
      ],
      "bounding_boxes": [
        [y_min, y_max, x_min, x_max],
        [y_min, y_max, x_min, x_max]
      ],
      "centroids": [
        [x, y],
        [x, y]
      ],
      "cell_types": [
        "tumor",
        "lymphocyte"
      ],
      "cell_type_probabilities": [
        0.95,
        0.87
      ],
      "n_cells": 2
    },
    "path/to/tile2.png": {
      ...
    }
  }
}
```

**Metadata Fields:**
- `mpp`: Microns per pixel used for inference
- `model`: Name of the segmentation model used
- `otsu_filtering`: Whether Otsu filtering was applied
- `otsu_threshold`: Otsu threshold value (if filtering was enabled)
- `n_tiles_input`: Total number of tiles found in input
- `n_tiles_processed`: Number of tiles actually processed (after filtering)

**Prediction Fields (per tile):**
- `contours`: List of polygon coordinates for each cell mask
- `bounding_boxes`: Bounding box coordinates [y_min, y_max, x_min, x_max] for each cell
- `centroids`: [x, y] coordinates of cell centroids
- `cell_types`: Cell type classification for each cell
- `cell_type_probabilities`: Confidence scores for cell type predictions
- `n_cells`: Total number of cells detected in the tile

### Per-Tile Format

When `--output_format per_tile` is used, individual JSON files are created in `export_dir/per_tile/` directory:

```
export_dir/
  per_tile/
    tile1_predictions.json
    tile2_predictions.json
    ...
```

Each file contains only the prediction data for that tile (without metadata wrapper).

## Technical Details

### Architecture

The script consists of several key components:

1. **TileDataset**: PyTorch Dataset class that:
   - Loads tile images from disk
   - Applies optional Otsu filtering
   - Resizes images to model input size (784x784 pixels)
   - Applies model-specific transforms (normalization)

2. **Segmentor Loading**: Uses HistoPLUS `CellViTSegmentor.from_histoplus()` to load pre-trained models:
   - Automatically downloads model weights from Hugging Face Hub if not cached
   - Supports both 20x (MPP 0.5) and 40x (MPP 0.25) models
   - Handles device placement (CPU/GPU) automatically

3. **Inference Pipeline**:
   - Batches tiles for efficient GPU utilization
   - Runs forward pass through CellViT model
   - Post-processes raw predictions to extract cell instances
   - Converts outputs to JSON-serializable format

4. **Otsu Filtering**:
   - Converts RGB images to grayscale
   - Applies Otsu's thresholding algorithm
   - Calculates background pixel ratio
   - Filters tiles based on threshold

### Model Details

The script uses HistoPLUS CellViT models trained on the HIPE dataset:
- **20x Model (MPP 0.5)**: `histoplus_cellvit_segmentor_20x`
- **40x Model (MPP 0.25)**: `histoplus_cellvit_segmentor_40x`

Models output three prediction maps:
- **NP (Nuclei Probability)**: Probability map of nuclei presence
- **TP (Type Prediction)**: Cell type classification map
- **HV (HoVer)**: Horizontal and vertical distance maps for instance separation

Post-processing combines these maps to extract individual cell instances with:
- Segmentation masks (polygon contours)
- Bounding boxes
- Centroids
- Cell type classifications
- Confidence scores

### Image Processing

- **Input Size**: Images are automatically resized to 784x784 pixels (model input size)
- **Normalization**: Images are normalized using model-specific mean and std values
- **Color Space**: All images are converted to RGB before processing
- **Supported Formats**: PNG, JPG, JPEG, TIF, TIFF

### Performance Considerations

- **Batch Size**: Larger batch sizes improve GPU utilization but require more memory
- **Workers**: More workers speed up data loading but may cause memory issues
- **Mixed Precision**: Enables faster inference on modern GPUs (recommended)
- **Otsu Filtering**: Adds preprocessing time but can significantly reduce inference time by skipping background tiles

## Examples

### Example 1: Process Directory with Otsu Filtering

```bash
python tile_predictor.py \
  --tiles ./my_tiles/ \
  --export_dir ./results/ \
  --otsu 1 \
  --otsu_threshold 0.75 \
  --batch_size 16 \
  --n_workers 4 \
  --verbose 1
```

This will:
- Find all image files in `./my_tiles/`
- Filter out tiles with >75% background using Otsu
- Process remaining tiles in batches of 16
- Use 4 workers for data loading
- Save results to `./results/tile_predictions.json`

### Example 2: Process Specific Files at 40x

```bash
python tile_predictor.py \
  --tiles tile1.png tile2.png tile3.png \
  --export_dir ./output/ \
  --mpp 0.25 \
  --batch_size 4 \
  --output_format per_tile
```

This will:
- Process only the three specified tiles
- Use the 40x (MPP 0.25) model
- Use smaller batch size (4) for memory efficiency
- Create separate JSON files for each tile

### Example 3: Large-Scale Processing

```bash
python tile_predictor.py \
  --tiles "./data/**/*.png" \
  --export_dir ./large_scale_output/ \
  --otsu 1 \
  --batch_size 32 \
  --n_workers 8 \
  --mpp 0.5 \
  --mixed_precision 1
```

This configuration is optimized for processing many tiles:
- Uses glob pattern to find all PNG files recursively
- Large batch size for GPU efficiency
- Many workers for fast data loading
- Otsu filtering to skip background tiles

## Troubleshooting

### Common Issues

1. **No tiles found**
   - Check that file paths are correct
   - Verify file extensions are supported (.png, .jpg, .jpeg, .tif, .tiff)
   - Ensure files are readable

2. **Out of Memory (OOM) errors**
   - Reduce `--batch_size`
   - Reduce `--n_workers`
   - Process tiles in smaller batches

3. **All tiles filtered by Otsu**
   - Increase `--otsu_threshold` value (e.g., 0.9 or 0.95)
   - Check that tiles actually contain tissue
   - Disable Otsu filtering with `--otsu 0`

4. **Model download issues**
   - Check internet connection
   - Verify Hugging Face Hub access
   - Check `~/.histoplus/hf_cache/` for cached models

5. **CUDA/GPU errors**
   - Verify PyTorch CUDA installation
   - Try `--mixed_precision 0`
   - Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

## Integration with HistoPLUS

This script is designed to complement the main HistoPLUS WSI processing pipeline:

- **WSI Processing**: Use `histoplus extract` for whole slide images
- **Tile Processing**: Use `tile_predictor.py` for individual tiles or extracted tiles

The output format is compatible with HistoPLUS data structures, making it easy to integrate results from both workflows.

## License

This script is part of the HistoPLUS project and follows the same license terms.

## Support

For issues, questions, or contributions related to this script, please refer to the main HistoPLUS project repository.

---

# Summarizer Implementation

## Overview

`summarizer.py` is a command-line tool for generating summary statistics from HistoPLUS WSI output folders. It scans directories containing `cell_masks.json` files (typically from `histoplus extract` runs) and creates a comprehensive CSV report with metrics such as total area, cell counts, tumor percentages, and processing times.

## Features

- **Automatic Folder Scanning**: Recursively finds all `cell_masks.json` files in a directory
- **Area Calculation**: Computes total tissue area in mm² using MPP (microns per pixel) from the model metadata
- **Cell Statistics**: Counts total cells, tiles, and tumor cells
- **Tumor Metrics**: Calculates percentage of tumor cells and tumor area coverage
- **Time Extraction**: Automatically extracts processing time from log files
- **CSV Output**: Generates a clean, structured CSV file for easy analysis

## Requirements

The script uses standard Python libraries:
- Python 3.8+
- numpy
- loguru
- Standard library: json, csv, pathlib, re, argparse

All dependencies should be available in the HistoPLUS environment.

## Installation

No additional installation is required. The script can be run directly:

```bash
python summarizer.py --help
```

## Usage

### Basic Usage

```bash
# Summarize wsi_output folder
python summarizer.py --input wsi_output --output summary.csv
```

### With Separate Log Directory

```bash
# Specify log directory separately
python summarizer.py --input wsi_output --output summary.csv --log_dir wsi_output
```

## Command-Line Arguments

### Required Arguments

| Argument | Short | Type | Description |
|----------|-------|------|-------------|
| `--input` | `-i` | path | Input folder containing WSI output (e.g., wsi_output). Will recursively search for cell_masks.json files. |
| `--output` | `-o` | path | Output CSV file path where summary will be saved. |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--log_dir` | path | None | Directory containing log files. If not specified, uses the input folder. |

## Output Format

The script generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `slide_name` | Name of the slide (extracted from folder name) |
| `model_name` | Model used for inference (e.g., histoplus_cellvit_segmentor_40x) |
| `mpp` | Microns per pixel used for inference (0.25 for 40x, 0.5 for 20x) |
| `total_area_mm2` | Total tissue area in square millimeters |
| `total_cells` | Total number of cells detected across all tiles |
| `total_tiles` | Number of tiles processed |
| `tumor_cells` | Number of cells classified as tumor |
| `pct_tumor_cells` | Percentage of cells that are tumor cells |
| `pct_tumor_area` | Percentage of total area covered by tumor cells |
| `time_taken_seconds` | Processing time in seconds (extracted from log files, if available) |

### Example CSV Output

```csv
slide_name,model_name,mpp,total_area_mm2,total_cells,total_tiles,tumor_cells,pct_tumor_cells,pct_tumor_area,time_taken_seconds
TCGA-G2-A2EO-01Z-00-DX3,histoplus_cellvit_segmentor_40x,0.25,14.6294,262494,4665,120012,45.72,8.02,864.50
TCGA-FU-A3HY-01Z-00-DX1,histoplus_cellvit_segmentor_40x,0.25,23.3914,684183,7459,430387,62.91,18.39,1549.50
MSB-00602-02-01,histoplus_cellvit_segmentor_40x,0.25,2.9792,42324,950,7700,18.19,1.36,214.00
```

## Technical Details

### Area Calculation

The total area in mm² is calculated using the formula:
```
Area (mm²) = (Total Pixels × MPP / 1000)²
```

Where:
- **Total Pixels** = Sum of all tile areas (width × height for each tile)
- **MPP** = Microns per pixel (from model metadata: 0.25 for 40x, 0.5 for 20x)
- **1000** = Conversion factor (1000 microns = 1 mm)

### Tumor Cell Detection

The script identifies tumor cells by matching cell type names against a predefined set of tumor-related terms:
- `tumor`, `tumor cell`, `tumor_cell`, `tumorcell`
- `malignant`, `cancer`, `carcinoma`

The matching is case-insensitive and uses substring matching, so variations like "Tumor Cell" or "tumor_cell" will be detected.

### Tumor Area Calculation

Tumor area is calculated by summing the polygon areas of all tumor cells using the shoelace formula:
```
Area = 0.5 × |Σ(x_i × y_{i+1}) - Σ(y_i × x_{i+1})|
```

The percentage of tumor area is then:
```
% Tumor Area = (Tumor Cell Area / Total Tissue Area) × 100
```

### Time Extraction

The script automatically searches for log files in the specified directory that contain processing information. It looks for patterns like:
- `"Finished processing of ... in X.X seconds"`
- `"--- Finished processing of ... in X.X seconds"`

If a matching log file is found for a slide, the processing time is extracted and included in the CSV output.

### File Structure

The script expects the following directory structure:
```
wsi_output/
├── slide1.svs/
│   └── cell_masks.json
├── slide2.svs/
│   └── cell_masks.json
└── *.log (log files)
```

The script recursively searches for all `cell_masks.json` files, so nested directory structures are supported.

## Examples

### Example 1: Basic Summary

```bash
python summarizer.py --input wsi_output --output summary.csv
```

This will:
- Scan `wsi_output` for all `cell_masks.json` files
- Extract statistics from each slide
- Generate `summary.csv` with all metrics

### Example 2: With Custom Log Directory

```bash
python summarizer.py \
  --input wsi_output \
  --output summary.csv \
  --log_dir /path/to/logs
```

This is useful when log files are stored in a different location than the output folder.

## Output Summary

After processing, the script prints a summary to the console:

```
Summary:
  Total slides processed: 9
  Total cells: 3,291,028
  Total tiles: 42,324
```

## Error Handling

The script handles various edge cases gracefully:

- **Missing files**: Skips slides with missing or corrupted `cell_masks.json` files
- **Empty data**: Handles slides with no cells or tiles
- **Missing MPP**: Leaves area calculation empty if MPP is not available
- **Missing logs**: Leaves time_taken_seconds empty if no matching log file is found
- **Invalid JSON**: Logs errors and continues processing other slides

## Integration with HistoPLUS Workflow

The summarizer is designed to work seamlessly with HistoPLUS output:

1. **Process WSIs**: Run `histoplus extract` to generate `cell_masks.json` files
2. **Generate Summary**: Run `summarizer.py` to create a CSV report
3. **Analyze Results**: Import CSV into Excel, Python pandas, or R for further analysis

The CSV output is compatible with standard data analysis tools and can be easily imported into:
- Excel/Google Sheets
- Python pandas
- R data frames
- SQL databases
- Statistical analysis software

## Troubleshooting

### Common Issues

1. **No cell_masks.json files found**
   - Verify the input folder path is correct
   - Check that `histoplus extract` has been run successfully
   - Ensure the output directory structure matches expected format

2. **Area calculation shows empty values**
   - Check that MPP is present in the JSON metadata
   - Verify the model was run with valid MPP values (0.25 or 0.5)

3. **Time extraction not working**
   - Verify log files exist in the specified directory
   - Check that log files contain the expected pattern
   - Try specifying `--log_dir` explicitly

4. **Tumor percentages seem incorrect**
   - Verify cell type names in the JSON match expected tumor types
   - Check that the model is correctly classifying cells
   - Review the cell type mapping used during inference

5. **CSV file is empty**
   - Check that at least one valid `cell_masks.json` file was found
   - Verify file permissions allow reading
   - Check for JSON parsing errors in the console output

## Performance

The script processes files sequentially and is optimized for:
- **Memory efficiency**: Processes one slide at a time
- **Large datasets**: Can handle hundreds of slides
- **Fast execution**: Typically processes 10-20 slides per minute (depending on file sizes)

For very large datasets (1000+ slides), consider:
- Running the script in batches
- Using parallel processing (future enhancement)
- Processing during off-peak hours

## License

This script is part of the HistoPLUS project and follows the same license terms.

## Support

For issues, questions, or contributions related to this script, please refer to the main HistoPLUS project repository.

