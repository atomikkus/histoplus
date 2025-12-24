#!/usr/bin/env python3
"""Pipeline script that processes WSI slides and generates a comprehensive PDF report."""

import argparse
import base64
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import importlib.util

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from histoplus.helpers.data.slide_segmentation_data import SlideSegmentationData

# Import local modules
_script_dir = Path(__file__).parent
_calc_area_spec = importlib.util.spec_from_file_location("calculate_cancer_area", _script_dir / "calculate_cancer_area.py")
_calc_area_module = importlib.util.module_from_spec(_calc_area_spec)
_calc_area_spec.loader.exec_module(_calc_area_module)

_view_overview_spec = importlib.util.spec_from_file_location("view_overview", _script_dir / "view_overview.py")
_view_overview_module = importlib.util.module_from_spec(_view_overview_spec)
_view_overview_spec.loader.exec_module(_view_overview_module)

# Import functions
calculate_polygon_area = _calc_area_module.calculate_polygon_area
calculate_cell_areas = _calc_area_module.calculate_cell_areas
calculate_cancer_ratio = _calc_area_module.calculate_cancer_ratio
create_overview_images = _view_overview_module.create_overview_images
calculate_class_statistics = _view_overview_module.calculate_class_statistics


def run_histoplus(slide_path: str, export_dir: str, batch_size: int = 12, verbose: int = 1) -> Path:
    """
    Run histoplus CLI command to process the slide.
    
    Args:
        slide_path: Path to the .svs slide file
        export_dir: Directory to export results
        batch_size: Batch size for processing
        verbose: Verbosity level
        
    Returns:
        Path to the generated cell_masks.json file
    """
    print("=" * 80)
    print("STEP 1: Running HistoPLUS Segmentation")
    print("=" * 80)
    
    slide_path = Path(slide_path)
    if not slide_path.exists():
        raise FileNotFoundError(f"Slide file not found: {slide_path}")
    
    # Run histoplus command
    cmd = [
        "histoplus",
        "--slides", str(slide_path),
        "--export_dir", export_dir,
        "--batch_size", str(batch_size),
        "--verbose", str(verbose),
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running histoplus:")
        print(result.stderr)
        raise RuntimeError(f"HistoPLUS processing failed with return code {result.returncode}")
    
    print(result.stdout)
    
    # Find the generated cell_masks.json file
    # HistoPLUS uses slide_path.name (includes .svs extension) for the folder name
    slide_name_with_ext = slide_path.name
    slide_name_without_ext = slide_path.stem
    
    # Try with extension first (as histoplus CLI does)
    json_path = Path(export_dir) / slide_name_with_ext / "cell_masks.json"
    
    # If not found, try without extension (fallback)
    if not json_path.exists():
        json_path = Path(export_dir) / slide_name_without_ext / "cell_masks.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected output file not found. Checked:\n"
            f"  - {Path(export_dir) / slide_name_with_ext / 'cell_masks.json'}\n"
            f"  - {Path(export_dir) / slide_name_without_ext / 'cell_masks.json'}"
        )
    
    print(f"✓ HistoPLUS processing complete. Results saved to: {json_path}")
    return json_path


def generate_overview(svs_path: str, json_path: Path, output_dir: Path) -> Path:
    """
    Generate overview images.
    
    Args:
        svs_path: Path to the .svs file
        json_path: Path to cell_masks.json
        output_dir: Directory to save overview image
        
    Returns:
        Path to the generated overview PNG file
    """
    print("\n" + "=" * 80)
    print("STEP 2: Generating Overview Images")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / f"{Path(svs_path).stem}_overview.png"
    
    create_overview_images(
        svs_path=str(svs_path),
        json_path=str(json_path),
        output_path=str(overview_path),
    )
    
    print(f"✓ Overview image saved to: {overview_path}")
    return overview_path


def calculate_areas(json_path: Path, mpp: float = None) -> tuple:
    """
    Calculate cell areas and cancer ratios.
    
    Args:
        json_path: Path to cell_masks.json
        mpp: Microns per pixel (optional)
        
    Returns:
        Tuple of (cell_type_areas, cancer_ratio, total_area, total_cells, cell_class_stats)
    """
    print("\n" + "=" * 80)
    print("STEP 3: Calculating Cell Areas")
    print("=" * 80)
    
    slide_data = SlideSegmentationData.load(str(json_path))
    cell_type_areas, total_area, total_cells = calculate_cell_areas(slide_data, mpp)
    cancer_ratio = calculate_cancer_ratio(cell_type_areas, total_area)
    
    # Use existing calculate_class_statistics from view_overview.py
    cell_class_stats, _ = calculate_class_statistics(slide_data)
    
    print(f"✓ Area calculation complete.")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Cancer cells: {cancer_ratio['cancer_count']:,} ({cancer_ratio['cancer_percentage']:.2f}%)")
    
    return cell_type_areas, cancer_ratio, total_area, total_cells, cell_class_stats


def create_pdf_report(
    svs_path: str,
    overview_path: Path,
    cell_type_areas: dict,
    cancer_ratio: dict,
    total_cells: int,
    output_path: Path,
    mpp: float = None,
):
    """
    Create a comprehensive PDF report.
    
    Args:
        svs_path: Path to the original slide file
        overview_path: Path to overview PNG image
        cell_type_areas: Dictionary with area statistics per cell type
        cancer_ratio: Dictionary with cancer vs non-cancer statistics
        total_cells: Total number of cells
        output_path: Path to save PDF report
        mpp: Microns per pixel (optional)
    """
    print("\n" + "=" * 80)
    print("STEP 4: Generating PDF Report")
    print("=" * 80)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Professional color scheme
    PRIMARY_COLOR = '#2C3E50'  # Dark blue-gray
    ACCENT_COLOR = '#3498DB'   # Blue
    SUCCESS_COLOR = '#27AE60'  # Green
    WARNING_COLOR = '#E67E22'  # Orange
    DANGER_COLOR = '#E74C3C'   # Red
    LIGHT_GRAY = '#ECF0F1'
    DARK_GRAY = '#34495E'
    
    slide_name = Path(svs_path).name
    date_str = datetime.now().strftime("%B %d, %Y at %H:%M")
    
    with PdfPages(str(output_path)) as pdf:
        # Page 1: Professional Cover Page
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        
        # Header bar
        header = plt.Rectangle((0, 0.88), 1, 0.12, transform=fig.transFigure, 
                              facecolor=PRIMARY_COLOR, edgecolor='none')
        fig.patches.append(header)
        
        # Title
        fig.text(0.5, 0.92, "HistoPLUS", 
                ha='center', va='center', fontsize=32, fontweight='bold', color='white')
        fig.text(0.5, 0.88, "Cell Segmentation Analysis Report", 
                ha='center', va='center', fontsize=18, color='white', style='italic')
        
        # Slide information box
        info_box_y = 0.70
        info_box = plt.Rectangle((0.15, info_box_y - 0.15), 0.7, 0.20, 
                                 transform=fig.transFigure, 
                                 facecolor=LIGHT_GRAY, edgecolor=DARK_GRAY, linewidth=1.5)
        fig.patches.append(info_box)
        
        fig.text(0.5, info_box_y + 0.02, "Slide Information", 
                ha='center', va='top', fontsize=14, fontweight='bold', color=DARK_GRAY)
        fig.text(0.5, info_box_y - 0.02, slide_name, 
                ha='center', va='top', fontsize=11, color=DARK_GRAY, family='monospace')
        fig.text(0.5, info_box_y - 0.08, f"Generated: {date_str}", 
                ha='center', va='top', fontsize=9, color=DARK_GRAY, style='italic')
        
        # Key metrics boxes
        metrics_y = 0.50
        box_width = 0.25
        box_height = 0.12
        spacing = 0.05
        
        # Total Cells box
        x1 = 0.15
        box1 = plt.Rectangle((x1, metrics_y), box_width, box_height,
                             transform=fig.transFigure,
                             facecolor=ACCENT_COLOR, edgecolor='white', linewidth=2)
        fig.patches.append(box1)
        fig.text(x1 + box_width/2, metrics_y + box_height/2 + 0.02, "Total Cells",
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        fig.text(x1 + box_width/2, metrics_y + box_height/2 - 0.02, f"{total_cells:,}",
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # Cancer Cells box
        x2 = x1 + box_width + spacing
        box2 = plt.Rectangle((x2, metrics_y), box_width, box_height,
                             transform=fig.transFigure,
                             facecolor=DANGER_COLOR, edgecolor='white', linewidth=2)
        fig.patches.append(box2)
        fig.text(x2 + box_width/2, metrics_y + box_height/2 + 0.02, "Cancer Cells",
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        fig.text(x2 + box_width/2, metrics_y + box_height/2 - 0.02, 
                f"{cancer_ratio['cancer_percentage']:.1f}%",
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # Non-Cancer Cells box
        x3 = x2 + box_width + spacing
        box3 = plt.Rectangle((x3, metrics_y), box_width, box_height,
                             transform=fig.transFigure,
                             facecolor=SUCCESS_COLOR, edgecolor='white', linewidth=2)
        fig.patches.append(box3)
        fig.text(x3 + box_width/2, metrics_y + box_height/2 + 0.02, "Non-Cancer Cells",
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        fig.text(x3 + box_width/2, metrics_y + box_height/2 - 0.02,
                f"{cancer_ratio['non_cancer_percentage']:.1f}%",
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # Detailed statistics section
        stats_y = 0.30
        stats_box = plt.Rectangle((0.1, stats_y - 0.20), 0.8, 0.22,
                                  transform=fig.transFigure,
                                  facecolor='white', edgecolor=DARK_GRAY, linewidth=1.5)
        fig.patches.append(stats_box)
        
        fig.text(0.5, stats_y + 0.01, "Executive Summary", 
                ha='center', va='top', fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
        
        # Two-column layout for statistics
        col1_x = 0.15
        col2_x = 0.55
        
        summary_text_col1 = f"""Cancer Cells:
  • Count: {cancer_ratio['cancer_count']:,} cells
  • Area: {cancer_ratio['cancer_area_pixels']:,.0f} px²
  • Percentage: {cancer_ratio['cancer_percentage']:.2f}%
  • Ratio: {cancer_ratio['cancer_to_total_ratio']:.4f}"""
        
        summary_text_col2 = f"""Non-Cancer Cells:
  • Count: {cancer_ratio['non_cancer_count']:,} cells
  • Area: {cancer_ratio['non_cancer_area_pixels']:,.0f} px²
  • Percentage: {cancer_ratio['non_cancer_percentage']:.2f}%
  • Ratio: {cancer_ratio['cancer_to_non_cancer_ratio']:.4f}"""
        
        if mpp is not None:
            cancer_area_um2 = cancer_ratio['cancer_area_pixels'] * (mpp ** 2)
            non_cancer_area_um2 = cancer_ratio['non_cancer_area_pixels'] * (mpp ** 2)
            summary_text_col1 += f"\n  • Physical Area: {cancer_area_um2:,.0f} μm²"
            summary_text_col2 += f"\n  • Physical Area: {non_cancer_area_um2:,.0f} μm²"
        
        fig.text(col1_x, stats_y - 0.05, summary_text_col1,
                ha='left', va='top', fontsize=9, color=DARK_GRAY, family='monospace')
        fig.text(col2_x, stats_y - 0.05, summary_text_col2,
                ha='left', va='top', fontsize=9, color=DARK_GRAY, family='monospace')
        
        # Footer
        footer = plt.Rectangle((0, 0), 1, 0.05, transform=fig.transFigure,
                              facecolor=PRIMARY_COLOR, edgecolor='none')
        fig.patches.append(footer)
        fig.text(0.5, 0.025, "Confidential - For Research Purposes Only",
                ha='center', va='center', fontsize=8, color='white', style='italic')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 2: Overview Images
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        
        # Header
        header = plt.Rectangle((0, 0.92), 1, 0.08, transform=fig.transFigure,
                              facecolor=PRIMARY_COLOR, edgecolor='none')
        fig.patches.append(header)
        fig.text(0.5, 0.94, "Whole Slide Image Overview", 
                ha='center', va='center', fontsize=18, fontweight='bold', color='white')
        
        # Load and display overview image
        img = plt.imread(str(overview_path))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        ax.set_position([0.05, 0.05, 0.9, 0.85])  # Leave space for header
        
        # Footer
        footer = plt.Rectangle((0, 0), 1, 0.03, transform=fig.transFigure,
                              facecolor=LIGHT_GRAY, edgecolor='none')
        fig.patches.append(footer)
        fig.text(0.5, 0.015, f"Page 2 of 4 - {slide_name}",
                ha='center', va='center', fontsize=8, color=DARK_GRAY)
        
        pdf.savefig(fig, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Page 3: Detailed Cell Type Statistics
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        
        # Header
        header = plt.Rectangle((0, 0.92), 1, 0.08, transform=fig.transFigure,
                              facecolor=PRIMARY_COLOR, edgecolor='none')
        fig.patches.append(header)
        fig.text(0.5, 0.94, "Detailed Cell Type Statistics", 
                ha='center', va='center', fontsize=18, fontweight='bold', color='white')
                # Create table
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_position([0.05, 0.10, 0.9, 0.80])  # Leave space for header and footer
        
        # Table headers
        if mpp is not None:
            headers = ["Cell Type", "Count", "Area (px²)", "Area (μm²)", "% of Total"]
            col_widths = [0.32, 0.14, 0.14, 0.14, 0.14]
        else:
            headers = ["Cell Type", "Count", "Area (px²)", "% of Total"]
            col_widths = [0.40, 0.20, 0.20, 0.20]
        
        # Table data
        table_data = []
        for cell_type, data in cell_type_areas.items():
            row = [cell_type, f"{data['count']:,}", f"{data['area_pixels']:,.0f}", f"{data['percentage']:.2f}%"]
            if mpp is not None:
                row.insert(3, f"{data['area_um2']:,.0f}")
            table_data.append(row)
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='left',
            loc='center',
            colWidths=col_widths,
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.8)
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(PRIMARY_COLOR)
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_edgecolor('white')
            table[(0, i)].set_linewidth(1.5)
        
        # Style data rows (alternating colors)
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                table[(i, j)].set_edgecolor('#BDC3C7')
                table[(i, j)].set_linewidth(0.5)
                if i % 2 == 0:
                    table[(i, j)].set_facecolor(LIGHT_GRAY)
                else:
                    table[(i, j)].set_facecolor('white')
        
        # Footer
        footer = plt.Rectangle((0, 0), 1, 0.03, transform=fig.transFigure,
                              facecolor=LIGHT_GRAY, edgecolor='none')
        fig.patches.append(footer)
        fig.text(0.5, 0.015, f"Page 3 of 4 - {slide_name}",
                ha='center', va='center', fontsize=8, color=DARK_GRAY)
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Page 4: Visualizations
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        
        # Header
        header = plt.Rectangle((0, 0.92), 1, 0.08, transform=fig.transFigure,
                              facecolor=PRIMARY_COLOR, edgecolor='none')
        fig.patches.append(header)
        fig.text(0.5, 0.94, "Cell Distribution Visualizations", 
                ha='center', va='center', fontsize=18, fontweight='bold', color='white')
        
        # Create subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Adjust subplot positions
        ax1.set_position([0.08, 0.15, 0.40, 0.70])
        ax2.set_position([0.52, 0.15, 0.40, 0.70])
        
        # Pie chart of cancer vs non-cancer
        cancer_pct = cancer_ratio['cancer_percentage']
        non_cancer_pct = cancer_ratio['non_cancer_percentage']
        
        colors_pie = [DANGER_COLOR, SUCCESS_COLOR]
        wedges, texts, autotexts = ax1.pie([cancer_pct, non_cancer_pct], 
                labels=['Cancer Cells', 'Non-Cancer Cells'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors_pie,
                textprops={'fontsize': 11, 'fontweight': 'bold'},
                explode=(0.05, 0))  # Slight explode for emphasis
        
        # Style pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax1.set_title("Cancer vs Non-Cancer\n(by area)", 
                     fontsize=13, fontweight='bold', color=PRIMARY_COLOR, pad=15)
        
        # Bar chart of top cell types
        top_n = min(10, len(cell_type_areas))
        top_types = list(cell_type_areas.items())[:top_n]
        
        cell_types = [t[0] for t in top_types]
        percentages = [t[1]['percentage'] for t in top_types]
        
        # Use gradient colors
        colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(cell_types)))
        bars = ax2.barh(range(len(cell_types)), percentages, color=colors_bar, edgecolor='white', linewidth=1)
        ax2.set_yticks(range(len(cell_types)))
        ax2.set_yticklabels(cell_types, fontsize=8.5)
        ax2.set_xlabel('Percentage of Total Area (%)', fontsize=10, fontweight='bold', color=DARK_GRAY)
        ax2.set_title(f"Top {top_n} Cell Types\n(by area)", 
                     fontsize=13, fontweight='bold', color=PRIMARY_COLOR, pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--', color='gray')
        ax2.set_facecolor('white')
        
        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax2.text(pct + max(percentages) * 0.02, i, f'{pct:.1f}%', 
                    va='center', fontsize=8, fontweight='bold', color=DARK_GRAY)
        
        # Style axes
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if ax == ax2:
                ax.spines['left'].set_color('#BDC3C7')
                ax.spines['bottom'].set_color('#BDC3C7')
        
        # Footer
        footer = plt.Rectangle((0, 0), 1, 0.03, transform=fig.transFigure,
                              facecolor=LIGHT_GRAY, edgecolor='none')
        fig.patches.append(footer)
        fig.text(0.5, 0.015, f"Page 4 of 4 - {slide_name}",
                ha='center', va='center', fontsize=8, color=DARK_GRAY)
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"✓ PDF report saved to: {output_path}")


def create_html_report(
    svs_path: str,
    overview_path: Path,
    cell_type_areas: dict,
    cancer_ratio: dict,
    total_cells: int,
    output_path: Path,
    mpp: float = None,
    cell_class_stats: dict = None,
):
    """
    Create a comprehensive HTML report.
    
    Args:
        svs_path: Path to the original slide file
        overview_path: Path to overview PNG image
        cell_type_areas: Dictionary with area statistics per cell type
        cancer_ratio: Dictionary with cancer vs non-cancer statistics
        total_cells: Total number of cells
        output_path: Path to save HTML report
        mpp: Microns per pixel (optional)
        cell_class_stats: Dictionary with cell class statistics (counts and percentages)
    """
    print("\n" + "=" * 80)
    print("STEP 4: Generating HTML Report")
    print("=" * 80)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Professional color scheme
    PRIMARY_COLOR = '#2C3E50'
    ACCENT_COLOR = '#3498DB'
    SUCCESS_COLOR = '#27AE60'
    DANGER_COLOR = '#E74C3C'
    LIGHT_GRAY = '#ECF0F1'
    DARK_GRAY = '#34495E'
    
    slide_name = Path(svs_path).name
    date_str = datetime.now().strftime("%B %d, %Y at %H:%M")
    
    # Calculate count-based percentages for cancer and non-cancer cells
    cancer_count_pct = 0.0
    non_cancer_count_pct = 0.0
    if cell_class_stats:
        for cell_type, stats in cell_class_stats.items():
            if "cancer" in cell_type.lower():
                cancer_count_pct += stats['percentage']
            else:
                non_cancer_count_pct += stats['percentage']
    else:
        # Fallback calculation from cancer_ratio
        cancer_count_pct = (cancer_ratio['cancer_count'] / total_cells * 100) if total_cells > 0 else 0.0
        non_cancer_count_pct = (cancer_ratio['non_cancer_count'] / total_cells * 100) if total_cells > 0 else 0.0
    
    # Convert overview image to base64
    with open(overview_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
        overview_img_base64 = f"data:image/png;base64,{img_data}"

    # Convert logo to base64 (ignore if missing)
    logo_base64 = ""
    logo_path = Path("/home/satyaprakash/Github/histoplus/4bC logo.svg")
    if logo_path.exists():
        with open(logo_path, "rb") as logo_file:
            logo_base64 = base64.b64encode(logo_file.read()).decode("utf-8")
    
    # Create visualizations
    # Pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    cancer_pct = cancer_ratio['cancer_percentage']
    non_cancer_pct = cancer_ratio['non_cancer_percentage']
    colors_pie = [DANGER_COLOR, SUCCESS_COLOR]
    wedges, texts, autotexts = ax_pie.pie(
        [cancer_pct, non_cancer_pct],
        labels=['Cancer Cells', 'Non-Cancer Cells'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        explode=(0.05, 0),
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax_pie.set_title("Cancer vs Non-Cancer\n(by area)", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
    plt.tight_layout()
    pie_img = io.BytesIO()
    fig_pie.savefig(pie_img, format='png', dpi=100, bbox_inches='tight')
    pie_img.seek(0)
    pie_base64 = base64.b64encode(pie_img.read()).decode('utf-8')
    plt.close(fig_pie)
    
    # Bar chart
    top_n = min(10, len(cell_type_areas))
    top_types = list(cell_type_areas.items())[:top_n]
    cell_types = [t[0] for t in top_types]
    percentages = [t[1]['percentage'] for t in top_types]
    
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(cell_types)))
    bars = ax_bar.barh(range(len(cell_types)), percentages, color=colors_bar, edgecolor='white', linewidth=1)
    ax_bar.set_yticks(range(len(cell_types)))
    ax_bar.set_yticklabels(cell_types, fontsize=9)
    ax_bar.set_xlabel('Percentage of Total Area (%)', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax_bar.set_title(f"Top {top_n} Cell Types (by area)", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
    ax_bar.grid(axis='x', alpha=0.3, linestyle='--', color='gray')
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax_bar.text(pct + max(percentages) * 0.02, i, f'{pct:.1f}%',
                   va='center', fontsize=9, fontweight='bold', color=DARK_GRAY)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    plt.tight_layout()
    bar_img = io.BytesIO()
    fig_bar.savefig(bar_img, format='png', dpi=100, bbox_inches='tight')
    bar_img.seek(0)
    bar_base64 = base64.b64encode(bar_img.read()).decode('utf-8')
    plt.close(fig_bar)
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HistoPLUS Analysis Report - {slide_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: {DARK_GRAY};
            background-color: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {DARK_GRAY} 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: relative;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-style: italic;
        }}

    .logo-container {{
        position: absolute;
        left: 2rem;
        top: 1.25rem;
        height: 48px;
        display: flex;
        align-items: center;
    }}

    .logo-container img {{
        height: 48px;
        width: auto;
    }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .section {{
            background: white;
            margin: 2rem 0;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: {PRIMARY_COLOR};
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid {ACCENT_COLOR};
        }}
        
        .info-box {{
            background: {LIGHT_GRAY};
            padding: 1.5rem;
            border-radius: 6px;
            margin: 1rem 0;
            border-left: 4px solid {ACCENT_COLOR};
        }}
        
        .info-box h3 {{
            color: {PRIMARY_COLOR};
            margin-bottom: 0.5rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .metric-card.total {{
            border-top: 4px solid {ACCENT_COLOR};
        }}
        
        .metric-card.cancer {{
            border-top: 4px solid {DANGER_COLOR};
        }}
        
        .metric-card.non-cancer {{
            border-top: 4px solid {SUCCESS_COLOR};
        }}
        
        .metric-card h3 {{
            color: {DARK_GRAY};
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {PRIMARY_COLOR};
        }}
        
        .metric-card .label {{
            font-size: 0.85rem;
            color: #7f8c8d;
            margin-top: 0.5rem;
        }}
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}
        
        .stats-table thead {{
            background: {PRIMARY_COLOR};
            color: white;
        }}
        
        .stats-table th {{
            padding: 1rem;
            text-align: left;
            font-weight: 600;
        }}
        
        .stats-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid {LIGHT_GRAY};
        }}
        
        .stats-table tbody tr:nth-child(even) {{
            background: {LIGHT_GRAY};
        }}
        
        .stats-table tbody tr:hover {{
            background: #d5dbdb;
        }}
        
        .overview-image {{
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin: 1rem 0;
        }}
        
        .visualizations {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }}
        
        .viz-container {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }}
        
        .viz-container img {{
            width: 100%;
            height: auto;
        }}
        
        .footer {{
            background: {PRIMARY_COLOR};
            color: white;
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
            font-size: 0.85rem;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}
        
        .summary-box {{
            background: {LIGHT_GRAY};
            padding: 1.5rem;
            border-radius: 6px;
            border-left: 4px solid {ACCENT_COLOR};
        }}
        
        .summary-box h4 {{
            color: {PRIMARY_COLOR};
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        
        .summary-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .summary-box li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid #bdc3c7;
        }}
        
        .summary-box li:last-child {{
            border-bottom: none;
        }}
        
        .summary-box strong {{
            color: {DARK_GRAY};
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        {"<div class='logo-container'><img src='data:image/svg+xml;base64," + logo_base64 + "' alt='Logo'></div>" if logo_base64 else ""}
        <h1>Tumor Content Analysis</h1>
        <p>Cell Segmentation Analysis Report</p>
    </div>
    
    <div class="container">
        <!-- Slide Information -->
        <div class="section">
            <h2>Slide Information</h2>
            <div class="info-box">
                <h3>{slide_name}</h3>
                <p><strong>Report Date:</strong> {date_str}</p>
            </div>
        </div>
        
        <!-- Key Metrics -->
        <div class="section">
            <h2>Key Metrics</h2>
            <p style="margin-bottom: 1rem; color: #7f8c8d; font-size: 0.9rem;">
                Percentages shown are by area. Count-based percentages are shown below.
            </p>
            <div class="metrics-grid">
                <div class="metric-card total">
                    <h3>Total Cells</h3>
                    <div class="value">{total_cells:,}</div>
                    <div class="label">Cells Detected</div>
                </div>
                <div class="metric-card cancer">
                    <h3>Cancer Cells</h3>
                    <div class="value">{cancer_ratio['cancer_percentage']:.1f}%</div>
                    <div class="label" style="font-size: 0.75rem; margin-top: 0.3rem; line-height: 1.4;">
                        {cancer_ratio['cancer_count']:,} cells<br>
                        <span style="color: #7f8c8d;">{cancer_count_pct:.1f}% by count</span>
                    </div>
                </div>
                <div class="metric-card non-cancer">
                    <h3>Non-Cancer Cells</h3>
                    <div class="value">{cancer_ratio['non_cancer_percentage']:.1f}%</div>
                    <div class="label" style="font-size: 0.75rem; margin-top: 0.3rem; line-height: 1.4;">
                        {cancer_ratio['non_cancer_count']:,} cells<br>
                        <span style="color: #7f8c8d;">{non_cancer_count_pct:.1f}% by count</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-box">
                    <h4>Cancer Cells</h4>
                    <ul>
                        <li><strong>Count:</strong> {cancer_ratio['cancer_count']:,} cells</li>
                        <li><strong>Area:</strong> {cancer_ratio['cancer_area_pixels']:,.0f} px²</li>
                        <li><strong>Percentage:</strong> {cancer_ratio['cancer_percentage']:.2f}%</li>
                        <li><strong>Ratio (Cancer/Total):</strong> {cancer_ratio['cancer_to_total_ratio']:.4f}</li>"""
    
    if mpp is not None:
        cancer_area_um2 = cancer_ratio['cancer_area_pixels'] * (mpp ** 2)
        html_content += f"""
                        <li><strong>Physical Area:</strong> {cancer_area_um2:,.0f} μm²</li>"""
    
    html_content += f"""
                    </ul>
                </div>
                <div class="summary-box">
                    <h4>Non-Cancer Cells</h4>
                    <ul>
                        <li><strong>Count:</strong> {cancer_ratio['non_cancer_count']:,} cells</li>
                        <li><strong>Area:</strong> {cancer_ratio['non_cancer_area_pixels']:,.0f} px²</li>
                        <li><strong>Percentage:</strong> {cancer_ratio['non_cancer_percentage']:.2f}%</li>
                        <li><strong>Ratio (Cancer/Non-Cancer):</strong> {cancer_ratio['cancer_to_non_cancer_ratio']:.4f}</li>"""
    
    if mpp is not None:
        non_cancer_area_um2 = cancer_ratio['non_cancer_area_pixels'] * (mpp ** 2)
        html_content += f"""
                        <li><strong>Physical Area:</strong> {non_cancer_area_um2:,.0f} μm²</li>"""
    
    html_content += f"""
                    </ul>
                </div>
            </div>
        </div>
        
        <!-- Overview Image -->
        <div class="section">
            <h2>Whole Slide Image Overview</h2>
            <img src="{overview_img_base64}" alt="WSI Overview" class="overview-image">
        </div>
        
        <!-- Visualizations -->
        <div class="section">
            <h2>Cell Distribution Visualizations</h2>
            <div class="visualizations">
                <div class="viz-container">
                    <h3 style="text-align: center; color: {PRIMARY_COLOR}; margin-bottom: 1rem;">Cancer vs Non-Cancer</h3>
                    <img src="data:image/png;base64,{pie_base64}" alt="Pie Chart">
                </div>
                <div class="viz-container">
                    <h3 style="text-align: center; color: {PRIMARY_COLOR}; margin-bottom: 1rem;">Top {top_n} Cell Types</h3>
                    <img src="data:image/png;base64,{bar_base64}" alt="Bar Chart">
                </div>
            </div>
        </div>
        
        <!-- Detailed Statistics -->
        <div class="section">
            <h2>Detailed Cell Type Statistics</h2>
            <table class="stats-table">
                <thead>
                    <tr>"""
    
    if mpp is not None:
        html_content += """
                        <th>Cell Type</th>
                        <th>Count</th>
                        <th>Area (px²)</th>
                        <th>Area (μm²)</th>
                        <th>% of Total</th>"""
    else:
        html_content += """
                        <th>Cell Type</th>
                        <th>Count</th>
                        <th>Area (px²)</th>
                        <th>% of Total</th>"""
    
    html_content += """
                    </tr>
                </thead>
                <tbody>"""
    
    for cell_type, data in cell_type_areas.items():
        html_content += f"""
                    <tr>
                        <td><strong>{cell_type}</strong></td>
                        <td>{data['count']:,}</td>
                        <td>{data['area_pixels']:,.0f}</td>"""
        if mpp is not None:
            html_content += f"""
                        <td>{data['area_um2']:,.0f}</td>"""
        html_content += f"""
                        <td>{data['percentage']:.2f}%</td>
                    </tr>"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <!-- Cell Class Statistics -->
        <div class="section">
            <h2>Cell Class Statistics</h2>
            <p style="margin-bottom: 1rem; color: #7f8c8d;">Distribution of cells by type (based on cell counts)</p>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Cell Type</th>
                        <th>Count</th>
                        <th>% of Total Cells</th>
                    </tr>
                </thead>
                <tbody>"""
    
    if cell_class_stats:
        for cell_type, stats in cell_class_stats.items():
            html_content += f"""
                    <tr>
                        <td><strong>{cell_type}</strong></td>
                        <td>{stats['count']:,}</td>
                        <td>{stats['percentage']:.2f}%</td>
                    </tr>"""
    else:
        # Fallback: use cell_type_areas if cell_class_stats not provided
        for cell_type, data in cell_type_areas.items():
            count_pct = (data['count'] / total_cells * 100) if total_cells > 0 else 0.0
            html_content += f"""
                    <tr>
                        <td><strong>{cell_type}</strong></td>
                        <td>{data['count']:,}</td>
                        <td>{count_pct:.2f}%</td>
                    </tr>"""
    
    html_content += """
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="footer">
        <p>Confidential - For Research Purposes Only</p>
    </div>
</body>
</html>"""
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved to: {output_path}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline script for WSI processing and report generation"
    )
    parser.add_argument(
        "slide_path",
        type=str,
        help="Path to the .svs slide file",
    )
    parser.add_argument(
        "-e", "--export_dir",
        type=str,
        default="wsi_output",
        help="Directory to export histoplus results (default: wsi_output)",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=12,
        help="Batch size for histoplus processing (default: 12)",
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        help="Verbosity level for histoplus (default: 1)",
    )
    parser.add_argument(
        "-m", "--mpp",
        type=float,
        default=None,
        help="Microns per pixel for area calculations (optional)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output HTML report path (default: <slide_name>_report.html)",
    )
    parser.add_argument(
        "--skip_histoplus",
        action="store_true",
        help="Skip histoplus processing (use existing results)",
    )
    
    args = parser.parse_args()
    
    try:
        slide_path = Path(args.slide_path)
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
        
        export_dir = Path(args.export_dir)
        slide_name = slide_path.stem
        
        # Step 1: Run histoplus (or skip if requested)
        if args.skip_histoplus:
            print("Skipping HistoPLUS processing (using existing results)")
            # HistoPLUS uses slide_path.name (includes .svs extension) for the folder name
            slide_name_with_ext = slide_path.name
            slide_name_without_ext = slide_path.stem
            
            # Try with extension first (as histoplus CLI does)
            json_path = export_dir / slide_name_with_ext / "cell_masks.json"
            
            # If not found, try without extension (fallback)
            if not json_path.exists():
                json_path = export_dir / slide_name_without_ext / "cell_masks.json"
            
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Expected results file not found. Checked:\n"
                    f"  - {export_dir / slide_name_with_ext / 'cell_masks.json'}\n"
                    f"  - {export_dir / slide_name_without_ext / 'cell_masks.json'}"
                )
        else:
            json_path = run_histoplus(
                str(slide_path),
                str(export_dir),
                args.batch_size,
                args.verbose,
            )
        
        # Step 2: Generate overview images
        overview_path = generate_overview(
            str(slide_path),
            json_path,
            export_dir,
        )
        
        # Step 3: Calculate areas
        cell_type_areas, cancer_ratio, total_area, total_cells, cell_class_stats = calculate_areas(
            json_path,
            args.mpp,
        )
        
        # Step 4: Generate HTML report
        if args.output:
            html_path = Path(args.output)
        else:
            html_path = export_dir / f"{slide_name}_report.html"
        
        create_html_report(
            str(slide_path),
            overview_path,
            cell_type_areas,
            cancer_ratio,
            total_cells,
            html_path,
            args.mpp,
            cell_class_stats,
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"✓ HistoPLUS results: {json_path}")
        print(f"✓ Overview image: {overview_path}")
        print(f"✓ HTML report: {html_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

