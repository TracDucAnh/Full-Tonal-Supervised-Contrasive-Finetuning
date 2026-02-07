"""
Data Exploration Script for Vietnamese Lexical Sound Dataset

This script provides comprehensive analysis of the dataset structure including:
- Total number of samples
- Distribution across lexical units (tones: sắc, hỏi, ngã, nặng, không dấu/ngang, huyền)
- Distribution across TTS models
- File format verification
- Dataset statistics and visualization

Dataset Structure:
- Root directory contains subdirectories for each TTS model
- Each model directory contains subdirectories for each lexical unit (tone)
- Audio files are named as: {lexical_unit}_{model_name}.wav
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Configuration
# ============================================================================
DATASET_ROOT = "dataset"  # Root path to dataset
OUTPUT_DIR = "exploration_results"  # Output directory for results


# ============================================================================
# Helper Functions
# ============================================================================

def parse_filename(filename):
    """
    Parse filename to extract lexical unit and model name information.
    
    Format: {lexical_unit}_{model_name}.wav
    
    Args:
        filename (str): File name (e.g., "hỏi_mms.wav")
    
    Returns:
        tuple: (lexical_unit, model_name) or (None, None) if invalid
    """
    if not filename.endswith('.wav'):
        return None, None
    
    # Remove .wav extension
    name_without_ext = filename[:-4]
    
    # Split by the last underscore
    parts = name_without_ext.rsplit('_', 1)
    
    if len(parts) == 2:
        lexical_unit, model_name = parts
        return lexical_unit, model_name
    else:
        return None, None


def scan_dataset(root_path):
    """
    Scan entire dataset and collect audio file information.
    
    Dataset structure:
    - root/
      - model1/
        - lexical_unit1/
          - file1.wav
          - file2.wav
        - lexical_unit2/
          - file1.wav
      - model2/
        - lexical_unit1/
          - file1.wav
    
    Args:
        root_path (str): Root path to dataset directory
    
    Returns:
        list: List of dictionaries containing file information
    """
    file_info_list = []
    root = Path(root_path)
    
    # Iterate through all model directories
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Iterate through all lexical unit directories within model
        for lexical_dir in model_dir.iterdir():
            if not lexical_dir.is_dir():
                continue
            
            lexical_unit_folder = lexical_dir.name
            
            # Iterate through all .wav files in lexical unit directory
            for audio_file in lexical_dir.glob('*.wav'):
                filename = audio_file.name
                lexical_unit_parsed, model_name_parsed = parse_filename(filename)
                
                file_info = {
                    'filepath': str(audio_file),
                    'filename': filename,
                    'model_name': model_name,
                    'lexical_unit_folder': lexical_unit_folder,
                    'lexical_unit_parsed': lexical_unit_parsed,
                    'model_name_parsed': model_name_parsed,
                    'file_size_bytes': audio_file.stat().st_size
                }
                file_info_list.append(file_info)
    
    return file_info_list


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_basic_statistics(df):
    """
    Compute basic statistics about the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset information
    
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        'total_samples': len(df),
        'total_lexical_units': df['lexical_unit_folder'].nunique(),
        'total_models': df['model_name'].nunique(),
        'total_size_mb': df['file_size_bytes'].sum() / (1024 * 1024),
        'avg_file_size_kb': df['file_size_bytes'].mean() / 1024,
        'min_file_size_kb': df['file_size_bytes'].min() / 1024,
        'max_file_size_kb': df['file_size_bytes'].max() / 1024
    }
    return stats


def analyze_distribution_by_lexical_unit(df):
    """
    Analyze distribution of samples by lexical unit (tone).
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset information
    
    Returns:
        pd.Series: Count of samples for each lexical unit
    """
    return df['lexical_unit_folder'].value_counts().sort_index()


def analyze_distribution_by_model(df):
    """
    Analyze distribution of samples by TTS model.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset information
    
    Returns:
        pd.Series: Count of samples for each model
    """
    return df['model_name'].value_counts()


def create_cross_tabulation(df):
    """
    Create cross-tabulation between lexical units and models.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset information
    
    Returns:
        pd.DataFrame: Cross-tabulation showing sample counts
    """
    return pd.crosstab(
        df['lexical_unit_folder'], 
        df['model_name'], 
        margins=True
    )


def analyze_model_completeness(df):
    """
    Analyze completeness of each model (whether all tones are present).
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset information
    
    Returns:
        pd.DataFrame: Model completeness report
    """
    # Get all unique tones from the dataset
    all_tones = df['lexical_unit_folder'].unique()
    expected_tones = set(all_tones)
    
    model_completeness = []
    for model in sorted(df['model_name'].unique()):
        model_data = df[df['model_name'] == model]
        present_tones = set(model_data['lexical_unit_folder'].unique())
        missing_tones = expected_tones - present_tones
        
        completeness = {
            'model': model,
            'total_samples': len(model_data),
            'tones_present': len(present_tones),
            'tones_expected': len(expected_tones),
            'tones_missing': len(missing_tones),
            'missing_tone_list': ', '.join(sorted(missing_tones)) if missing_tones else 'None',
            'completeness_percentage': (len(present_tones) / len(expected_tones)) * 100 if expected_tones else 0
        }
        model_completeness.append(completeness)
    
    return pd.DataFrame(model_completeness)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_distribution_by_lexical_unit(lexical_dist, output_dir):
    """
    Plot distribution by lexical unit (tone).
    
    Args:
        lexical_dist (pd.Series): Distribution by lexical unit
        output_dir (str): Directory to save plot
    """
    plt.figure(figsize=(12, 6))
    lexical_dist.plot(kind='bar', color='steelblue')
    plt.title('Sample Distribution by Lexical Unit (Tone)', fontsize=14, fontweight='bold')
    plt.xlabel('Lexical Unit', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribution_by_lexical_unit.png', dpi=300)
    plt.close()
    print(f"✓ Saved plot: distribution_by_lexical_unit.png")


def plot_distribution_by_model(model_dist, output_dir):
    """
    Plot distribution by TTS model.
    
    Args:
        model_dist (pd.Series): Distribution by model
        output_dir (str): Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    model_dist.plot(kind='bar', color='coral')
    plt.title('Sample Distribution by TTS Model', fontsize=14, fontweight='bold')
    plt.xlabel('TTS Model', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribution_by_model.png', dpi=300)
    plt.close()
    print(f"✓ Saved plot: distribution_by_model.png")


def plot_heatmap(crosstab_df, output_dir):
    """
    Plot heatmap for lexical units vs models cross-tabulation.
    
    Args:
        crosstab_df (pd.DataFrame): Cross-tabulation table
        output_dir (str): Directory to save plot
    """
    # Remove 'All' row and column if present
    heatmap_data = crosstab_df.iloc[:-1, :-1] if 'All' in crosstab_df.index else crosstab_df
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Sample Count'})
    plt.title('Distribution Matrix: Lexical Units × TTS Models', fontsize=14, fontweight='bold')
    plt.xlabel('TTS Model', fontsize=12)
    plt.ylabel('Lexical Unit', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_lexical_vs_model.png', dpi=300)
    plt.close()
    print(f"✓ Saved plot: heatmap_lexical_vs_model.png")


def plot_file_size_distribution(df, output_dir):
    """
    Plot file size distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing dataset information
        output_dir (str): Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['file_size_bytes'] / 1024, bins=50, color='mediumseagreen', edgecolor='black')
    plt.title('Audio File Size Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('File Size (KB)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/file_size_distribution.png', dpi=300)
    plt.close()
    print(f"✓ Saved plot: file_size_distribution.png")


def plot_model_completeness(completeness_df, output_dir):
    """
    Plot model completeness comparison.
    
    Args:
        completeness_df (pd.DataFrame): Model completeness data
        output_dir (str): Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Completeness percentage
    ax1.barh(completeness_df['model'], completeness_df['completeness_percentage'], 
             color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Completeness (%)', fontsize=12)
    ax1.set_ylabel('TTS Model', fontsize=12)
    ax1.set_title('Model Completeness (Tone Coverage)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Total samples per model
    ax2.barh(completeness_df['model'], completeness_df['total_samples'], 
             color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Total Samples', fontsize=12)
    ax2.set_ylabel('TTS Model', fontsize=12)
    ax2.set_title('Total Samples per Model', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_completeness.png', dpi=300)
    plt.close()
    print(f"✓ Saved plot: model_completeness.png")


# ============================================================================
# Report Generation
# ============================================================================

def generate_text_report(stats, lexical_dist, model_dist, crosstab_df, 
                        completeness_df, output_dir):
    """
    Generate detailed text report about the dataset.
    
    Args:
        stats (dict): Basic statistics
        lexical_dist (pd.Series): Distribution by lexical unit
        model_dist (pd.Series): Distribution by model
        crosstab_df (pd.DataFrame): Cross-tabulation table
        completeness_df (pd.DataFrame): Model completeness data
        output_dir (str): Directory to save report
    """
    report_path = f'{output_dir}/dataset_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ANALYSIS REPORT - VIETNAMESE LEXICAL SOUND\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples:            {stats['total_samples']:,}\n")
        f.write(f"Total lexical units:      {stats['total_lexical_units']}\n")
        f.write(f"Total TTS models:         {stats['total_models']}\n")
        f.write(f"Total storage size:       {stats['total_size_mb']:.2f} MB\n")
        f.write(f"Average file size:        {stats['avg_file_size_kb']:.2f} KB\n")
        f.write(f"Min/Max file size:        {stats['min_file_size_kb']:.2f} / {stats['max_file_size_kb']:.2f} KB\n")
        f.write("\n")
        
        # Distribution by Lexical Unit
        f.write("2. DISTRIBUTION BY LEXICAL UNIT (TONE)\n")
        f.write("-" * 80 + "\n")
        for unit, count in lexical_dist.items():
            f.write(f"{unit:20s}: {count:4d} samples\n")
        f.write("\n")
        
        # Distribution by Model
        f.write("3. DISTRIBUTION BY TTS MODEL\n")
        f.write("-" * 80 + "\n")
        for model, count in model_dist.items():
            f.write(f"{model:20s}: {count:4d} samples\n")
        f.write("\n")
        
        # Model Completeness
        f.write("4. MODEL COMPLETENESS ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(completeness_df.to_string(index=False))
        f.write("\n\n")
        
        # Cross-tabulation
        f.write("5. DISTRIBUTION MATRIX (Lexical Units × Models)\n")
        f.write("-" * 80 + "\n")
        f.write(crosstab_df.to_string())
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved text report: dataset_report.txt")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function to execute the entire dataset analysis workflow.
    """
    print("=" * 80)
    print("DATASET ANALYSIS STARTED")
    print("=" * 80)
    
    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n✓ Created output directory: {OUTPUT_DIR}")
    
    # 1. Scan dataset
    print(f"\n[1/7] Scanning dataset from: {DATASET_ROOT}...")
    file_info_list = scan_dataset(DATASET_ROOT)
    df = pd.DataFrame(file_info_list)
    print(f"✓ Found {len(df)} audio files")
    
    # Save DataFrame to CSV
    csv_path = f'{OUTPUT_DIR}/dataset_files.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved file list: dataset_files.csv")
    
    # 2. Compute basic statistics
    print("\n[2/7] Computing basic statistics...")
    stats = compute_basic_statistics(df)
    print("✓ Completed statistical computation")
    
    # 3. Analyze distributions
    print("\n[3/7] Analyzing distributions...")
    lexical_dist = analyze_distribution_by_lexical_unit(df)
    model_dist = analyze_distribution_by_model(df)
    crosstab_df = create_cross_tabulation(df)
    completeness_df = analyze_model_completeness(df)
    print("✓ Completed distribution analysis")
    
    # 4. Generate text report
    print("\n[4/7] Generating text report...")
    generate_text_report(stats, lexical_dist, model_dist, crosstab_df, 
                        completeness_df, OUTPUT_DIR)
    
    # 5. Create visualizations
    print("\n[5/7] Creating visualizations...")
    plot_distribution_by_lexical_unit(lexical_dist, OUTPUT_DIR)
    plot_distribution_by_model(model_dist, OUTPUT_DIR)
    plot_heatmap(crosstab_df, OUTPUT_DIR)
    plot_file_size_distribution(df, OUTPUT_DIR)
    plot_model_completeness(completeness_df, OUTPUT_DIR)
    
    # 6. Save completeness analysis to CSV
    print("\n[6/7] Saving completeness analysis...")
    completeness_path = f'{OUTPUT_DIR}/model_completeness.csv'
    completeness_df.to_csv(completeness_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved completeness report: model_completeness.csv")
    
    # 7. Display results in console
    print("\n[7/7] Displaying results:")
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key:30s}: {value}")
    
    print("\n" + "=" * 80)
    print("MODEL COMPLETENESS")
    print("=" * 80)
    print(completeness_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - dataset_files.csv                  : Complete file listing")
    print("  - model_completeness.csv             : Model completeness analysis")
    print("  - dataset_report.txt                 : Detailed text report")
    print("  - distribution_by_lexical_unit.png   : Distribution by tone")
    print("  - distribution_by_model.png          : Distribution by model")
    print("  - heatmap_lexical_vs_model.png       : Distribution matrix")
    print("  - file_size_distribution.png         : File size distribution")
    print("  - model_completeness.png             : Model completeness chart")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()