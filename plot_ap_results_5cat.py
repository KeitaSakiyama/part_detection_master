#!/usr/bin/env python3
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_files(log_dir):
    """Parse log files to extract AP values and corresponding parameters"""
    results = {
        'prompt': {},
        'linear_prob': {},
        'full': {}
    }
    
    category_results = {
        'prompt': {},
        'linear_prob': {},
        'full': {}
    }
    
    log_files = Path(log_dir).glob("combo_*.log")
    
    for log_file in log_files:
        filename = log_file.name
        print(f"Processing {filename}")
        
        # Extract ft_type and max_train from filename
        # Updated pattern to handle filenames without timestamp
        match = re.match(r'combo_(.+?)_(\d+)(?:_\d+_\d+)?\.log', filename)
        if not match:
            print(f"Skipping {filename} - doesn't match expected pattern")
            continue
            
        ft_type = match.group(1)
        max_train = int(match.group(2))
        
        # Read the log file and extract AP values
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Look for the main AP value (IoU=0.50)
                ap_patterns = [
                    # Pattern for IoU=0.50 (found in the logs)
                    r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d\.]+)',
                    # Alternative patterns
                    r"'AP',\s*([\d\.]+)\)",
                    r"'AP':\s*([\d\.]+)"
                ]
                
                ap_value = None
                for pattern in ap_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Take the last match (usually the final validation result)
                        ap_value = float(matches[-1])
                        break
                
                # Look for category-specific AP values after "Average Precision [%]"
                category_ap_pattern = r'Average Precision \[%\]\s*\n((?:---.*?---\s*\n[\d\.]+\s*\n?)*)'
                category_match = re.search(category_ap_pattern, content, re.MULTILINE)
                
                category_aps = {}
                if category_match:
                    category_block = category_match.group(1)
                    # Extract individual category AP values
                    category_pattern = r'---(.+?)---\s*\n([\d\.]+)'
                    category_matches = re.findall(category_pattern, category_block)
                    
                    for cat_name, cat_ap in category_matches:
                        category_aps[cat_name.strip()] = float(cat_ap)
                        print(f"  Found category AP: {cat_name.strip()} = {cat_ap}")
                
                if ap_value is not None and ft_type in results:
                    results[ft_type][max_train] = ap_value
                    print(f"  Found overall AP = {ap_value:.4f} for {ft_type}, MAX_TRAIN = {max_train}")
                        
                if category_aps and ft_type in category_results:
                    category_results[ft_type][max_train] = category_aps
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return results, category_results

def plot_overall_ap(results, output_dir="./plot/5cat"):
    """Plot overall AP results for all fine-tuning types"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(results.values()):
        print("No overall AP data found")
        return
    
    # Available fine-tuning types
    available_ft_types = [ft for ft in ['prompt', 'linear_prob', 'full'] if ft in results and results[ft]]
    
    if not available_ft_types:
        print("No data available for any ft_type")
        return
    
    # Color palette for fine-tuning types
    ft_colors = {
        'prompt': '#1f77b4',      # Blue
        'linear_prob': '#ff7f0e', # Orange
        'full': '#2ca02c'         # Green
    }
    
    # Marker styles for fine-tuning types
    ft_markers = {
        'prompt': 'o',
        'linear_prob': 's',  # Square
        'full': '^'          # Triangle
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate y-axis range
    min_y = float('inf')
    max_y = float('-inf')
    
    # Plot each fine-tuning type
    for ft_type in available_ft_types:
        ft_data = results[ft_type]
        max_trains = []
        ap_values = []
        
        for max_train in sorted(ft_data.keys()):
            max_trains.append(max_train)
            ap_values.append(ft_data[max_train])
            min_y = min(min_y, ft_data[max_train])
            max_y = max(max_y, ft_data[max_train])
        
        if max_trains and ap_values:
            ax.plot(max_trains, ap_values,
                   color=ft_colors[ft_type],
                   marker=ft_markers[ft_type],
                   linewidth=3, markersize=10,
                   label=f'{ft_type.upper()} Fine-tuning')
            
            # Add value labels
            for x, y in zip(max_trains, ap_values):
                ax.annotate(f'{y:.3f}', (x, y),
                           textcoords="offset points",
                           xytext=(0,15), ha='center', fontsize=11,
                           fontweight='bold',
                           color=ft_colors[ft_type])
    
    # Set axis properties
    ax.set_xlabel('MAX_TRAIN_ANNOTATIONS_PER_CATEGORY', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Precision (AP)', fontsize=14, fontweight='bold')
    ax.set_title('Overall AP Comparison\n5 Categories Detection', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis range with margin
    if min_y != float('inf') and max_y != float('-inf'):
        y_margin = (max_y - min_y) * 0.15
        ax.set_ylim(max(0, min_y - y_margin), max_y + y_margin)
    
    # Set x-axis properties
    all_max_trains = set()
    for ft_type in available_ft_types:
        if ft_type in results:
            all_max_trains.update(results[ft_type].keys())
    
    if all_max_trains:
        max_trains_sorted = sorted(all_max_trains)
        ax.set_xlim(0, max(max_trains_sorted) * 1.05)
        ax.set_xticks(max_trains_sorted)
        ax.set_xticklabels(max_trains_sorted)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'overall_ap_5cat.png'
    plt.savefig(os.path.join(output_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved overall AP plot as {filename}")

def plot_individual_categories(category_results, output_dir="./plot/5cat"):
    """Plot individual graphs for each category with all three fine-tuning types"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Define categories in the specified order
    all_categories = [
        "Cable Component",          # ケーブル部材
        "Cable Support Structure",  # ケーブル付属構造物
        "Cable Anchorage",          # ケーブル定着部
        "Cable Rods and Bolts",     # ケーブル関連ロッド、ボルト類
        "Other Cable Accessories"   # その他ケーブル関連部品
    ]
    
    # Filter to only include categories that exist in the data
    existing_categories = set()
    for ft_type_data in category_results.values():
        for max_train_data in ft_type_data.values():
            existing_categories.update(max_train_data.keys())
    
    # Keep only categories that exist in the data, in the specified order
    all_categories = [cat for cat in all_categories if cat in existing_categories]
    
    if not all_categories:
        print("No categories found")
        return
    
    # Available fine-tuning types
    available_ft_types = [ft for ft in ['prompt', 'linear_prob', 'full'] if ft in category_results and category_results[ft]]
    
    if not available_ft_types:
        print("No data available for any ft_type")
        return
    
    # Color palette for fine-tuning types
    ft_colors = {
        'prompt': '#1f77b4',      # Blue
        'linear_prob': '#ff7f0e', # Orange
        'full': '#2ca02c'         # Green
    }
    
    # Marker styles for fine-tuning types
    ft_markers = {
        'prompt': 'o',
        'linear_prob': 's',  # Square
        'full': '^'          # Triangle
    }
    
    # Create individual plots for each category
    for category in all_categories:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate y-axis range for this category
        min_y = float('inf')
        max_y = float('-inf')
        
        # Plot each fine-tuning type
        for ft_type in available_ft_types:
            ft_data = category_results[ft_type]
            max_trains = []
            ap_values = []
            
            for max_train in sorted(ft_data.keys()):
                if category in ft_data[max_train]:
                    max_trains.append(max_train)
                    ap_values.append(ft_data[max_train][category])
                    min_y = min(min_y, ft_data[max_train][category])
                    max_y = max(max_y, ft_data[max_train][category])
            
            if max_trains and ap_values:
                ax.plot(max_trains, ap_values,
                       color=ft_colors[ft_type],
                       marker=ft_markers[ft_type],
                       linewidth=3, markersize=10,
                       label=f'{ft_type.upper()} Fine-tuning')
                
                # Add value labels
                for x, y in zip(max_trains, ap_values):
                    ax.annotate(f'{y:.1f}', (x, y),
                               textcoords="offset points",
                               xytext=(0,15), ha='center', fontsize=11,
                               fontweight='bold',
                               color=ft_colors[ft_type])
        
        # Set axis properties
        ax.set_xlabel('MAX_TRAIN_ANNOTATIONS_PER_CATEGORY', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Precision (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{category}\nComparison of Fine-tuning Methods', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis range with margin
        if min_y != float('inf') and max_y != float('-inf'):
            y_margin = (max_y - min_y) * 0.15
            ax.set_ylim(max(0, min_y - y_margin), max_y + y_margin)
        
        # Set x-axis properties
        all_max_trains = set()
        for ft_type in available_ft_types:
            if ft_type in category_results:
                all_max_trains.update(category_results[ft_type].keys())
        
        if all_max_trains:
            max_trains_sorted = sorted(all_max_trains)
            ax.set_xlim(0, max(max_trains_sorted) * 1.05)
            ax.set_xticks(max_trains_sorted)
            ax.set_xticklabels(max_trains_sorted)
        
        plt.tight_layout()
        
        # Save the plot with category name in filename
        safe_category_name = category.replace(" ", "_").replace(",", "")
        filename = f'individual_category_{safe_category_name}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot for {category} as {filename}")

def plot_all_categories_combined(category_results, output_dir="./plot/5cat"):
    """Plot all 5 categories in one figure with subplots, each showing all three fine-tuning types"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Define categories in the specified order
    all_categories = [
        "Cable Component",          # ケーブル部材
        "Cable Support Structure",  # ケーブル付属構造物
        "Cable Anchorage",          # ケーブル定着部
        "Cable Rods and Bolts",     # ケーブル関連ロッド、ボルト類
        "Other Cable Accessories"   # その他ケーブル関連部品
    ]
    
    # Filter to only include categories that exist in the data
    existing_categories = set()
    for ft_type_data in category_results.values():
        for max_train_data in ft_type_data.values():
            existing_categories.update(max_train_data.keys())
    
    # Keep only categories that exist in the data, in the specified order
    all_categories = [cat for cat in all_categories if cat in existing_categories]
    
    if not all_categories:
        print("No categories found")
        return
    
    # Available fine-tuning types
    available_ft_types = [ft for ft in ['prompt', 'linear_prob', 'full'] if ft in category_results and category_results[ft]]
    
    if not available_ft_types:
        print("No data available for any ft_type")
        return
    
    # Color palette for fine-tuning types
    ft_colors = {
        'prompt': '#1f77b4',      # Blue
        'linear_prob': '#ff7f0e', # Orange
        'full': '#2ca02c'         # Green
    }
    
    # Marker styles for fine-tuning types
    ft_markers = {
        'prompt': 'o',
        'linear_prob': 's',  # Square
        'full': '^'          # Triangle
    }
    
    # Create figure with subplots (2x3 layout for 5 categories)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Calculate global y-axis range for consistency
    global_min_y = float('inf')
    global_max_y = float('-inf')
    
    for category in all_categories:
        for ft_type in available_ft_types:
            ft_data = category_results[ft_type]
            for max_train in ft_data.keys():
                if category in ft_data[max_train]:
                    ap_value = ft_data[max_train][category]
                    global_min_y = min(global_min_y, ap_value)
                    global_max_y = max(global_max_y, ap_value)
    
    # Add margin to y-axis range
    if global_min_y != float('inf') and global_max_y != float('-inf'):
        y_margin = (global_max_y - global_min_y) * 0.15
        global_min_y = max(0, global_min_y - y_margin)
        global_max_y = global_max_y + y_margin
    
    # Get all MAX_TRAIN values for consistent x-axis
    all_max_trains = set()
    for ft_type in available_ft_types:
        if ft_type in category_results:
            all_max_trains.update(category_results[ft_type].keys())
    max_trains_sorted = sorted(all_max_trains) if all_max_trains else []
    
    # Plot each category in a subplot
    for idx, category in enumerate(all_categories):
        ax = axes[idx]
        
        # Plot each fine-tuning type
        for ft_type in available_ft_types:
            ft_data = category_results[ft_type]
            max_trains = []
            ap_values = []
            
            for max_train in sorted(ft_data.keys()):
                if category in ft_data[max_train]:
                    max_trains.append(max_train)
                    ap_values.append(ft_data[max_train][category])
            
            if max_trains and ap_values:
                ax.plot(max_trains, ap_values,
                       color=ft_colors[ft_type],
                       marker=ft_markers[ft_type],
                       linewidth=2.5, markersize=8,
                       label=f'{ft_type.upper()}')
                
                # Add value labels
                for x, y in zip(max_trains, ap_values):
                    ax.annotate(f'{y:.1f}', (x, y),
                               textcoords="offset points",
                               xytext=(0,10), ha='center', fontsize=9,
                               fontweight='bold',
                               color=ft_colors[ft_type])
        
        # Set axis properties
        ax.set_xlabel('MAX_TRAIN_ANNOTATIONS', fontsize=11, fontweight='bold')
        ax.set_ylabel('AP (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{category}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis range
        if global_min_y != float('inf') and global_max_y != float('-inf'):
            ax.set_ylim(global_min_y, global_max_y)
        
        # Set x-axis properties
        if max_trains_sorted:
            ax.set_xlim(0, max(max_trains_sorted) * 1.05)
            ax.set_xticks(max_trains_sorted)
            ax.set_xticklabels(max_trains_sorted)
        
        # Make tick labels smaller
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Hide the unused subplot (6th position)
    if len(all_categories) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout(pad=3.0)
    
    # Save the combined plot
    filename = 'all_categories_combined_subplot.png'
    plt.savefig(os.path.join(output_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved combined plot for all categories as {filename}")

def plot_combined_5categories(category_results, output_dir="./plot/5cat"):
    """Plot 5 categories combined in one figure for prompt, linear_prob, and full"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Define categories in the specified order
    all_categories = [
        "Cable Component",          # ケーブル部材
        "Cable Support Structure",  # ケーブル付属構造物
        "Cable Anchorage",          # ケーブル定着部
        "Cable Rods and Bolts",     # ケーブル関連ロッド、ボルト類
        "Other Cable Accessories"   # その他ケーブル関連部品
    ]
    
    # Filter to only include categories that exist in the data
    existing_categories = set()
    for ft_type_data in category_results.values():
        for max_train_data in ft_type_data.values():
            existing_categories.update(max_train_data.keys())
    
    # Keep only categories that exist in the data, in the specified order
    all_categories = [cat for cat in all_categories if cat in existing_categories]
    
    if not all_categories:
        print("No categories found")
        return
    
    # Create combined plot for all ft_types (up to 3 subplots)
    available_ft_types = [ft for ft in ['prompt', 'linear_prob', 'full'] if ft in category_results and category_results[ft]]
    n_plots = len(available_ft_types)
    
    if n_plots == 0:
        print("No data available for any ft_type")
        return
    
    # Calculate global y-axis range for all subplots
    global_min_y = float('inf')
    global_max_y = float('-inf')
    
    for ft_type in available_ft_types:
        ft_data = category_results[ft_type]
        for category in all_categories:
            for max_train in ft_data.keys():
                if category in ft_data[max_train]:
                    ap_value = ft_data[max_train][category]
                    global_min_y = min(global_min_y, ap_value)
                    global_max_y = max(global_max_y, ap_value)
    
    # Add some margin to the y-axis range
    y_margin = (global_max_y - global_min_y) * 0.1
    global_min_y = max(0, global_min_y - y_margin)  # Don't go below 0
    global_max_y = global_max_y + y_margin
    
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 8))
    if n_plots == 1:
        axes = [axes]
    
    # Color palette for categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, ft_type in enumerate(available_ft_types):
        ax = axes[idx]
        ft_data = category_results[ft_type]
        
        # For each category, plot the line
        for cat_idx, category in enumerate(all_categories):
            max_trains = []
            ap_values = []
            
            for max_train in sorted(ft_data.keys()):
                if category in ft_data[max_train]:
                    max_trains.append(max_train)
                    ap_values.append(ft_data[max_train][category])
            
            if max_trains and ap_values:
                ax.plot(max_trains, ap_values,
                       color=colors[cat_idx % len(colors)],
                       marker='o',
                       linewidth=3, markersize=8,
                       label=f'{category}')
                
                # Add value labels
                for x, y in zip(max_trains, ap_values):
                    ax.annotate(f'{y:.1f}', (x, y),
                               textcoords="offset points",
                               xytext=(0,10), ha='center', fontsize=10)
        
        ax.set_xlabel('MAX_TRAIN_ANNOTATIONS_PER_CATEGORY', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Precision (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{ft_type.upper()} Fine-tuning\n(5 Categories Combined)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set unified y-axis range for all subplots
        ax.set_ylim(global_min_y, global_max_y)
        
        # Set x-axis to start from 0 with equal intervals
        if ft_data:
            all_max_trains = set()
            for max_train_data in ft_data.values():
                all_max_trains.update([k for k in ft_data.keys()])
            
            if all_max_trains:
                max_trains_sorted = sorted(all_max_trains)
                # Set x-axis limits and ticks
                ax.set_xlim(0, max(max_trains_sorted) * 1.05)  # Start from 0, add 5% margin
                ax.set_xticks(max_trains_sorted)
                ax.set_xticklabels(max_trains_sorted)
        
        # Calculate and display mAP if data exists
        if ft_data:
            # Get the latest (highest MAX_TRAIN) results
            latest_max_train = max(ft_data.keys())
            latest_aps = []
            for category in all_categories:
                if category in ft_data[latest_max_train]:
                    latest_aps.append(ft_data[latest_max_train][category])
            
            if latest_aps:
                mAP = np.mean(latest_aps)
                ax.text(0.02, 0.98, f'mAP = {mAP:.1f}%', 
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'combined_5categories_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Set up paths
    log_dir = "./logs/5cat"
    
    # Create output directory if it doesn't exist
    os.makedirs("./plot/5cat", exist_ok=True)
    
    if os.path.exists(log_dir):
        print("Parsing training log files...")
        results, category_results = parse_log_files(log_dir)
        
        # Debug information
        print(f"Debug: results = {results}")
        print(f"Debug: any(results.values()) = {any(results.values())}")
        for ft_type, data in results.items():
            print(f"Debug: {ft_type} has {len(data)} entries: {data}")
        
        print("Saving plots to ./plot/5cat...")

        if any(results.values()):
            print("Creating overall AP plot...")
            plot_overall_ap(results)
        else:
            print("No overall AP data found - skipping overall plot")
        
        if any(category_results.values()):
            print("Creating all categories combined plot...")
            plot_all_categories_combined(category_results)
            print("\nCreating individual category plots...")
            plot_individual_categories(category_results)
            print("\nCreating combined 5-category plots...")
            plot_combined_5categories(category_results)
        else:
            print("No category AP data found - skipping category plots")

    else:
        print(f"Log directory {log_dir} not found!")
    
    print("Done!")