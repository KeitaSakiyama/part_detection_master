#!/usr/bin/env python3
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define cable component groups
cable_component_parts = [
    "Main Cable", "Main Cable (Multiple Cables)", "Hanger Rope", "Stays", "Stay Support",
    "Handrail Rope", "Parallel Wire Strand", "Strand Rope", "Spiral Rope", "Locked Coil Rope",
    "PC Steel Wire", "Semi-Parallel Wire Strand", "Zinc-Plated Steel Wire",
]
cable_support_parts = [
    "Cable Band",
    "Saddle Cover", "Cable Cover", "Socket Cover", "Anchor Cover", "Tower Saddle",
    "Spray Saddle", "Intersection Fitting", "Stay Grip Fitting", "Saddle Anchor End",
    "Band Anchor End", "Vibration Damper",
]
cable_anchorage_parts = [
    "Socket (Open Type)", 
    "Socket (Rod Anchor Type)", "Socket (Pressure Anchor Type)",
    "End Clamp (SG Socket)", "End Clamp (Screw Type)", "Crimp Anchor", "Wire Clip Anchor",
    "U-bolt Anchor", "Embedded Anchor", "Cable Anchor",
    "Socket Anchor", "Rod Anchor",
]
cable_rods_and_bolts_parts = [
    "Anchor Rod",
    "Rod Thread Part", "Rod Anchor Nut", "Fixing Bolt",
]
other_cable_accessories_parts = [
    "Anchor Piece", "Shackle",
    "Shinble Connection", "Turnbuckle", "Wire Clip", "Connecting Fitting", "Pressure Plate", "Wire Seeging",
    "Rubber Boot", "Stainless Band", "Grout Injection Port", "Water Stop Material"
]

# Create a mapping from category to group
category_to_group = {}
for cat in cable_component_parts:
    category_to_group[cat] = "Cable Components"
for cat in cable_support_parts:
    category_to_group[cat] = "Cable Support"
for cat in cable_anchorage_parts:
    category_to_group[cat] = "Cable Anchorage"
for cat in cable_rods_and_bolts_parts:
    category_to_group[cat] = "Cable Rods & Bolts"
for cat in other_cable_accessories_parts:
    category_to_group[cat] = "Other Cable Accessories"

# Define the desired order for all categories
all_categories_ordered = (cable_component_parts + cable_support_parts + 
                         cable_anchorage_parts + cable_rods_and_bolts_parts + 
                         other_cable_accessories_parts)

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

def plot_overall_ap(results, output_dir="./plot/all_item"):
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
    ax.set_title('Overall AP Comparison\nAll Items Detection', 
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
    filename = 'overall_ap_all_item.png'
    plt.savefig(os.path.join(output_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved overall AP plot as {filename}")

def plot_individual_categories(category_results, output_dir="./plot/all_item"):
    """Plot individual graphs for each category with all three fine-tuning types"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Get all unique categories from the data
    all_categories_from_data = set()
    for ft_type_data in category_results.values():
        for max_train_data in ft_type_data.values():
            all_categories_from_data.update(max_train_data.keys())
    
    # Use the predefined order, but only include categories that exist in the data
    all_categories = [cat for cat in all_categories_ordered if cat in all_categories_from_data]
    
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
        ax.set_title(f'{category}\nComparison of Fine-tuning Methods (All Items)', 
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
        safe_category_name = category.replace(" ", "_").replace(",", "").replace("/", "_")
        filename = f'individual_category_all_item_{safe_category_name}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot for {category} as {filename}")

def plot_all_categories_combined(category_results, output_dir="./plot/all_item"):
    """Plot all categories in one figure with subplots, each showing all three fine-tuning types"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Get all unique categories from the data
    all_categories_from_data = set()
    for ft_type_data in category_results.values():
        for max_train_data in ft_type_data.values():
            all_categories_from_data.update(max_train_data.keys())
    
    # Use the predefined order, but only include categories that exist in the data
    all_categories = [cat for cat in all_categories_ordered if cat in all_categories_from_data]
    
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
    
    # Calculate subplot layout
    n_categories = len(all_categories)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
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
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
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
        ax.set_title(f'{category}\nComparison of Fine-tuning Methods (All Items)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis range with margin
        if global_min_y != float('inf') and global_max_y != float('-inf'):
            ax.set_ylim(global_min_y, global_max_y)
        
        # Set x-axis properties
        if max_trains_sorted:
            ax.set_xlim(0, max(max_trains_sorted) * 1.05)
            ax.set_xticks(max_trains_sorted)
            ax.set_xticklabels(max_trains_sorted)
    
    # Hide unused subplots
    for idx in range(n_categories, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout(pad=3.0)
    
    # Save the combined plot
    filename = 'all_categories_combined_all_item.png'
    plt.savefig(os.path.join(output_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved combined plot for all categories as {filename}")

def plot_combined_categories(category_results, output_dir="./plot/all_item"):
    """Plot all categories combined in one figure for prompt, linear_prob, and full"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Get all unique categories from the data
    all_categories_from_data = set()
    for ft_type_data in category_results.values():
        for max_train_data in ft_type_data.values():
            all_categories_from_data.update(max_train_data.keys())
    
    # Use the predefined order, but only include categories that exist in the data
    all_categories = [cat for cat in all_categories_ordered if cat in all_categories_from_data]
    
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
    
    # Color palette for categories (expanded for more categories)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
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
        ax.set_title(f'{ft_type.upper()} Fine-tuning\n(All Items Combined)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
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
    plt.savefig(os.path.join(output_dir, 'combined_categories_all_item.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_grouped_categories(category_results, output_dir="./plot/all_item"):
    """Plot categories grouped by cable component types in 5 separate subplots - FULL fine-tuning only"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not any(category_results.values()):
        print("No category-specific AP data found")
        return
    
    # Check if 'full' fine-tuning data exists
    if 'full' not in category_results or not category_results['full']:
        print("No data available for full fine-tuning")
        return
    
    # Get all unique categories from the full fine-tuning data only
    all_categories_from_data = set()
    for max_train_data in category_results['full'].values():
        all_categories_from_data.update(max_train_data.keys())
    
    # Group categories by their type using predefined order
    grouped_categories = {
        "Cable Components": [cat for cat in cable_component_parts if cat in all_categories_from_data],
        "Cable Support": [cat for cat in cable_support_parts if cat in all_categories_from_data],
        "Cable Anchorage": [cat for cat in cable_anchorage_parts if cat in all_categories_from_data],
        "Cable Rods & Bolts": [cat for cat in cable_rods_and_bolts_parts if cat in all_categories_from_data],
        "Other Cable Accessories": [cat for cat in other_cable_accessories_parts if cat in all_categories_from_data]
    }
    
    # Add any categories not in predefined groups
    categorized_items = set()
    for group_cats in grouped_categories.values():
        categorized_items.update(group_cats)
    
    uncategorized = all_categories_from_data - categorized_items
    if uncategorized:
        if "Other" not in grouped_categories:
            grouped_categories["Other"] = []
        grouped_categories["Other"].extend(sorted(uncategorized))
    
    # Remove empty groups
    grouped_categories = {k: v for k, v in grouped_categories.items() if v}
    
    # Use only full fine-tuning data
    ft_data = category_results['full']
    
    # Create subplot layout for groups
    n_groups = len(grouped_categories)
    n_cols = 3 if n_groups > 3 else n_groups
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    # Create figure with subplots for each group
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_groups == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Calculate global y-axis range for consistency (only from full fine-tuning data)
    global_min_y = float('inf')
    global_max_y = float('-inf')
    
    for group_name, categories in grouped_categories.items():
        for category in categories:
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
    max_trains_sorted = sorted(ft_data.keys()) if ft_data else []
    
    # Plot each group in separate subplots
    for group_idx, (group_name, categories) in enumerate(grouped_categories.items()):
        row = group_idx // n_cols
        col = group_idx % n_cols
        ax = axes[row, col] if n_groups > 1 else axes[0]
        
        # Color palette for categories within each group
        category_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                          '#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        # Plot each category in this group (only full fine-tuning)
        for cat_idx, category in enumerate(categories):
            max_trains = []
            ap_values = []
            
            for max_train in sorted(ft_data.keys()):
                if category in ft_data[max_train]:
                    max_trains.append(max_train)
                    ap_values.append(ft_data[max_train][category])
            
            if max_trains and ap_values:
                # Use different colors for categories
                color = category_colors[cat_idx % len(category_colors)]
                
                ax.plot(max_trains, ap_values,
                       color=color,
                       marker='o',
                       linewidth=2.5, markersize=8,
                       label=f'{category}',
                       alpha=0.8)
                
                # Add value labels
                for x, y in zip(max_trains, ap_values):
                    ax.annotate(f'{y:.1f}', (x, y),
                               textcoords="offset points",
                               xytext=(0,10), ha='center', fontsize=9,
                               color=color)
        
        # Set axis properties for this subplot
        ax.set_xlabel('MAX_TRAIN_ANNOTATIONS_PER_CATEGORY', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Precision (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{group_name}', fontsize=14, fontweight='bold')
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
        
        # Calculate and display mAP for this group (full fine-tuning only)
        if ft_data:
            latest_max_train = max(ft_data.keys())
            group_aps = []
            for category in categories:
                if category in ft_data[latest_max_train]:
                    group_aps.append(ft_data[latest_max_train][category])
            
            if group_aps:
                group_mAP = np.mean(group_aps)
                ax.text(0.02, 0.98, f'mAP: {group_mAP:.1f}%', 
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                       verticalalignment='top')
    
    # Hide unused subplots
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_groups > 1:
            axes[row, col].set_visible(False)
    
    plt.suptitle('Cable Component Groups Analysis\nFULL Fine-tuning Results Only', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    filename = 'grouped_categories_full_only_5subplots_all_item.png'
    plt.savefig(os.path.join(output_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved grouped plot with 5 subplots (FULL fine-tuning only) as {filename}")

if __name__ == "__main__":
    # Set up paths
    log_dir = "./logs/all_item"
    
    if os.path.exists(log_dir):
        print("Parsing training log files...")
        results, category_results = parse_log_files(log_dir)
        
        print("Saving plots to ./plot/all_item...")

        if any(results.values()):
            print("Creating overall AP plot...")
            plot_overall_ap(results)
        
        if any(category_results.values()):
            print("Creating grouped categories plot...")
            plot_grouped_categories(category_results)
            print("\nCreating all categories combined plot...")
            plot_all_categories_combined(category_results)
            print("\nCreating individual category plots...")
            plot_individual_categories(category_results)
            print("\nCreating combined category plots...")
            plot_combined_categories(category_results)
    else:
        print(f"Log directory {log_dir} not found!")
    
    print("Done!")