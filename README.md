# Bridge Inspection Part Detection
[日本語版READMEはこちら](README_ja.md)

This project contains machine learning models and scripts for detecting and classifying bridge components in inspection images.

## Project Overview

This repository includes:
- Fine-tuning scripts for object detection models
- Validation and evaluation scripts
- Visualization tools for detection results
- Training and validation automation scripts
- Automatic monitoring and execution system

## Key Components

### Training Scripts
- `1st_finetuning_*.py` - Fine-tuning scripts with various configurations
- Training supports different fine-tuning types: prompt, linear_prob, full

### Validation Scripts
- `2nd_validation_*.py` - Model validation and evaluation scripts
- Generates performance metrics and detection results

### Visualization Tools
- `plot_ap_results_*.py` - Scripts for plotting Average Precision (AP) results
- `visualize*.py` - Tools for visualizing detection outputs
- `show_output*.py` - Scripts for displaying model outputs

### Automation Scripts
- `run_*.sh` - Shell scripts for automated training and validation workflows
- `auto_run_after_completion.sh` - Automated execution after training completion
- `auto_git_pull_monitor.sh` - Automatic git repository monitoring and execution system
- `auto_run.sh` - Execution script triggered by the monitoring system

## Automatic Monitoring System

### Auto Git Pull Monitor

The `auto_git_pull_monitor.sh` script provides automatic monitoring of the git repository and execution of training scripts when changes are detected.

#### Features
- **Continuous Monitoring**: Checks for git repository changes every 5 minutes
- **Automatic Git Pull**: Pulls latest changes from the master branch
- **Smart Execution**: Executes specified scripts only when changes are detected
- **Process Safety**: Skips execution if training processes are already running
- **Comprehensive Logging**: Logs all activities with timestamps

#### Usage
```bash
# Start monitoring with default script
./auto_git_pull_monitor.sh

# Start monitoring with specific script
./auto_git_pull_monitor.sh run_training_validation_5cat.sh

# Available scripts for execution:
# - run_training_validation_all_item.sh
# - run_training_validation_5cat.sh
# - run_training_validation_without_cable_components.sh
# - run_all_item_training_validation_full_data.sh
```

#### Configuration
- **Check Interval**: 5 minutes (300 seconds)
- **Log Files**: 
  - `./logs/git_auto_pull.log` - Main monitoring log
  - `./logs/git_changes.log` - Git changes log

#### How It Works
1. **Repository Monitoring**: Continuously monitors git repository for changes
2. **Change Detection**: Compares commit hashes before and after git pull
3. **Smart Execution**: Only executes scripts when actual changes are detected
4. **Process Management**: Prevents conflicts by checking for running training processes
5. **Automatic Recovery**: Continues monitoring even after script execution

#### Process Safety
- Checks for running training processes (`1st_finetuning*.py`, `2nd_validation*.py`, `run_*.sh`)
- Skips git pull and execution if training is already in progress
- Waits 30 seconds after detecting changes before execution
- Double-checks for new training processes before final execution

## Categories Detected

The model can detect 5 main categories of bridge components:
1. Cable Component (ケーブル部材)
2. Cable Support Structure (ケーブル付属構造物)
3. Cable Anchorage (ケーブル定着部)
4. Cable Rods and Bolts (ケーブル関連ロッド、ボルト類)
5. Other Cable Accessories (その他ケーブル関連部品)

## Directory Structure

- `1st_OUTPUT/` - Training outputs
- `2nd_RESULT/` - Validation results
- `evaluation_results/` - Evaluation metrics
- `GLIP/` - GLIP model related files
- `logs/` - Training and validation logs
- `plot/` - Generated plots and visualizations
- `visualized_output_images*/` - Visualization outputs

## Usage

### Training
Run training scripts using the provided shell scripts:
```bash
./run_training_validation_5cat.sh
./run_training_validation_all_item.sh
```

### Evaluation
Evaluate models using validation scripts:
```bash
python 2nd_validation_*.py
```

### Visualization
Generate plots and visualizations:
```bash
python plot_ap_results_5cat.py
python visualize.py
```

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Matplotlib
- NumPy
- Other dependencies as specified in individual scripts

## Notes

This project is designed for bridge inspection automation and focuses on cable-related component detection.

## Workflow Integration

### Automatic Training Pipeline
1. **Code Updates**: Push changes to the git repository
2. **Auto Detection**: The monitor detects changes within 5 minutes
3. **Auto Execution**: Specified training/validation scripts are executed automatically
4. **Result Generation**: Training outputs and validation results are generated
5. **Continuous Monitoring**: System continues monitoring for next changes

### Manual Override
Even with automatic monitoring active, you can still run scripts manually:
```bash
./run_training_validation_5cat.sh
./run_training_validation_all_item.sh
```