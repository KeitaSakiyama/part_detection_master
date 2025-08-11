#!/bin/bash

# Script to run training and validation for different configurations (all item version)
# MAX_TRAIN_ANNOTATIONS_PER_CATEGORY: 10, 50, 100, 150, 200, 300, 400
# ft_type: prompt, linear_prob

# Remove set -e to allow script to continue even if training fails
# set -e  # Exit on any error

# Define arrays for the parameters
MAX_TRAIN_VALUES=(10 20 30 40 50 100 150 200)
FT_TYPES=("full" "prompt" "linear_prob")

# Log file
LOG_DIR="./logs/all_item"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/training_validation.log"

echo "Starting training and validation process (all item version) at $(date)" | tee "$MAIN_LOG"
echo "Will run for MAX_TRAIN_ANNOTATIONS_PER_CATEGORY values: ${MAX_TRAIN_VALUES[*]}" | tee -a "$MAIN_LOG"
echo "Will run for ft_type values: ${FT_TYPES[*]}" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Function to update Python script parameter
update_python_param() {
    local script_path="$1"
    local param_name="$2"
    local param_value="$3"
    
    # Create a temporary backup
    cp "$script_path" "$script_path.bak"
    
    # Update the parameter in the Python script
    if [ "$param_name" = "MAX_TRAIN_ANNOTATIONS_PER_CATEGORY" ]; then
        sed -i "s/^MAX_TRAIN_ANNOTATIONS_PER_CATEGORY = [0-9]*/MAX_TRAIN_ANNOTATIONS_PER_CATEGORY = $param_value/" "$script_path"
    elif [ "$param_name" = "ft_type" ]; then
        # Comment out other ft_type lines and uncomment the target one
        sed -i 's/^ft_type = /# ft_type = /' "$script_path"
        sed -i "s/^# ft_type = \"$param_value\"/ft_type = \"$param_value\"/" "$script_path"
        # If the line doesn't exist, add it
        if ! grep -q "^ft_type = \"$param_value\"" "$script_path"; then
            sed -i "s/^# ft_type = \".*\"/ft_type = \"$param_value\"/" "$script_path"
        fi
    fi
}

# Function to restore Python script from backup
restore_python_script() {
    local script_path="$1"
    if [ -f "$script_path.bak" ]; then
        mv "$script_path.bak" "$script_path"
    fi
}

# Function to run training
run_training() {
    local max_train="$1"
    local ft_type="$2"
    local log_file="$3"
    
    echo "Running training: MAX_TRAIN=$max_train, ft_type=$ft_type" | tee -a "$log_file"
    
    # Update parameters in training script
    update_python_param "1st_finetuning_20250411共有_all_item_train_limited.py" "MAX_TRAIN_ANNOTATIONS_PER_CATEGORY" "$max_train"
    update_python_param "1st_finetuning_20250411共有_all_item_train_limited.py" "ft_type" "$ft_type"
    
    # Run training
    echo "  Starting training at $(date)..." | tee -a "$log_file"
    if python 1st_finetuning_20250411共有_all_item_train_limited.py >> "$log_file" 2>&1; then
        echo "  Training completed successfully at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "  Training failed at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Function to run validation
run_validation() {
    local max_train="$1"
    local ft_type="$2"
    local log_file="$3"
    
    echo "Running validation: MAX_TRAIN=$max_train, ft_type=$ft_type" | tee -a "$log_file"
    
    # Update parameters in validation script
    update_python_param "2nd_validation_20250411共有_all_item_train_limited.py" "MAX_TRAIN_ANNOTATIONS_PER_CATEGORY" "$max_train"
    update_python_param "2nd_validation_20250411共有_all_item_train_limited.py" "ft_type" "$ft_type"
    
    # Run validation
    echo "  Starting validation at $(date)..." | tee -a "$log_file"
    if python 2nd_validation_20250411共有_all_item_train_limited.py >> "$log_file" 2>&1; then
        echo "  Validation completed successfully at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "  Validation failed at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Main execution loop
total_combinations=$((${#MAX_TRAIN_VALUES[@]} * ${#FT_TYPES[@]}))
current_combination=0
training_successful=0
training_failed=0
validation_successful=0
validation_failed=0

for ft_type in "${FT_TYPES[@]}"; do
    for max_train in "${MAX_TRAIN_VALUES[@]}"; do
        current_combination=$((current_combination + 1))
        
        echo "" | tee -a "$MAIN_LOG"
        echo "========================================" | tee -a "$MAIN_LOG"
        echo "Processing combination $current_combination/$total_combinations" | tee -a "$MAIN_LOG"
        echo "MAX_TRAIN_ANNOTATIONS_PER_CATEGORY: $max_train" | tee -a "$MAIN_LOG"
        echo "ft_type: $ft_type" | tee -a "$MAIN_LOG"
        echo "========================================" | tee -a "$MAIN_LOG"
        
        # Create specific log file for this combination
        COMBO_LOG="$LOG_DIR/combo_${ft_type}_${max_train}.log"
        
        # Run training (but don't stop if it fails)
        if run_training "$max_train" "$ft_type" "$COMBO_LOG"; then
            echo "✓ Training for combination $current_combination/$total_combinations completed successfully" | tee -a "$MAIN_LOG"
            training_successful=$((training_successful + 1))
        else
            echo "✗ Training failed for combination $current_combination/$total_combinations" | tee -a "$MAIN_LOG"
            echo "  Continuing with validation anyway..." | tee -a "$MAIN_LOG"
            training_failed=$((training_failed + 1))
        fi
        
        # Always run validation regardless of training result
        if run_validation "$max_train" "$ft_type" "$COMBO_LOG"; then
            echo "✓ Validation for combination $current_combination/$total_combinations completed successfully" | tee -a "$MAIN_LOG"
            validation_successful=$((validation_successful + 1))
        else
            echo "✗ Validation failed for combination $current_combination/$total_combinations" | tee -a "$MAIN_LOG"
            validation_failed=$((validation_failed + 1))
        fi
        
        # Clean up GPU memory
        echo "  Cleaning up GPU memory..." | tee -a "$MAIN_LOG"
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
        
        echo "  Combination processing time: $(date)" | tee -a "$MAIN_LOG"
    done
done

# Restore original scripts
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Restoring original script configurations..." | tee -a "$MAIN_LOG"
restore_python_script "1st_finetuning_20250411共有_all_item_train_limited.py"
restore_python_script "2nd_validation_20250411共有_all_item_train_limited.py"

# Final summary
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "All training and validation tasks completed at $(date)" | tee -a "$MAIN_LOG"
echo "Total combinations processed: $total_combinations" | tee -a "$MAIN_LOG"
echo "Training results: $training_successful successful, $training_failed failed" | tee -a "$MAIN_LOG"
echo "Validation results: $validation_successful successful, $validation_failed failed" | tee -a "$MAIN_LOG"
echo "Main log file: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "Individual combination logs are in: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

echo "Process completed. Check the log files for detailed results."
echo "Training: $training_successful successful, $training_failed failed"
echo "Validation: $validation_successful successful, $validation_failed failed"