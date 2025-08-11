#!/bin/bash

# Script to run training and validation for 20250411共有_all_item scripts
# ft_type: prompt, linear_prob

# Remove set -e to allow script to continue even if training fails
# set -e  # Exit on any error

# Define array for the ft_type parameter
FT_TYPES=("full" "prompt" "linear_prob")

# Log file
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/all_item_training_validation.log"

echo "Starting 20250411共有_all_item training and validation process at $(date)" | tee "$MAIN_LOG"
echo "Will run for ft_type values: ${FT_TYPES[*]}" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Function to update Python script parameter
update_python_param() {
    local script_path="$1"
    local param_name="$2"
    local param_value="$3"
    
    # Create a temporary backup
    cp "$script_path" "$script_path.bak"
    
    # Update the ft_type parameter in the Python script
    if [ "$param_name" = "ft_type" ]; then
        # Comment out all ft_type lines first
        sed -i 's/^ft_type = /# ft_type = /' "$script_path"
        # Find and uncomment the target ft_type line
        if grep -q "^# ft_type = \"$param_value\"" "$script_path"; then
            sed -i "s/^# ft_type = \"$param_value\"/ft_type = \"$param_value\"/" "$script_path"
        else
            # If the specific line doesn't exist, replace the first commented ft_type line
            sed -i "0,/^# ft_type = /{s/^# ft_type = .*/ft_type = \"$param_value\"/}" "$script_path"
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
    local ft_type="$1"
    local log_file="$2"
    
    echo "Running training: ft_type=$ft_type" | tee -a "$log_file"
    
    # Update parameter in training script
    update_python_param "1st_finetuning_20250411共有_all_item.py" "ft_type" "$ft_type"
    
    # Run training
    echo "  Starting training at $(date)..." | tee -a "$log_file"
    if python 1st_finetuning_20250411共有_all_item.py >> "$log_file" 2>&1; then
        echo "  Training completed successfully at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "  Training failed at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Function to run validation
run_validation() {
    local ft_type="$1"
    local log_file="$2"
    
    echo "Running validation: ft_type=$ft_type" | tee -a "$log_file"
    
    # Update parameter in validation script
    update_python_param "2nd_validation_20250411共有_all_item.py" "ft_type" "$ft_type"
    
    # Run validation
    echo "  Starting validation at $(date)..." | tee -a "$log_file"
    if python 2nd_validation_20250411共有_all_item.py >> "$log_file" 2>&1; then
        echo "  Validation completed successfully at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "  Validation failed at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Main execution loop
total_combinations=${#FT_TYPES[@]}
current_combination=0
training_successful=0
training_failed=0
validation_successful=0
validation_failed=0

for ft_type in "${FT_TYPES[@]}"; do
    current_combination=$((current_combination + 1))
    
    echo "" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    echo "Processing combination $current_combination/$total_combinations" | tee -a "$MAIN_LOG"
    echo "ft_type: $ft_type" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    
    # Create specific log file for this combination
    COMBO_LOG="$LOG_DIR/all_item_${ft_type}.log"
    
    # Run training (but don't stop if it fails)
    if run_training "$ft_type" "$COMBO_LOG"; then
        echo "✓ Training for combination $current_combination/$total_combinations (ft_type: $ft_type) completed successfully" | tee -a "$MAIN_LOG"
        training_successful=$((training_successful + 1))
    else
        echo "✗ Training failed for combination $current_combination/$total_combinations (ft_type: $ft_type)" | tee -a "$MAIN_LOG"
        echo "  Continuing with validation anyway..." | tee -a "$MAIN_LOG"
        training_failed=$((training_failed + 1))
    fi
    
    # Always run validation regardless of training result
    if run_validation "$ft_type" "$COMBO_LOG"; then
        echo "✓ Validation for combination $current_combination/$total_combinations (ft_type: $ft_type) completed successfully" | tee -a "$MAIN_LOG"
        validation_successful=$((validation_successful + 1))
    else
        echo "✗ Validation failed for combination $current_combination/$total_combinations (ft_type: $ft_type)" | tee -a "$MAIN_LOG"
        validation_failed=$((validation_failed + 1))
    fi
    
    # Clean up GPU memory
    echo "  Cleaning up GPU memory..." | tee -a "$MAIN_LOG"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
    
    echo "  Combination processing time: $(date)" | tee -a "$MAIN_LOG"
done

# Restore original scripts
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Restoring original script configurations..." | tee -a "$MAIN_LOG"
restore_python_script "1st_finetuning_20250411共有_all_item.py"
restore_python_script "2nd_validation_20250411共有_all_item.py"

# Final summary
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "All 20250411共有_all_item training and validation tasks completed at $(date)" | tee -a "$MAIN_LOG"
echo "Total combinations processed: $total_combinations" | tee -a "$MAIN_LOG"
echo "Training results: $training_successful successful, $training_failed failed" | tee -a "$MAIN_LOG"
echo "Validation results: $validation_successful successful, $validation_failed failed" | tee -a "$MAIN_LOG"
echo "Main log file: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "Individual combination logs are in: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

echo "Process completed. Check the log files for detailed results."
echo "Training: $training_successful successful, $training_failed failed"
echo "Validation: $validation_successful successful, $validation_failed failed"