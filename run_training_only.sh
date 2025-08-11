#!/bin/bash

# Script to run only training for different configurations
# MAX_TRAIN_ANNOTATIONS_PER_CATEGORY: 10, 50, 100, 150, 200, 300, 400
# ft_type: prompt, linear_prob

set -e  # Exit on any error

# Define arrays for the parameters
MAX_TRAIN_VALUES=(10 50 100 150 200 300 400)
FT_TYPES=("prompt" "linear_prob")

# Log file
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/training_only.log"

echo "Starting training process at $(date)" | tee "$MAIN_LOG"
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
    update_python_param "1st_finetuning_train_limited_other_tuning.py" "MAX_TRAIN_ANNOTATIONS_PER_CATEGORY" "$max_train"
    update_python_param "1st_finetuning_train_limited_other_tuning.py" "ft_type" "$ft_type"
    
    # Run training
    echo "  Starting training at $(date)..." | tee -a "$log_file"
    if python 1st_finetuning_train_limited_other_tuning.py >> "$log_file" 2>&1; then
        echo "  Training completed successfully at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "  Training failed at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Main execution loop
total_combinations=$((${#MAX_TRAIN_VALUES[@]} * ${#FT_TYPES[@]}))
current_combination=0
successful_combinations=0
failed_combinations=0

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
        COMBO_LOG="$LOG_DIR/training_${ft_type}_${max_train}.log"
        
        # Run training
        if run_training "$max_train" "$ft_type" "$COMBO_LOG"; then
            echo "✓ Training combination $current_combination/$total_combinations completed successfully" | tee -a "$MAIN_LOG"
            successful_combinations=$((successful_combinations + 1))
        else
            echo "✗ Training failed for combination $current_combination/$total_combinations" | tee -a "$MAIN_LOG"
            failed_combinations=$((failed_combinations + 1))
        fi
        
        # Clean up GPU memory
        echo "  Cleaning up GPU memory..." | tee -a "$MAIN_LOG"
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
        
        echo "  Combination processing time: $(date)" | tee -a "$MAIN_LOG"
    done
done

# Restore original script
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Restoring original script configuration..." | tee -a "$MAIN_LOG"
restore_python_script "1st_finetuning_train_limited_other_tuning.py"

# Final summary
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "All training tasks completed at $(date)" | tee -a "$MAIN_LOG"
echo "Total combinations processed: $total_combinations" | tee -a "$MAIN_LOG"
echo "Successful combinations: $successful_combinations" | tee -a "$MAIN_LOG"
echo "Failed combinations: $failed_combinations" | tee -a "$MAIN_LOG"
echo "Main log file: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "Individual training logs are in: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

echo "Training process completed. Check the log files for detailed results."
echo "Successful: $successful_combinations, Failed: $failed_combinations out of $total_combinations total combinations."