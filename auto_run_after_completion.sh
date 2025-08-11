#!/bin/bash

# Auto execution script that monitors each script completion individually
# and runs the next script in sequence (with loop back to first script)

# Configuration
SCRIPTS_SEQUENCE=(
    "run_training_validation_all_item.sh"
    "run_training_validation_without_cable_components.sh"
    "run_training_validation_5cat.sh"
    "run_training_validation_all_item.sh"  # Loop back to first script
)
CHECK_INTERVAL=300  # Check every 5 minutes
LOG_FILE="./logs/auto_execution.log"

# Create logs directory
mkdir -p "./logs"

# Function to check if a specific script is still running
is_script_running() {
    local script_name="$1"
    # Check if the script process exists
    if pgrep -f "$script_name" > /dev/null; then
        return 0  # Still running
    else
        return 1  # Not running
    fi
}

# Function to check if any training Python processes are running
is_training_running() {
    # Check for any Python training processes
    if pgrep -f "1st_finetuning.*\.py" > /dev/null || pgrep -f "2nd_validation.*\.py" > /dev/null; then
        return 0  # Training processes still running
    else
        return 1  # No training processes
    fi
}

# Function to wait for a script to complete
wait_for_script_completion() {
    local script_name="$1"
    local script_num="$2"
    local total_scripts="$3"
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$script_num/$total_scripts] Monitoring script: $script_name" | tee -a "$LOG_FILE"
    echo "Check interval: $CHECK_INTERVAL seconds" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Main monitoring loop for this script
    while true; do
        current_time=$(date)
        
        if is_script_running "$script_name"; then
            echo "[$current_time] Script '$script_name' is still running..." | tee -a "$LOG_FILE"
        elif is_training_running; then
            echo "[$current_time] Training processes are still running..." | tee -a "$LOG_FILE"
        else
            echo "[$current_time] Script '$script_name' has completed!" | tee -a "$LOG_FILE"
            break
        fi
        
        # Wait before next check
        sleep "$CHECK_INTERVAL"
    done
    
    echo "[$script_num/$total_scripts] Script '$script_name' monitoring completed at $(date)" | tee -a "$LOG_FILE"
}

# Function to execute a script
execute_script() {
    local script_path="$1"
    local script_num="$2"
    local total_scripts="$3"
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$script_num/$total_scripts] Starting script: $script_path" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    if [ -x "$script_path" ]; then
        # Run the script and capture its output
        echo "Script execution started at $(date)" | tee -a "$LOG_FILE"
        $script_path 2>&1 | tee -a "$LOG_FILE"
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "[$script_num/$total_scripts] Script completed successfully at $(date)" | tee -a "$LOG_FILE"
        else
            echo "[$script_num/$total_scripts] Script failed with exit code $exit_code at $(date)" | tee -a "$LOG_FILE"
        fi
    else
        echo "Error: Script $script_path is not executable or does not exist" | tee -a "$LOG_FILE"
    fi
}

# Main execution
echo "Auto execution monitor started at $(date)" | tee "$LOG_FILE"
echo "Scripts sequence:" | tee -a "$LOG_FILE"
for i in "${!SCRIPTS_SEQUENCE[@]}"; do
    echo "  $((i+1)). ${SCRIPTS_SEQUENCE[$i]}" | tee -a "$LOG_FILE"
done
echo "========================================" | tee -a "$LOG_FILE"

total_scripts=${#SCRIPTS_SEQUENCE[@]}

# Process each script in sequence
for i in "${!SCRIPTS_SEQUENCE[@]}"; do
    script_name="${SCRIPTS_SEQUENCE[$i]}"
    script_num=$((i+1))
    
    if [ $i -eq 0 ]; then
        # First script - just monitor (should already be running)
        wait_for_script_completion "$script_name" "$script_num" "$total_scripts"
    else
        # Subsequent scripts - wait for cleanup, then execute and monitor
        echo "Waiting 60 seconds for cleanup before starting next script..." | tee -a "$LOG_FILE"
        sleep 60
        
        # Execute the script
        script_path="./$script_name"
        execute_script "$script_path" "$script_num" "$total_scripts"
        
        # Monitor its completion
        wait_for_script_completion "$script_name" "$script_num" "$total_scripts"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All scripts execution completed at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"