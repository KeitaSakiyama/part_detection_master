#!/bin/bash

# Auto Git Pull Monitor Script
# This script monitors git repository every 5 minutes and executes specified files when changes are detected

# Configuration
CHECK_INTERVAL=300  # 5 minutes in seconds
LOG_FILE="./logs/git_auto_pull.log"
GIT_LOG_FILE="./logs/git_changes.log"
EXECUTION_SCRIPT="auto_run.sh"  # Default script to execute when changes detected

# Create logs directory
mkdir -p "./logs"

# Function to log messages with timestamp
log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# Function to check if git repository has changes
check_git_changes() {
    local current_commit=$(git rev-parse HEAD)
    
    # Perform git pull
    log_message "Performing git pull..."
    git pull origin master 2>&1 | tee -a "$GIT_LOG_FILE"
    
    local new_commit=$(git rev-parse HEAD)
    
    # Check if commit hash changed
    if [ "$current_commit" != "$new_commit" ]; then
        log_message "Changes detected! Old commit: $current_commit, New commit: $new_commit"
        
        # Show what changed
        log_message "Changed files:"
        git diff --name-only "$current_commit" "$new_commit" | tee -a "$GIT_LOG_FILE"
        
        return 0  # Changes detected
    else
        log_message "No changes detected"
        return 1  # No changes
    fi
}

# Function to execute the specified script
execute_script() {
    local script_to_run="$1"
    
    log_message "=========================================="
    log_message "Executing script: $script_to_run"
    log_message "=========================================="
    
    if [ -f "$script_to_run" ] && [ -x "$script_to_run" ]; then
        # Execute the script and log its output
        log_message "Starting execution of $script_to_run..."
        
        # Run the script in background and capture its PID
        nohup "./$script_to_run" >> "$LOG_FILE" 2>&1 &
        local script_pid=$!
        
        log_message "Script $script_to_run started with PID: $script_pid"
        log_message "Execution started at: $(date)"
        
        # Wait for the script to complete
        wait $script_pid
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            log_message "Script $script_to_run completed successfully"
        else
            log_message "Script $script_to_run failed with exit code: $exit_code"
        fi
        
        log_message "Execution completed at: $(date)"
        log_message "=========================================="
        
    else
        log_message "Error: Script $script_to_run not found or not executable"
    fi
}

# Function to check if any training processes are running
is_training_running() {
    if pgrep -f "1st_finetuning.*\.py" > /dev/null || pgrep -f "2nd_validation.*\.py" > /dev/null || pgrep -f "run_.*\.sh" > /dev/null; then
        return 0  # Training processes are running
    else
        return 1  # No training processes
    fi
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [script_to_execute]"
    echo ""
    echo "Options:"
    echo "  script_to_execute    Script to execute when git changes are detected"
    echo "                       Default: $EXECUTION_SCRIPT"
    echo ""
    echo "Available scripts:"
    echo "  - run_training_validation_all_item.sh"
    echo "  - run_training_validation_5cat.sh" 
    echo "  - run_training_validation_without_cable_components.sh"
    echo "  - run_all_item_training_validation_full_data.sh"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 run_training_validation_5cat.sh"
    echo "  $0 run_all_item_training_validation_full_data.sh"
}

# Parse command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

if [ -n "$1" ]; then
    EXECUTION_SCRIPT="$1"
fi

# Validate execution script exists
if [ ! -f "$EXECUTION_SCRIPT" ]; then
    log_message "Error: Execution script '$EXECUTION_SCRIPT' not found"
    echo "Available scripts in current directory:"
    ls -la run_*.sh 2>/dev/null || echo "No run_*.sh scripts found"
    exit 1
fi

# Make sure the script is executable
chmod +x "$EXECUTION_SCRIPT"

# Main execution
log_message "=========================================="
log_message "Auto Git Pull Monitor Started"
log_message "Check interval: $CHECK_INTERVAL seconds (5 minutes)"
log_message "Execution script: $EXECUTION_SCRIPT"
log_message "Log file: $LOG_FILE"
log_message "Git log file: $GIT_LOG_FILE"
log_message "=========================================="

# Initial git status check
log_message "Initial git status:"
git status --porcelain | tee -a "$GIT_LOG_FILE"

# Main monitoring loop
while true; do
    log_message "Starting git pull check..."
    
    # Check if training is currently running
    if is_training_running; then
        log_message "Training processes are currently running. Skipping git pull check."
    else
        # Check for git changes
        if check_git_changes; then
            log_message "Git changes detected! Waiting 30 seconds before execution..."
            sleep 30
            
            # Double-check that no training started in the meantime
            if is_training_running; then
                log_message "Training started during wait period. Skipping execution."
            else
                # Execute the specified script
                execute_script "$EXECUTION_SCRIPT"
            fi
        fi
    fi
    
    log_message "Next check in $CHECK_INTERVAL seconds..."
    log_message "------------------------------------------"
    
    # Wait for the next check
    sleep "$CHECK_INTERVAL"
done