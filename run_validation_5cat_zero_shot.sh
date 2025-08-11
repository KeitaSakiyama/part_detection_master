#!/bin/bash

# Script to run zero-shot validation with ft_type="full"

# Log file
LOG_DIR="./logs/5cat"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/validation_zero_shot.log"
RUN_LOG="$LOG_DIR/combo_full_0.log"

echo "Starting zero-shot validation (ft_type=full) at $(date)" > "$MAIN_LOG"

# Create backup and update parameters
cp "2nd_validation_20250411共有_5cat.py" "2nd_validation_20250411共有_5cat.py.bak"

# Set ft_type to "full" and shot_num to "zero"
sed -i 's/^ft_type = /# ft_type = /' "2nd_validation_20250411共有_5cat.py"
sed -i 's/^# ft_type = "full"/ft_type = "full"/' "2nd_validation_20250411共有_5cat.py"
sed -i 's/^shot_num = /# shot_num = /' "2nd_validation_20250411共有_5cat.py"
sed -i 's/^# shot_num = "zero"/shot_num = "zero"/' "2nd_validation_20250411共有_5cat.py"

# Run validation
echo "Running validation: ft_type=full, shot_num=zero" >> "$MAIN_LOG"
echo "Starting validation at $(date)..." >> "$MAIN_LOG"

if python 2nd_validation_20250411共有_5cat.py > "$RUN_LOG" 2>&1; then
    echo "✓ Validation completed successfully at $(date)" >> "$MAIN_LOG"
    echo "✓ Validation successful"
else
    echo "✗ Validation failed at $(date)" >> "$MAIN_LOG"
    echo "✗ Validation failed"
fi

# Clean up GPU memory
echo "Cleaning up GPU memory..." >> "$MAIN_LOG"
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Restore original script
mv "2nd_validation_20250411共有_5cat.py.bak" "2nd_validation_20250411共有_5cat.py"

echo "Process completed. Check log files for details:" >> "$MAIN_LOG"
echo "Main log: $MAIN_LOG"
echo "Run log: $RUN_LOG"