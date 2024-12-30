#!/bin/bash

# Experiment settings
SEEDS=(0)
EXPERIMENT_SCRIPT="run_bo.py"
SETTINGS_FILE="test.json"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_NAME=$(basename "${SCRIPT_PATH}")

# Generate timestamp (format: YYYY-MM-DD_HH-mm-ss)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Create backup directory and copy files
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
cp "${SCRIPT_PATH}" "${RESULTS_DIR}/"
cp "config/${SETTINGS_FILE}" "${RESULTS_DIR}/"

# Define run function
run_experiment() {
    local seed=$1
    python3 "experiments/${EXPERIMENT_SCRIPT}" \
        --settings "config/${SETTINGS_FILE}" \
        --timestamp "${TIMESTAMP}" \
        --seed "${seed}"
}

# Run experiments in parallel
for seed in "${SEEDS[@]}"; do
    run_experiment "${seed}" &
done

# Wait for all background processes to complete
wait

# Create completion file
COMPLETION_FILE="${RESULTS_DIR}/completion.txt"
cat > "${COMPLETION_FILE}" << EOF
Experiment completed at: $(date)
Settings file: ${SETTINGS_FILE}
Seeds: ${SEEDS[@]}
Shell script: ${SCRIPT_NAME}
EOF