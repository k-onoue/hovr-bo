#!/bin/bash

# Experiment settings
SEEDS=(0 1 2)
EXPERIMENT_SCRIPT="run_bo.py"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_NAME=$(basename "${SCRIPT_PATH}")

# Check if SETTINGS_FILE is passed as an argument
if [ $# -lt 2 ]; then
    echo "Usage: $SCRIPT_NAME <problem_settings_file> <sampler_settings_file>"
    exit 1
fi
PROBLEM_SETTINGS_FILE=$1
SAMPLER_SETTINGS_FILE=$2

# Generate timestamp (format: YYYY-MM-DD_HH-mm-ss)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Create backup directory and copy files
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
cp "${SCRIPT_PATH}" "${RESULTS_DIR}/"
cp "config/problems/${PROBLEM_SETTINGS_FILE}" "${RESULTS_DIR}/"
cp "config/samplers/${SAMPLER_SETTINGS_FILE}" "${RESULTS_DIR}/"

# Define run function
run_experiment() {
    local seed=$1
    python3 "experiments/${EXPERIMENT_SCRIPT}" \
        --problem "config/problems/${PROBLEM_SETTINGS_FILE}" \
        --sampler "config/samplers/${SAMPLER_SETTINGS_FILE}" \
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