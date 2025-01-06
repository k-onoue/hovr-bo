#!/bin/bash

# Experiment settings
SEEDS=(0 1)
EXPERIMENT_SCRIPT="run_bo.py"
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_NAME=$(basename "${SCRIPT_PATH}")

# Check if SETTINGS_FILE is passed as an argument
if [ $# -lt 1 ]; then
    echo "Usage: $SCRIPT_NAME <settings_file>"
    exit 1
fi
SETTINGS_FILE=$1

# Generate timestamp (format: YYYY-MM-DD_HH-mm-ss)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Create backup directory and copy files
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
cp "${SCRIPT_PATH}" "${RESULTS_DIR}/"
cp "${SETTINGS_FILE}" "${RESULTS_DIR}/"

# Get sampler types from config file
SAMPLER_TYPES=($(python3 -c "
import json
with open('${SETTINGS_FILE}') as f:
    config = json.load(f)
print(' '.join(config['samplers'].keys()))
"))

# Define run function
run_experiment() {
    local seed=$1
    local sampler_type=$2
    python3 "experiments/${EXPERIMENT_SCRIPT}" \
        --config "${SETTINGS_FILE}" \
        --sampler_type "${sampler_type}" \
        --timestamp "${TIMESTAMP}" \
        --seed "${seed}"
}

# Run experiments in parallel for each sampler and seed
for sampler_type in "${SAMPLER_TYPES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "${seed}" "${sampler_type}" &
    done

    wait
done

# Wait for all background processes to complete
wait

# Create completion file
COMPLETION_FILE="${RESULTS_DIR}/completion.txt"
cat > "${COMPLETION_FILE}" << EOF
Experiment completed at: $(date)
Settings file: ${SETTINGS_FILE}
Seeds: ${SEEDS[@]}
Samplers: ${SAMPLER_TYPES[@]}
Shell script: ${SCRIPT_NAME}
EOF