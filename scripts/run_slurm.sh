#!/bin/bash -l

# SLURM directives
#SBATCH --job-name=bo_experiment
#SBATCH --output=%x_%j.log
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=4:00:00

# Experiment settings
SEEDS=(0 1 2 3 4 5 6 7 8 9)
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

# Function to create SLURM job scripts
create_slurm_job() {
    local seed=$1
    local sampler_type=$2
    local job_file="${RESULTS_DIR}/job_${sampler_type}_${seed}.slurm"

    cat > "$job_file" << EOF
#!/bin/bash -l
#SBATCH --job-name=bo_${sampler_type}_${seed}
#SBATCH --output=${RESULTS_DIR}/%x_%j.log
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00

python3 "experiments/${EXPERIMENT_SCRIPT}" \
    --config "${SETTINGS_FILE}" \
    --sampler_type "${sampler_type}" \
    --timestamp "${TIMESTAMP}" \
    --seed "${seed}"
EOF

    echo "$job_file"
}

# Submit SLURM jobs for each sampler and seed
for sampler_type in "${SAMPLER_TYPES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        job_file=$(create_slurm_job "$seed" "$sampler_type")
        sbatch "$job_file"
    done

done

# Delete job files
rm "${RESULTS_DIR}/job_"*

# Create completion file
COMPLETION_FILE="${RESULTS_DIR}/completion.txt"
cat > "${COMPLETION_FILE}" << EOF
Experiment setup completed at: $(date)
Settings file: ${SETTINGS_FILE}
Seeds: ${SEEDS[@]}
Samplers: ${SAMPLER_TYPES[@]}
Shell script: ${SCRIPT_NAME}
EOF
