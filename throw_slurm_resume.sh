#!/bin/bash -l

# Function to run sbatch and sleep for a specified time
run_with_delay() {
    sbatch "$1" "$2" "$3"  # Added timestamp parameter
    sleep 60
}

# Timestamps from existing runs
TIMESTAMP_1="2025-01-08_20-04-06.951"
TIMESTAMP_2="2025-01-08_20-10-07.176"
TIMESTAMP_3="2025-01-08_20-35-07.296"
TIMESTAMP_4="2025-01-08_20-36-08.099"

# Config set 1
run_with_delay scripts/run_slurm.sh config/ackley2_1.json "${TIMESTAMP_1}"
run_with_delay scripts/run_slurm.sh config/ackley5_1.json "${TIMESTAMP_1}"
run_with_delay scripts/run_slurm.sh config/branin_1.json "${TIMESTAMP_1}"
run_with_delay scripts/run_slurm.sh config/hartmann_1.json "${TIMESTAMP_1}"
run_with_delay scripts/run_slurm.sh config/synthetic_1.json "${TIMESTAMP_1}"

# Config set 2
run_with_delay scripts/run_slurm.sh config/ackley2_2.json "${TIMESTAMP_2}"
run_with_delay scripts/run_slurm.sh config/ackley5_2.json "${TIMESTAMP_2}"
run_with_delay scripts/run_slurm.sh config/branin_2.json "${TIMESTAMP_2}"
run_with_delay scripts/run_slurm.sh config/hartmann_2.json "${TIMESTAMP_2}"
run_with_delay scripts/run_slurm.sh config/synthetic_2.json "${TIMESTAMP_2}"

# Config set 3
run_with_delay scripts/run_slurm.sh config/ackley2_3.json "${TIMESTAMP_3}"
run_with_delay scripts/run_slurm.sh config/ackley5_3.json "${TIMESTAMP_3}"
run_with_delay scripts/run_slurm.sh config/branin_3.json "${TIMESTAMP_3}"
run_with_delay scripts/run_slurm.sh config/hartmann_3.json "${TIMESTAMP_3}"
run_with_delay scripts/run_slurm.sh config/synthetic_3.json "${TIMESTAMP_3}"

# Config set 4
run_with_delay scripts/run_slurm.sh config/ackley2_4.json "${TIMESTAMP_4}"
run_with_delay scripts/run_slurm.sh config/ackley5_4.json "${TIMESTAMP_4}"
run_with_delay scripts/run_slurm.sh config/branin_4.json "${TIMESTAMP_4}"
run_with_delay scripts/run_slurm.sh config/hartmann_4.json "${TIMESTAMP_4}"
run_with_delay scripts/run_slurm.sh config/synthetic_4.json "${TIMESTAMP_4}"