#!/bin/bash -l

# Function to run sbatch and sleep for a specified time
run_with_delay() {
    sbatch "$1" "$2" "$3"
    sleep 60
}

# Timestamps for config set 1
TIMESTAMP_ACKLEY2_1="2025-01-08_20-04-06.951"
TIMESTAMP_ACKLEY5_1="2025-01-08_20-05-06.508"
TIMESTAMP_BRANIN_1="2025-01-08_20-06-07.324"
TIMESTAMP_HARTMANN_1="2025-01-08_20-10-07.178"
TIMESTAMP_SYNTHETIC_1="2025-01-08_20-10-07.164"

# Timestamps for config set 2
TIMESTAMP_ACKLEY2_2="2025-01-08_20-10-07.176"
TIMESTAMP_ACKLEY5_2="2025-01-08_20-10-07.164"
TIMESTAMP_BRANIN_2="2025-01-08_20-35-07.266"
TIMESTAMP_HARTMANN_2="2025-01-08_20-35-07.279"
TIMESTAMP_SYNTHETIC_2="2025-01-08_20-35-07.302"

# Timestamps for config set 3
TIMESTAMP_ACKLEY2_3="2025-01-08_20-35-07.296"
TIMESTAMP_ACKLEY5_3="2025-01-08_20-35-07.296"
TIMESTAMP_BRANIN_3="2025-01-08_20-35-07.301"
TIMESTAMP_HARTMANN_3="2025-01-08_20-35-07.258"
TIMESTAMP_SYNTHETIC_3="2025-01-08_20-35-07.316"

# Timestamps for config set 4
TIMESTAMP_ACKLEY2_4="2025-01-08_20-35-07.329"
TIMESTAMP_ACKLEY5_4="2025-01-08_20-36-08.101"
TIMESTAMP_BRANIN_4="2025-01-08_20-36-08.104"
TIMESTAMP_HARTMANN_4="2025-01-08_20-36-08.099"
TIMESTAMP_SYNTHETIC_4="2025-01-08_20-36-08.105"

# Config set 1
run_with_delay scripts/run_slurm.sh config/ackley2_1.json "${TIMESTAMP_ACKLEY2_1}"
run_with_delay scripts/run_slurm.sh config/ackley5_1.json "${TIMESTAMP_ACKLEY5_1}"
run_with_delay scripts/run_slurm.sh config/branin_1.json "${TIMESTAMP_BRANIN_1}"
run_with_delay scripts/run_slurm.sh config/hartmann_1.json "${TIMESTAMP_HARTMANN_1}"
run_with_delay scripts/run_slurm.sh config/synthetic_1.json "${TIMESTAMP_SYNTHETIC_1}"

# Config set 2
run_with_delay scripts/run_slurm.sh config/ackley2_2.json "${TIMESTAMP_ACKLEY2_2}"
run_with_delay scripts/run_slurm.sh config/ackley5_2.json "${TIMESTAMP_ACKLEY5_2}"
run_with_delay scripts/run_slurm.sh config/branin_2.json "${TIMESTAMP_BRANIN_2}"
run_with_delay scripts/run_slurm.sh config/hartmann_2.json "${TIMESTAMP_HARTMANN_2}"
run_with_delay scripts/run_slurm.sh config/synthetic_2.json "${TIMESTAMP_SYNTHETIC_2}"

# Config set 3
run_with_delay scripts/run_slurm.sh config/ackley2_3.json "${TIMESTAMP_ACKLEY2_3}"
run_with_delay scripts/run_slurm.sh config/ackley5_3.json "${TIMESTAMP_ACKLEY5_3}"
run_with_delay scripts/run_slurm.sh config/branin_3.json "${TIMESTAMP_BRANIN_3}"
run_with_delay scripts/run_slurm.sh config/hartmann_3.json "${TIMESTAMP_HARTMANN_3}"
run_with_delay scripts/run_slurm.sh config/synthetic_3.json "${TIMESTAMP_SYNTHETIC_3}"

# Config set 4
run_with_delay scripts/run_slurm.sh config/ackley2_4.json "${TIMESTAMP_ACKLEY2_4}"
run_with_delay scripts/run_slurm.sh config/ackley5_4.json "${TIMESTAMP_ACKLEY5_4}"
run_with_delay scripts/run_slurm.sh config/branin_4.json "${TIMESTAMP_BRANIN_4}"
run_with_delay scripts/run_slurm.sh config/hartmann_4.json "${TIMESTAMP_HARTMANN_4}"
run_with_delay scripts/run_slurm.sh config/synthetic_4.json "${TIMESTAMP_SYNTHETIC_4}"