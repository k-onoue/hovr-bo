#!/bin/bash -l

# Base timestamp
TIMESTAMP_BASE="2025-01-09_23-00-00.000"

# Function to increment timestamp by N seconds
increment_timestamp() {
    local base="$1"
    local offset="$2"

    # Extract hours, minutes, seconds
    local date_part="${base%%_*}"      # 2025-01-09
    local time_part="${base#*_}"       # 23-00-00.000

    local hours=${time_part:0:2}       # 23
    local minutes=${time_part:3:2}     # 00
    local seconds=${time_part:6:2}     # 00
    local millis=${time_part:9:3}      # 000

    # Convert to total seconds
    local total_seconds=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds + offset))

    # Calculate new hours, minutes, seconds
    local new_hours=$((total_seconds / 3600 % 24))
    local new_minutes=$((total_seconds / 60 % 60))
    local new_seconds=$((total_seconds % 60))

    # Format new time part
    printf "%s_%02d-%02d-%02d.%s\n" "$date_part" "$new_hours" "$new_minutes" "$new_seconds" "$millis"
}

# Function to run sbatch and sleep for a specified time
run_with_delay() {
    sbatch "$1" "$2" "$3"
    sleep 0
}

# Config set 1
TIMESTAMP_ACKLEY2_1=$(increment_timestamp "$TIMESTAMP_BASE" 0)
TIMESTAMP_ACKLEY5_1=$(increment_timestamp "$TIMESTAMP_BASE" 1)
TIMESTAMP_BRANIN_1=$(increment_timestamp "$TIMESTAMP_BASE" 2)
TIMESTAMP_HARTMANN_1=$(increment_timestamp "$TIMESTAMP_BASE" 3)
TIMESTAMP_SYNTHETIC_1=$(increment_timestamp "$TIMESTAMP_BASE" 4)

run_with_delay scripts/run_slurm.sh config/ackley2_1.json "${TIMESTAMP_ACKLEY2_1}"
run_with_delay scripts/run_slurm.sh config/ackley5_1.json "${TIMESTAMP_ACKLEY5_1}"
run_with_delay scripts/run_slurm.sh config/branin_1.json "${TIMESTAMP_BRANIN_1}"
run_with_delay scripts/run_slurm.sh config/hartmann_1.json "${TIMESTAMP_HARTMANN_1}"
run_with_delay scripts/run_slurm.sh config/synthetic_1.json "${TIMESTAMP_SYNTHETIC_1}"

# Config set 2
TIMESTAMP_ACKLEY2_2=$(increment_timestamp "$TIMESTAMP_BASE" 5)
TIMESTAMP_ACKLEY5_2=$(increment_timestamp "$TIMESTAMP_BASE" 6)
TIMESTAMP_BRANIN_2=$(increment_timestamp "$TIMESTAMP_BASE" 7)
TIMESTAMP_HARTMANN_2=$(increment_timestamp "$TIMESTAMP_BASE" 8)
TIMESTAMP_SYNTHETIC_2=$(increment_timestamp "$TIMESTAMP_BASE" 9)

run_with_delay scripts/run_slurm.sh config/ackley2_2.json "${TIMESTAMP_ACKLEY2_2}"
run_with_delay scripts/run_slurm.sh config/ackley5_2.json "${TIMESTAMP_ACKLEY5_2}"
run_with_delay scripts/run_slurm.sh config/branin_2.json "${TIMESTAMP_BRANIN_2}"
run_with_delay scripts/run_slurm.sh config/hartmann_2.json "${TIMESTAMP_HARTMANN_2}"
run_with_delay scripts/run_slurm.sh config/synthetic_2.json "${TIMESTAMP_SYNTHETIC_2}"

# Config set 3
TIMESTAMP_ACKLEY2_3=$(increment_timestamp "$TIMESTAMP_BASE" 10)
TIMESTAMP_ACKLEY5_3=$(increment_timestamp "$TIMESTAMP_BASE" 11)
TIMESTAMP_BRANIN_3=$(increment_timestamp "$TIMESTAMP_BASE" 12)
TIMESTAMP_HARTMANN_3=$(increment_timestamp "$TIMESTAMP_BASE" 13)
TIMESTAMP_SYNTHETIC_3=$(increment_timestamp "$TIMESTAMP_BASE" 14)

run_with_delay scripts/run_slurm.sh config/ackley2_3.json "${TIMESTAMP_ACKLEY2_3}"
run_with_delay scripts/run_slurm.sh config/ackley5_3.json "${TIMESTAMP_ACKLEY5_3}"
run_with_delay scripts/run_slurm.sh config/branin_3.json "${TIMESTAMP_BRANIN_3}"
run_with_delay scripts/run_slurm.sh config/hartmann_3.json "${TIMESTAMP_HARTMANN_3}"
run_with_delay scripts/run_slurm.sh config/synthetic_3.json "${TIMESTAMP_SYNTHETIC_3}"

# Config set 4
TIMESTAMP_ACKLEY2_4=$(increment_timestamp "$TIMESTAMP_BASE" 15)
TIMESTAMP_ACKLEY5_4=$(increment_timestamp "$TIMESTAMP_BASE" 16)
TIMESTAMP_BRANIN_4=$(increment_timestamp "$TIMESTAMP_BASE" 17)
TIMESTAMP_HARTMANN_4=$(increment_timestamp "$TIMESTAMP_BASE" 18)
TIMESTAMP_SYNTHETIC_4=$(increment_timestamp "$TIMESTAMP_BASE" 19)

run_with_delay scripts/run_slurm.sh config/ackley2_4.json "${TIMESTAMP_ACKLEY2_4}"
run_with_delay scripts/run_slurm.sh config/ackley5_4.json "${TIMESTAMP_ACKLEY5_4}"
run_with_delay scripts/run_slurm.sh config/branin_4.json "${TIMESTAMP_BRANIN_4}"
run_with_delay scripts/run_slurm.sh config/hartmann_4.json "${TIMESTAMP_HARTMANN_4}"
run_with_delay scripts/run_slurm.sh config/synthetic_4.json "${TIMESTAMP_SYNTHETIC_4}"
