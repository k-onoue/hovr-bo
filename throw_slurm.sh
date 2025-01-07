#!/bin/bash -l

# Function to run sbatch and sleep for a specified time
run_with_delay() {
    sbatch "$1" "$2"
    sleep 1  # Delay for 1 second
}

run_with_delay scripts/run_slurm.sh config/ackley2_1.json
run_with_delay scripts/run_slurm.sh config/ackley5_1.json
run_with_delay scripts/run_slurm.sh config/branin_1.json
run_with_delay scripts/run_slurm.sh config/hartmann_1.json
run_with_delay scripts/run_slurm.sh config/synthetic_1.json

run_with_delay scripts/run_slurm.sh config/ackley2_2.json
run_with_delay scripts/run_slurm.sh config/ackley5_2.json
run_with_delay scripts/run_slurm.sh config/branin_2.json
run_with_delay scripts/run_slurm.sh config/hartmann_2.json
run_with_delay scripts/run_slurm.sh config/synthetic_2.json

run_with_delay scripts/run_slurm.sh config/ackley2_3.json
run_with_delay scripts/run_slurm.sh config/ackley5_3.json
run_with_delay scripts/run_slurm.sh config/branin_3.json
run_with_delay scripts/run_slurm.sh config/hartmann_3.json
run_with_delay scripts/run_slurm.sh config/synthetic_3.json

run_with_delay scripts/run_slurm.sh config/ackley2_3.json
run_with_delay scripts/run_slurm.sh config/ackley5_3.json
run_with_delay scripts/run_slurm.sh config/branin_3.json
run_with_delay scripts/run_slurm.sh config/hartmann_3.json
run_with_delay scripts/run_slurm.sh config/synthetic_3.json
