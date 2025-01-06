#!/bin/bash -l

sbatch scripts/run_slurm.sh config/ackley2_1.json
sbatch scripts/run_slurm.sh config/ackley5_1.json
sbatch scripts/run_slurm.sh config/branin_1.json
sbatch scripts/run_slurm.sh config/hartmann_1.json
sbatch scripts/run_slurm.sh config/synthetic_1.json