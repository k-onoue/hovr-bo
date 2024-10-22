#!/bin/bash

# Create results and logs directories if they don't exist
mkdir -p results/

# Get the current date and time
DATE="2024-10-22"

# Create directories based on experimental ID
mkdir -p logs

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /home/onoue/ws/hovr-bo
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(results_dir)s/logs"

# Overwrite config.ini file only if necessary
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

python3 experiments/$DATE/hmc.py
