import datetime
import logging
import os
import sys


def set_logger(log_filename_base, save_dir):
    # Set up logging
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_time}_{log_filename_base}.log"
    log_filepath = os.path.join(save_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )
