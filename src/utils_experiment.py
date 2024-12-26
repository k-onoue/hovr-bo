import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from .samplers import gp_sampler, llla_l2_sampler, llla_artl_sampler, vbla_sampler
from .test_functions import SyntheticSine, BraninFoo, Ackley2d, Ackley5d, Hartmann6d


def set_logger(log_filename_base, save_dir):
    # Set up logging
    log_filename = f"{log_filename_base}.log"
    log_filepath = os.path.join(save_dir, log_filename)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_filepath,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Capture warnings
    logging.captureWarnings(True)

    return log_filepath


def get_objective_function(name):
    if name == "synthetic":
        return SyntheticSine
    elif name == "branin":
        return BraninFoo
    elif name == "hartmann":
        return Hartmann6d
    elif name == "ackley2":
        return Ackley2d
    elif name == "ackley5":
        return Ackley5d
    else:
        raise ValueError(f"Objective function {name} not recognized.")
    

def get_surrogate_model(name):
    if name == "gp":
        return gp_sampler
    elif name == "vbla":
        return vbla_sampler
    elif name == "llla_l2":
        return llla_l2_sampler
    elif name == "llla_artl":
        return llla_artl_sampler
    else:
        raise ValueError(f"Surrogate model {name} not recognized.")
    

def get_acquisition_function(name):
    if name == "log_ei":
        return qLogExpectedImprovement
    elif name == "log_nei":
        return qLogNoisyExpectedImprovement
    elif name == "ucb":
        return qUpperConfidenceBound
    else:
        raise ValueError(f"Acquisition function {name} not recognized.")