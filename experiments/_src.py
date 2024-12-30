import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
RESULTS_DIR = config["paths"]["results_dir"]
sys.path.append(PROJECT_DIR)

from src.bo import BayesianOptimization
from src.utils_experiment import set_logger
from src.utils_experiment import get_objective_function, get_surrogate_model, get_acquisition_function


__all__ = [
    "BayesianOptimization",
    "get_objective_function",
    "get_surrogate_model",
    "get_acquisition_function",
    "set_logger",
]