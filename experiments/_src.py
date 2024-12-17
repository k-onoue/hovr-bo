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
from src.samplers import gp_sampler, laplace_sampler
from src.test_functions import SyntheticSine, BraninFoo


__all__ = [
    "BayesianOptimization",
    "gp_sampler",
    "laplace_sampler",    
    "SyntheticSine",
    "BraninFoo"
]