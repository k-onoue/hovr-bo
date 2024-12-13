import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./../config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
RESULTS_DIR = config["paths"]["results_dir"]
sys.path.append(PROJECT_DIR)

from src.laplace import LaplaceBNN


__all__ = [
    "LaplaceBNN",    
]