import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./../config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
DB_DIR = config["paths"]["dbs_dir"]
sys.path.append(PROJECT_DIR)

from src.models import MLP, hovr_regularization, train_model

__all__ = [
    "MLP",
    "hovr_regularization",
    "train_model",
]