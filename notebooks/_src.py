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

from src.ern import EvidentialMLP
from src.ern import train_ern
from src.ern import nsu_reg, hovr_reg

__all__ = [
    "EvidentialMLP",
    "train_ern",
    "nsu_reg",
    "hovr_reg",
]