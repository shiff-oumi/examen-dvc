import yaml
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE_PATH = PROJECT_DIR / "src" / "config" / "config.yaml"

def load_config(config_file=CONFIG_FILE_PATH):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
