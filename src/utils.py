import yaml
from pathlib import Path
from box import ConfigBox
import logging

def load_config(config_path: Path = Path('config/config.yaml')) -> ConfigBox:
    """
    It loads the YAML file and returns it as a ConfigBox object.
    ConfigBox provides dot notation access, such as config.paths.raw_data.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from: {config_path}")
        return ConfigBox(config_data)
    except FileNotFoundError:
        logging.error(f"FATAL: Config file not found at: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise