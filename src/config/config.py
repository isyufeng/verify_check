import json
from pathlib import Path


class Config:

    @staticmethod
    def load_experiment_config(config_path: str = "experiment_config.json"):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
