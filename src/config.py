"""
Configuration Management Module
Loads and manages configuration from YAML files
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration class for managing experiment settings"""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = "configs/clinc150_config.yaml"

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str):
        """Allow dict-like access"""
        return self.get(key)

    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"

    def save(self, save_path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._deep_update(self.config, updates)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config("configs/clinc150_config.yaml")

    print("Configuration loaded successfully!")
    print(f"Dataset: {config.get('dataset.name')}")
    print(f"Model: {config.get('model.bert_model')}")
    print(f"Batch size: {config.get('training.batch_size')}")
    print(f"Num GPUs: {config.get('hardware.num_gpus')}")
