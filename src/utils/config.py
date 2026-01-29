"""
Configuration utilities for loading and managing configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the fintech AI platform."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        if config_path is None:
            # Get project root (two levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys with dot notation (e.g., 'credit_risk.model.algorithm')

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()

    def update(self, key: str, value: Any):
        """
        Update configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, filepath: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration (default: original config path)
        """
        if filepath is None:
            filepath = self.config_path

        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    return Config(config_path)


if __name__ == "__main__":
    # Example usage
    config = get_config()

    print("Configuration loaded successfully!")
    print("\nSample configuration values:")
    print(f"Credit Risk Algorithm: {config.get('credit_risk.model.algorithm')}")
    print(f"Fraud Detection Algorithm: {config.get('fraud_detection.model.algorithm')}")
    print(f"Data Raw Directory: {config.get('data.raw_dir')}")
    print(f"Model Save Directory: {config.get('models.save_dir')}")
