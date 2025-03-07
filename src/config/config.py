import json
import os
from typing import Dict


class Config:
    def __init__(self, config_path: str = "src/config/config.json"):
        """
        Loads configuration from a JSON file.

        Args:
            config_path: Path to the configuration JSON file.
        """
        self.config_path = config_path
        self.data: Dict = self._load_config()

    def _load_config(self) -> Dict:
        """
        Loads the configuration data from the JSON file.

        Returns:
            A dictionary containing the configuration.
        """
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in config file: {self.config_path}")

    def get(self, key: str, default=None):
        """
        Retrieves a configuration value by key.

        Args:
            key: The key to look up.
            default: The default value to return if the key is not found.

        Returns:
            The value associated with the key, or the default value.
        """
        return self.data.get(key, default)
