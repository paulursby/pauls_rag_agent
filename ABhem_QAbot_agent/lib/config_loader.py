"""
This module provides a singleton class `Config`.

The class is used for loading and managing configuration settings from a JSON file.
"""

import json


class Config:
    """Singleton class for loading and accessing configuration data.

    This class ensures that the configuration file is loaded only once and
    provides a global access point to configuration settings.

    Attributes:
        _instance (Config): The single instance of the class.
        data (dict): The configuration settings loaded from the JSON file.
    """

    _instance = None

    def __new__(cls, config_path=None):
        """Creates a new instance of Config if one does not already exist.

        Args:
            config_path (str, optional): Path to the JSON configuration file.

        Returns:
            Config: The singleton instance of the Config class.

        Raises:
            ValueError: If config_path is not provided on first instantiation.
        """
        if cls._instance is None:
            if config_path is None:
                raise ValueError("Config path must be provided on first instantiation!")
            with open(config_path, "r") as file:
                cls._instance = super().__new__(cls)
                cls._instance.data = json.load(file)
        return cls._instance

    @staticmethod
    def get():
        """Retrieves the loaded configuration data.

        Returns:
            dict: The configuration settings.

        Raises:
            RuntimeError: If the Config instance has not been initialized.
        """
        if Config._instance is None:
            raise RuntimeError("Config must be initialized first!")
        # logger.info("Configuration data fetched.")
        return Config._instance.data
