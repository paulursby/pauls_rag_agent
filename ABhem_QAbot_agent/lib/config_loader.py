"""
This module provides a singleton class `Config`.

The class is used for loading and managing configuration settings from a JSON file,
with support for secure parameters stored in environment variables.
"""

import json
import os
from typing import Any


class Config:
    """Singleton class for loading and accessing configuration data.

    This class ensures that the configuration file is loaded only once and
    provides a global access point to configuration settings.

    Attributes:
        _instance (Config): The single instance of the class.
        data (dict): The configuration settings loaded from the JSON file.
        _secure_params (dict): Mapping of configuration paths to env variable names.
    """

    _instance = None
    _secure_params = {}

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
                cls._instance._secure_params = {}
        return cls._instance

    @classmethod
    def register_secure_param(cls, config_path: str, env_var_name: str):
        """Register a configuration parameter as secure.

        Args:
            config_path (str): Path to the parameter in the config
                (e.g., 'backend.back_office_email_sender_pwd')
            env_var_name (str): Name of the environment variable that contains the
                secure value
        """
        if cls._instance is None:
            raise RuntimeError("Config must be initialized first!")
        cls._instance._secure_params[config_path] = env_var_name

    def get_param(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration parameter, checking if it's a secure parameter.

        Args:
            *keys: The nested keys to navigate to the parameter
            default: Default value if parameter doesn't exist

        Returns:
            The parameter value, either from config or from environment variable if secure
        """
        # Create the config path for lookup in secure params
        config_path = ".".join(keys)

        # Check if this is a registered secure parameter
        if config_path in self._secure_params:
            env_var_name = self._secure_params[config_path]
            # Get the value from environment variable, or return default if not set
            return os.environ.get(env_var_name, default)

        # Regular parameter lookup
        current = self.data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
