"""
This module provides functionality for setting up and retrieving loggers.

It configures logging to both console and file output with customizable log levels
based on application configuration.
"""

import logging
import os

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(config):
    """
    Configure global logging settings based on provided configuration.

    Sets up the root logger with appropriate handlers for both file and console output.
    Creates the log directory if it doesn't exist.

    Args:
        config (Config): Configuration object with get_param method to access settings.

    Returns:
        None
    """
    # Create the directory if it doesn't exist
    log_filepath = config.get_param("logging", "log_filepath")
    os.makedirs(log_filepath, exist_ok=True)

    # Fetch log_level_basic from config
    log_level_basic = LOG_LEVELS.get(config.get_param("logging", "level_basic"))

    logging.basicConfig(
        level=log_level_basic,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_filepath, "ABhem_Chatbot_backend.log")
            ),
            logging.StreamHandler(),
        ],
    )


def get_logger(name, config):
    """
    Retrieves and configures a logger instance with the given name and custom log level.

    Creates a child logger with the specified name and sets its level according to
    the chatbot-specific configuration setting.

    Args:
        name (str): The name of the logger, typically __name__ of the calling module
        config (Config): Configuration object with get_param method to access settings.

    Returns:
        logging.Logger: A configured logger instance with the specified name and level.
    """
    logger = logging.getLogger(name)
    log_level_chatbot = LOG_LEVELS.get(config.get_param("logging", "level_chatbot"))
    logger.setLevel(log_level_chatbot)

    return logger
