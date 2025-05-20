"""
This module provides helper functions for the AB Hem Chat bot agent.

Precondition is that these packages are pip installed:
pip install email-validator
"""

from email_validator import EmailNotValidError, validate_email
from lib.config_loader import Config

# Initialize configuration
config = Config("ABhem_QAbot_agent/config.json")

from lib.logger_setup import get_logger  # noqa: E402 (import not at top of file)

# Setup a logger which can be used by all helper functions
logger = get_logger(__name__, config)


def is_valid_email(email):
    """
    Validates an email address using the email-validator library.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
        str: Error message if validation fails, empty string otherwise.
    """
    try:
        # Validate the email
        # normalizes=True will fix minor formatting issues if possible
        # check_deliverability=True verifies that the domain has valid MX records
        validation = validate_email(email, check_deliverability=True)

        # Get the normalized form of the email address
        normalized_email = validation.email

        return True, f"Valid email: {normalized_email}"
    except EmailNotValidError as e:
        # Return False along with the error message
        return False, str(e)
