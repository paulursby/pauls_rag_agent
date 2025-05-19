"""
This module provides helper functions for the AB Hem Chat bot agent.

Precondition is that these packages are pip installed:
pip install email-validator
"""

from email_validator import EmailNotValidError, validate_email


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
