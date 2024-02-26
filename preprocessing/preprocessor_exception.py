"""This module contains the exceptions needed for the preprocessing classes."""

from typing import Optional


class PreprocessorException(Exception):
    """Abstract Preprocessor Exception class.

    This class must be extended and not instantiated.
    """

    def __init__(self, error_message: str, code: Optional[int]):
        """Raise RuntimeError if this exception is instantiated."""
        if type(self) == PreprocessorException:
            raise RuntimeError("Abstract class <PreprocessorException> must not be instantiated.")
        self.status = error_message
        self.error = {"message": error_message}
        self.code = code or 400  # DEFAULT_EXCEPTION_CODE = 400
