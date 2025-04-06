import sys  # Used to get details about the current exception (like file name, line number)
from src.logger import logging

# Function to generate a detailed error message
def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message including:
    - File where the error occurred
    - Line number of the error
    - Actual error message
    """
    # Unpack exception info from sys
    # _       → the error type (e.g., ZeroDivisionError)
    # _       → the actual exception instance (e.g., "division by zero")
    # exc_tb  → the traceback object (holds info about where the error happened)
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the file name from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Construct a custom error message using file name, line number, and error message
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error) )
    # Return the complete error message string
    return error_message  

# Custom Exception Class
class CustomException(Exception):
    """
    Custom exception class that extends Python's built-in Exception class.
    Automatically includes:
    - File name where the error happened
    - Line number of the error
    - Error message
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with a detailed message.
        Parameters:
        - error_message: The original error message
        - error_detail: sys module to extract traceback info
        """
        super().__init__(error_message)  # Initialize the base Exception class
        # Use our helper function to build the detailed error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        When this exception is printed, show the detailed message.
        """
        return self.error_message
    
# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divide by zero")
#         raise CustomException(e, sys)

