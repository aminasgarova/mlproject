import logging  # Python’s built-in logging module for tracking events
import os       # To handle file system paths and directories
from datetime import datetime  # To timestamp log files

# 1. Create a unique log file name using current date and time
# Example: '04_05_2025_15_32_10.log'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path where the log file should be stored
# Logs will be stored inside a "logs" folder in the current working directory
# Final path: /your/project/path/logs/<timestamp>.log
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# 3. Create the "logs" directory if it doesn’t already exist
# exist_ok=True prevents error if the directory already exists
os.makedirs(logs_path, exist_ok=True)

# 4. Join directory and filename to get full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 5. Configure the logging system:
# - filename: where to save logs
# - format: how log messages will look
# - level: what level of messages to log (INFO and above)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")

