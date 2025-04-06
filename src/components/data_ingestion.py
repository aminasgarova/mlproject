# 📦 Import necessary Python modules
import os   
import sys  # For handling system-specific errors
import pandas as pd  
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# 🧱 STEP 1: Define where to save the data files using a configuration class
@dataclass
class DataIngestionConfig:
    # File path for saving training dataset
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    # File path for saving testing dataset
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    # File path for saving the raw (original) dataset
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# 🔄 STEP 2: Create the DataIngestion class that does the actual job
class DataIngestion:
    def __init__(self):
        # Create a config object that gives us file paths to save our data
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")

        try:
            # 📥 STEP 3: Read the dataset from a CSV file (in your notebook/data folder)
            df = pd.read_csv('notebook/data/stud.csv')  
            logging.info("Dataset loaded into DataFrame")

            # 📁 STEP 4: Create the 'artifacts' folder if it doesn't already exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 💾 STEP 5: Save the original/raw data to 'artifacts/data.csv' for backup
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # ✂️ STEP 6: Split the data into training (80%) and testing (20%)
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 💽 STEP 7: Save the train and test data to separate CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            # 🔁 STEP 8: Return the file paths so that other components can use the data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # 🧯 If anything goes wrong, handle the exception using custom logic
        except Exception as e:
            logging.error("Error occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)
