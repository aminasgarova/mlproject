# 📦 Import necessary Python modules
import os   
import sys  # For handling system-specific errors
import pandas as pd  
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# 🧱Define where to save the data files using a configuration class training/testing/the raw (original) dataset
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# 🔄Create the DataIngestion class that does the actual job
class DataIngestion:
    def __init__(self):
        # Create a config object that gives us file paths to save our data
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            # Read the dataset from a CSV file (in your notebook/data folder)
            logging.info("Read the dataset as dataframe")
            df = pd.read_csv('notebook\data\stud.csv')  

            # Create the 'artifacts' folder if it doesn't already exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the original/raw data to 'artifacts/data.csv' for backup
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Split the data into training (80%) and testing (20%), separate CSV files
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully")

            # Return the file paths so that other components can use the data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        #  If anything goes wrong, handle the exception using custom logic
        except Exception as e:
            logging.error("Error occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))