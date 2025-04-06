import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

# ⚙️ Configuration class to define where the preprocessor object will be saved
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

# 🔄 Class responsible for transforming the data
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # 🧩 Function to create and return a preprocessing pipeline
    def get_data_transformer_object(self):
        '''
        Builds and returns a data preprocessing pipeline:
        - Imputes missing values
        - Encodes categorical features
        - Scales features appropriately
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = \
            [ "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course" ]

            # Pipeline for numerical features: median imputation + standard scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  
                    ("scaler", StandardScaler())  
                ] )

            # Pipeline for categorical features: mode imputation + one-hot encoding + scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), 
                    ("one_hot_encoder", OneHotEncoder()),  
                    ("scaler", StandardScaler(with_mean=False))  
                ] )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #  Combine both pipelines into one preprocessor using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ] )

            return preprocessor

        # Raise custom exception on error 
        except Exception as e:
            raise CustomException(e, sys)
        
    # 🚀 Function to apply the preprocessing steps to train and test data
    def initiate_data_transformation(self, train_path, test_path):
        '''
        Reads datasets, applies transformations, and saves the preprocessing object.
        Returns:
        - Transformed train and test arrays
        - File path of the saved preprocessing object
        '''
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Get the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target variable and separate features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split train and test into input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply transformations
            logging.info(" Applying preprocessing object to data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine processed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df) ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor to a file for later use (e.g. in model inference)
            logging.info(" Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the processed arrays and preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        # Catch and raise errors using custom exception handler
        except Exception as e:
            raise CustomException(e, sys)
