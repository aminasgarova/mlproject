import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# 🚀 Class to handle the prediction pipeline using saved model and preprocessor
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        '''
        Loads the trained model and preprocessor,
        transforms the input features, and returns predictions.
        '''
        try:
            #  Define paths to model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)
            # Make predictions
            preds = model.predict(data_scaled)

            return preds
        
        #  Handle and raise exceptions with trace info       
        except Exception as e:
            raise CustomException(e, sys)

# 🧾 Class to capture and structure user inputs for prediction
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        # Assign user-provided values to instance attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        '''
        Converts the user input into a pandas DataFrame,
        structured the same way as the training data.
        '''
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            #  Return input as DataFrame for processing/prediction
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
