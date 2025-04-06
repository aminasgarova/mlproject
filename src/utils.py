import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

# 💾 Function to save any Python object (like models, transformers, etc.) to disk using pickle
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  #  Extract directory from file path
        os.makedirs(dir_path, exist_ok=True)   #  Create directory if it doesn't exist
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)          # Serialize and save the object

    # Raise custom exception if saving fails
    except Exception as e:
        raise CustomException(e, sys)
    
# 📊 Function to train and evaluate multiple models using GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Dictionary to store evaluation results

        for i in range(len(list(models))):
            model = list(models.values())[i]      #  Get model
            para = param[list(models.keys())[i]]  #  Get parameters for GridSearch

            # Hyperparameter tuning using GridSearchCV with 3-fold CV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            #  Set the best found parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  

            # Predictions on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores for evaluation
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test R² score in report
            report[list(models.keys())[i]] = test_model_score

        return report  

    except Exception as e:
        raise CustomException(e, sys)

# 📂 Function to load any saved Python object (model, pipeline, etc.)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  #  Load the object from disk

    except Exception as e:
        raise CustomException(e, sys)
