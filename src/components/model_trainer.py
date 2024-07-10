import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifact", "model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "RandomForest": RandomForestClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "LogisticRegression": LogisticRegression()
            }

            params = {
                "RandomForest": {
                    "class_weight": ["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },

                "DecisionTree": {
                    "class_weight": ["balanced"],
                    "criterion": ['gini', "entropy", "log_loss"],
                    "splitter": ['best', 'random'],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_split": [2, 3, 4, 5],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": ["auto", "sqrt", "log2"]
                },

                "LogisticRegression": {
                    "class_weight": ["balanced"],
                    "penalty": ["l1", "l2"],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "saga"]
                }
            }

            model_report: dict = evaluate_model(X_train=X_train, 
                                                y_train=y_train, 
                                                X_test=X_test, 
                                                y_test=y_test, 
                                                models=models, 
                                                params=params)
            
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            logging.info(f"Best model found: {best_model_name} with an accuracy of {best_model_score}.")

        except Exception as e:
            raise CustomException(e, sys)

