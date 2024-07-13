#create prediction pipeline class
#create function to load an object
#create custom class based upon the dataset
#create function to convert data into Dataframe with the help of dict

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    @staticmethod
    def predict(features):
        preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
        model_path = os.path.join("artifact", "model_trainer", "model.pkl")

        processor = load_object(preprocessor_path)
        model = load_object(model_path)

        scaled = processor.transform(features)
        pred = model.predict(scaled)

        return pred
    
class customClass:
    def __init__(self, age : int,
                      workclass: int, 
                      education_num: int, 
                      marital_status: int, 
                      occupation: int, 
                      relationship: int, 
                      race: int, 
                      sex: int, 
                      capital_loss: int, 
                      capital_gain: int, 
                      hours_per_week: int, 
                      native_country: int):
            
            self.age = age
            self.workclass = workclass
            self.education_num = education_num
            self.marital_status = marital_status
            self.occupation = occupation
            self.relationship = relationship
            self.race = race
            self.sex = sex
            self.capital_loss = capital_loss
            self.capital_gain = capital_gain
            self.hours_per_week = hours_per_week
            self.native_country = native_country

    def get_data_DataFrame(self):
            try:
                custom_input = {
                    'age': [self.age],
                    'workclass': [self.workclass],
                    'education_num': [self.education_num],
                    'marital_status': [self.marital_status],
                    'occupation': [self.occupation],
                    'relationship': [self.relationship],
                    'race': [self.race],
                    'sex': [self.sex],
                    'capital_gain': [self.capital_gain],
                    'capital_loss': [self.capital_loss],
                    'hours_per_week': [self.hours_per_week],
                    'native_country': [self.native_country],                    
                }

                data = pd.DataFrame(custom_input)

                return data
            except Exception as e:
                raise CustomException(e, sys)       