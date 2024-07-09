import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.Data_Transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'data_ingestion', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'data_ingestion', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data_ingestion', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            data_path = os.path.join("notebook", "data", "income_dataset.csv")
            data = pd.read_csv(data_path)

            logging.info("Data reading from local system")   

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)
            logging.info("Train-test split completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion stage")
            raise CustomException(e, sys)
            
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path, _ = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
