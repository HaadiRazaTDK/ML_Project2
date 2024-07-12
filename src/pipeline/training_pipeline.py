import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.logger import logging
from src.exception import CustomException
from src.components.Data_ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path, _ = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_training = ModelTrainer()
    model_training.initiate_model_trainer(train_arr, test_arr)