import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.logger.logger_config import logger
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self):
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started")

        try:
            data = load_iris(as_frame=True).frame
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logger.info("Data Ingestion completed successfully")
            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)