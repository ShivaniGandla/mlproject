import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger.logger_config import logger
import pickle

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

    def get_data_transformer_object(self, X):
        try:
            numerical_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(exclude=[np.number]).columns

            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Data Transformation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(columns=['target'], axis=1)
            y_train = train_df['target']

            X_test = test_df.drop(columns=['target'], axis=1)
            y_test = test_df['target']

            preprocessor = self.get_data_transformer_object(X_train)

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Save preprocessor object
            with open(self.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logger.info("Data Transformation completed successfully")

            return (X_train_scaled, X_test_scaled, y_train, y_test, self.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)