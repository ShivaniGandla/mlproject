import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.logger.logger_config import logger

class CustomData:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "sepal length (cm)": self.sepal_length,
                "sepal width (cm)": self.sepal_width,
                "petal length (cm)": self.petal_length,
                "petal width (cm)": self.petal_width
            }
            return pd.DataFrame([data_dict])
        except Exception as e:
            from src.exception import CustomException
            import sys
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join('artifacts', 'model.pkl')
            self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)

            logger.info("PredictPipeline initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df):
        try:
            # Convert input to DataFrame
            df = input_df
            logger.info(f"Input for prediction: {df.to_dict(orient='records')}")

            # Transform input using saved preprocessor
            transformed = self.preprocessor.transform(df)

            # Predict using saved model
            preds = self.model.predict(transformed)
            return preds[0]

        except Exception as e:
            raise CustomException(e, sys)