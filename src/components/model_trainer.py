import os
import sys
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger.logger_config import logger

class ModelTrainer:
    def __init__(self):
        self.model_file_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logger.info("Model training started")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor()
            }

            model_scores = {}
            best_model = None
            best_score = -1

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_scores[name] = score
                logger.info(f"{name} R2 Score: {score}")

                if score > best_score:
                    best_score = score
                    best_model = model

            # Save best model
            with open(self.model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            logger.info(f"Best model saved: {best_model.__class__.__name__} with R2 score {best_score}")
            return model_scores, self.model_file_path

        except Exception as e:
            raise CustomException(e, sys)