from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Step 1: Ingest data
ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()

# Step 2: Transform data
transformer = DataTransformation()
X_train_scaled, X_test_scaled, y_train, y_test, _ = transformer.initiate_data_transformation(train_path, test_path)

# Step 3: Train models
trainer = ModelTrainer()
scores, model_path = trainer.initiate_model_training(X_train_scaled, X_test_scaled, y_train, y_test)

print("Model Scores:", scores)
print("Best model saved at:", model_path)