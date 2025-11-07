from src.pipeline.predict_pipeline import PredictPipeline

sample_input = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}

pipeline = PredictPipeline()
prediction = pipeline.predict(sample_input)
print("Predicted class:", prediction)
