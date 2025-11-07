from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Mapping numeric predictions to flower names
flower_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extract data from the form
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))

        # Prepare custom data
        data = CustomData(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )

        pred_df = data.get_data_as_data_frame()

        # Run prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Map numeric prediction to flower name
        results_name = flower_mapping.get(results, "Unknown")

        return render_template('home.html', results=results_name)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)