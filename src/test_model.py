import mlflow
import mlflow.sklearn
import pandas as pd

# Load registered model Version 1
model_uri = "models:/iris-classifier-prod/1"

print("Loading model...")

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully!")

# Use the SAME features used during training
sample = pd.DataFrame([{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "sepal_area": 5.1 * 3.5,
    "petal_area": 1.4 * 0.2
}])

# Keep only the features expected by the model
expected_features = model.feature_names_in_

sample = sample.reindex(columns=expected_features, fill_value=0)

# Prediction
prediction = model.predict(sample)

print("\nInput data:")
print(sample)

print("\nPredicted class:")
print(prediction)
