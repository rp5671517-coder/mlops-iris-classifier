import mlflow
import mlflow.sklearn

MODEL_URI = "models:/iris-classifier-prod/Staging"

print("Loading registered model...")
print("Model URI:", MODEL_URI)

model = mlflow.sklearn.load_model(MODEL_URI)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)