import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "iris-classification-baseline"
REGISTERED_MODEL_NAME = "iris-classifier-prod"

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

# Find the best run based on F1 score
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_macro DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No runs found.")

best_run = runs[0]

run_id = best_run.info.run_id
f1_score = best_run.data.metrics["f1_macro"]

print("Best run ID:", run_id)
print("Best F1 score:", f1_score)

# Model URI
model_uri = f"runs:/{run_id}/model"

print("Model URI:", model_uri)

# Register the model
result = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)

print("Registered model:", result.name)
print("Version:", result.version)

# Move model to Staging
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=result.version,
    stage="Staging"
)

print(f"Model version {result.version} moved to Staging.")