import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------
# 1. Set MLflow tracking URI
# ---------------------------------------------------------
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

# Create/select MLflow experiment
mlflow.set_experiment("iris-classification-baseline")


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# 3. Select features and target
# ---------------------------------------------------------
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio"
]

TARGET = "species"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Handle missing values
X = X.fillna(X.median())

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# ---------------------------------------------------------
# 4. Split dataset
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------
models = [
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=200,
            C=1.0
        )
    ),
    (
        "random_forest_shallow",
        RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
    ),
    (
        "random_forest_deep",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )
    )
]


# ---------------------------------------------------------
# 6. Train and track models using MLflow
# ---------------------------------------------------------
results = []

for model_name, model in models:

    with mlflow.start_run(run_name=model_name):

        print("\nTraining:", model_name)

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        # Log parameters
        if model_name == "logistic_regression":
            mlflow.log_param("model_type", "logistic_regression")
            mlflow.log_param("max_iter", 200)
            mlflow.log_param("C", 1.0)

        else:
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()

        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        plt.tight_layout()

        cm_file = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_file)
        plt.close()

        # Log confusion matrix artifact
        mlflow.log_artifact(cm_file)

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # Save result
        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1
            }
        )

        print("Accuracy :", f"{accuracy:.4f}")
        print("Precision:", f"{precision:.4f}")
        print("Recall   :", f"{recall:.4f}")
        print("F1 Score :", f"{f1:.4f}")


# ---------------------------------------------------------
# 7. Compare models
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

best_model = results_df.loc[
    results_df["f1_macro"].idxmax()
]

print("\nBest model based on F1 score:")
print(best_model)