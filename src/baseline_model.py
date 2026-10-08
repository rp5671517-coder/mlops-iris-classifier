import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "data/processed/iris_features.csv"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["species"])

    X = df[FEATURE_COLUMNS].copy()
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def run_baseline():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset()

    model = DecisionTreeClassifier(random_state=42)

    with mlflow.start_run(run_name="baseline_decision_tree"):

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1_macro",
        )

        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        model.fit(X_train, y_train)

        test_accuracy = model.score(X_test, y_test)

        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_metric("cv_f1_macro_mean", cv_mean)
        mlflow.log_metric("cv_f1_macro_std", cv_std)
        mlflow.log_metric("test_accuracy", test_accuracy)

        print(
            f"Baseline CV f1_macro: "
            f"{cv_mean:.4f} (+/- {cv_std:.4f})"
        )

        print(
            f"Baseline test accuracy: "
            f"{test_accuracy:.4f}"
        )


if __name__ == "__main__":
    run_baseline()