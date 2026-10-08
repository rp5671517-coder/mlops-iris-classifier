import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder


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


def run_grid_search():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset()

    model = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=True,
    )

    with mlflow.start_run(run_name="grid_search_random_forest"):

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        best_cv_score = grid_search.best_score_
        test_accuracy = best_model.score(X_test, y_test)

        total_combinations = len(grid_search.cv_results_["params"])
        total_fits = total_combinations * 5

        mlflow.log_param("search_type", "GridSearchCV")
        mlflow.log_param("total_combinations", total_combinations)
        mlflow.log_param("cv_folds", 5)

        mlflow.log_metric("best_cv_f1_macro", best_cv_score)
        mlflow.log_metric("test_accuracy", test_accuracy)

        for key, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{key}", value)

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("grid_search_all_candidates.csv", index=False)

        mlflow.log_artifact("grid_search_all_candidates.csv")

        print(
            f"Grid Search evaluated "
            f"{total_combinations} combinations x 5 folds = "
            f"{total_fits} total fits"
        )

        print("Best params:")
        print(grid_search.best_params_)

        print(f"Best CV f1_macro: {best_cv_score:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    run_grid_search()