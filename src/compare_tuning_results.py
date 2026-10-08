import mlflow
from mlflow.tracking import MlflowClient


def compare_results():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        "iris-hyperparameter-tuning"
    )

    if experiment is None:
        print("Experiment not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
    )

    print()
    print("Hyperparameter Tuning Comparison")
    print("=" * 75)

    print(
        f"{'Run Name':<35}"
        f"{'CV f1_macro':<15}"
        f"{'Test Accuracy':<18}"
        f"{'Total Fits':<10}"
    )

    print("-" * 75)

    for run in runs:

        run_name = run.data.tags.get(
            "mlflow.runName",
            "unknown"
        )

        metrics = run.data.metrics
        params = run.data.params

        if "cv_f1_macro_mean" in metrics:
            cv_score = metrics["cv_f1_macro_mean"]
            total_fits = 5

        elif "best_cv_f1_macro" in metrics:
            cv_score = metrics["best_cv_f1_macro"]

            if "total_combinations" in params:
                total_fits = int(
                    params["total_combinations"]
                ) * 5

            elif "n_iter" in params:
                total_fits = int(
                    params["n_iter"]
                ) * 5

            else:
                total_fits = "N/A"

        else:
            continue

        test_accuracy = metrics.get(
            "test_accuracy",
            0
        )

        print(
            f"{run_name:<35}"
            f"{cv_score:<15.4f}"
            f"{test_accuracy:<18.4f}"
            f"{total_fits:<10}"
        )

    print("=" * 75)


if __name__ == "__main__":
    compare_results()