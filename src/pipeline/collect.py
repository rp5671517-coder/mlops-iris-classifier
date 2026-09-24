import csv
import logging
from datetime import datetime, timezone
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def collect_data(output_path="data/raw/iris_raw.csv"):
    iris = load_iris()

    species_map = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "species",
            "collected_at"
        ])

        for values, target in zip(iris.data, iris.target):
            writer.writerow([
                values[0],
                values[1],
                values[2],
                values[3],
                species_map[int(target)],
                datetime.now(timezone.utc).isoformat()
            ])

    logging.info(
        "Collected %d rows to %s",
        len(iris.data),
        output_path
    )


if __name__ == "__main__":
    collect_data()