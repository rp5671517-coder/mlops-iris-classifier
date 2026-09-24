import csv
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_features(
    input_path="data/processed/iris_preprocessed.csv",
    output_path="data/processed/iris_features.csv"
):
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        sepal_length = float(row["sepal length (cm)"])
        sepal_width = float(row["sepal width (cm)"])
        petal_length = float(row["petal length (cm)"])
        petal_width = float(row["petal width (cm)"])

        # Feature 1
        row["sepal_area"] = sepal_length * sepal_width

        # Feature 2
        row["petal_area"] = petal_length * petal_width

        # Feature 3
        if petal_length != 0:
            row["sepal_to_petal_length_ratio"] = (
                sepal_length / petal_length
            )
        else:
            row["sepal_to_petal_length_ratio"] = 0

        # Feature 4
        if petal_length <= 2:
            row["petal_length_bin"] = "short"
        elif petal_length <= 4.5:
            row["petal_length_bin"] = "medium"
        else:
            row["petal_length_bin"] = "long"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
        "sepal_area",
        "petal_area",
        "sepal_to_petal_length_ratio",
        "petal_length_bin"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(
        "Feature engineering complete: %d rows, %d columns",
        len(rows),
        len(fieldnames)
    )


if __name__ == "__main__":
    create_features()