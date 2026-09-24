import csv
import logging
import os
import statistics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def preprocess_data(
    input_path="data/raw/iris_raw.csv",
    output_path="data/processed/iris_preprocessed.csv"
):
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Remove exact duplicate rows
    unique_rows = []
    seen = set()

    for row in rows:
        row_tuple = tuple(row.items())

        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)

    rows = unique_rows

    numeric_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    # Convert numeric values to float
    for row in rows:
        for column in numeric_columns:
            try:
                row[column] = float(row[column])
            except (ValueError, TypeError):
                row[column] = None

    # Fill missing numeric values with median
    for column in numeric_columns:
        values = [
            row[column]
            for row in rows
            if row[column] is not None
        ]

        if values:
            median_value = statistics.median(values)

            for row in rows:
                if row[column] is None:
                    row[column] = median_value

    # Remove rows with missing species
    rows = [
        row for row in rows
        if row.get("species") not in (None, "")
    ]

    # Remove collected_at column
    for row in rows:
        row.pop("collected_at", None)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(
        "Preprocessing complete: %d rows written to %s",
        len(rows),
        output_path
    )


if __name__ == "__main__":
    preprocess_data()