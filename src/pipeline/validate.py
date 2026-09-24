import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


EXPECTED_COLUMNS = [
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

VALID_SPECIES = {
    "setosa",
    "versicolor",
    "virginica"
}


def validate_data(input_path="data/processed/iris_features.csv"):
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = list(reader)
        actual_columns = reader.fieldnames

    errors = []

    # Check columns
    if actual_columns != EXPECTED_COLUMNS:
        errors.append(
            f"Expected columns {EXPECTED_COLUMNS}, "
            f"but found {actual_columns}"
        )

    # Validate each row
    for row_number, row in enumerate(rows, start=2):

        # Check species
        if row["species"] not in VALID_SPECIES:
            errors.append(
                f"Row {row_number}: invalid species "
                f"{row['species']}"
            )

        # Check numeric values
        numeric_ranges = {
            "sepal length (cm)": (3, 9),
            "sepal width (cm)": (1.5, 5.5),
            "petal length (cm)": (0.5, 8),
            "petal width (cm)": (0.05, 3)
        }

        for column, (minimum, maximum) in numeric_ranges.items():
            try:
                value = float(row[column])

                if not minimum <= value <= maximum:
                    errors.append(
                        f"Row {row_number}: {column} "
                        f"value {value} outside range"
                    )

            except (ValueError, TypeError):
                errors.append(
                    f"Row {row_number}: invalid value in {column}"
                )

        # Check null/empty values
        for column in EXPECTED_COLUMNS:
            if row[column] is None or row[column] == "":
                errors.append(
                    f"Row {row_number}: missing value in {column}"
                )

    if errors:
        for error in errors:
            logging.error(error)

        raise ValueError(
            f"Validation FAILED with {len(errors)} error(s)"
        )

    logging.info(
        "Validation PASSED: %d rows, %d columns, "
        "all checks satisfied",
        len(rows),
        len(actual_columns)
    )


if __name__ == "__main__":
    validate_data()