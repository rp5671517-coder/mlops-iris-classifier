from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "digits.csv"

def main():
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["feature_pipeline"]

    output = ROOT / params["output_file"]
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW)
    pixel_cols = [f"pixel_{i}" for i in range(64)]

    expected = {"digit_id", "label", *pixel_cols}
    missing = expected - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["digit_id"].duplicated().any():
        raise ValueError("digit_id values must be unique")

    if df["label"].isna().any() or not df["label"].between(0, 9).all():
        raise ValueError("label must be an integer from 0 to 9")

    features = df[["digit_id"] + pixel_cols + ["label"]].copy()

    max_value = float(params["pixel_max_value"])
    features[pixel_cols] = features[pixel_cols].astype("float32") / max_value
    features["label"] = features["label"].astype("int64")

    features["event_timestamp"] = pd.Timestamp("2024-01-01T00:00:00Z")

    features.to_parquet(output, index=False)

    check = pd.read_parquet(output)
    assert len(check) == 1797
    assert check[pixel_cols].min().min() >= 0.0
    assert check[pixel_cols].max().max() <= 1.0
    assert check["digit_id"].is_unique

    print(f"Feature table saved: {output.relative_to(ROOT)}")
    print(f"Rows: {len(check)} | Features: {len(pixel_cols)}")
    print("Feature validation: PASSED")

if __name__ == "__main__":
    main()
