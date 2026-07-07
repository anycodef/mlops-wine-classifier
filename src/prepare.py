"""Prepare stage: load the raw dataset and produce a reproducible train/test split.

The Wine dataset ships inside scikit-learn, so there is no external raw file to
version; instead this stage materialises the *processed* split as CSV, and DVC
tracks those outputs so the rest of the pipeline is fully reproducible.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.config import TARGET, TEST_CSV, TRAIN_CSV, ensure_dirs, load_params


def main() -> None:
    ensure_dirs()
    params = load_params()["data"]

    wine = load_wine(as_frame=True)
    frame = wine.frame  # 13 feature columns + a "target" column

    train_df, test_df = train_test_split(
        frame,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=frame[TARGET],
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"[prepare] train={len(train_df)} rows -> {TRAIN_CSV}")
    print(f"[prepare] test ={len(test_df)} rows -> {TEST_CSV}")


if __name__ == "__main__":
    main()
