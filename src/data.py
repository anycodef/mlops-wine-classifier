"""Small helpers shared by the training / evaluation stages."""
from __future__ import annotations

import pandas as pd

from src.config import TARGET, TEST_CSV, TRAIN_CSV


def load_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def load_train():
    return load_split(TRAIN_CSV)


def load_test():
    return load_split(TEST_CSV)
