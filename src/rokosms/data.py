from pathlib import Path

import pandas as pd

from .text_utils import mask_phones

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def load_train():
    return pd.read_csv(PROCESSED / "pipeline_train.csv")


def load_dev():
    return pd.read_csv(PROCESSED / "pipeline_dev.csv")


def split_xy(df, mask_phone=False):
    X = df["text"]
    if mask_phone:
        X = X.apply(mask_phones)
    y = df["Labels"]
    groups = df["template_key"]
    return X, y, groups
