"""Data loading, type conversion, and the EDA missing-charge assumption."""
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def convert_to_boolean(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with the specified Yes/No or 0/1 columns converted."""
    result = df.copy()
    mapping = {"Yes": True, "No": False, 0: False, 1: True}
    for column in columns:
        invalid = ~result[column].isin(mapping)
        if invalid.any():
            unexpected = result.loc[invalid, column].unique().tolist()
            raise ValueError(f"Unexpected values in {column!r}: {unexpected}")
        result[column] = result[column].map(mapping).astype(bool)
    return result


def convert_to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with blanks treated as missing and other values parsed."""
    result = df.copy()
    for column in columns:
        values = result[column].replace(r"^\s*$", np.nan, regex=True)
        result[column] = pd.to_numeric(values, errors="raise")
    return result
