"""Shared code for training and the Streamlit app.

Kept in its own module so the saved joblib pipeline can be unpickled
from any script (a class defined in __main__ would break loading).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Subject, Sender_Email and Email_ID carry no signal in this dataset
# (spam rate is ~30% in every subject / sender domain), so they are not used.
FEATURES = [
    "Email_Length",
    "Num_Links",
    "Num_Special_Chars",
    "Capital_Words",
    "Has_Attachment",
]
TARGET = "Spam"


class NegativeToNaN(BaseEstimator, TransformerMixin):
    """Counts and lengths can't be negative; treat such values as missing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[X < 0] = np.nan
        return X


def make_pipeline(model):
    return Pipeline(
        [
            ("negatives", NegativeToNaN()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
