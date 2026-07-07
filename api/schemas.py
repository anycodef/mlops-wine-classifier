"""Request/response schemas for the prediction API.

The model was trained on a DataFrame whose column names come straight from
scikit-learn's Wine dataset (one of them contains a ``/``). The request model
uses Python-safe field names and maps them back to the canonical column order
expected by the packaged model's signature.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

# Canonical column names/order the trained model expects.
CANONICAL_COLUMNS: List[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]


class WineSample(BaseModel):
    alcohol: float = Field(..., examples=[14.23])
    malic_acid: float = Field(..., examples=[1.71])
    ash: float = Field(..., examples=[2.43])
    alcalinity_of_ash: float = Field(..., examples=[15.6])
    magnesium: float = Field(..., examples=[127.0])
    total_phenols: float = Field(..., examples=[2.8])
    flavanoids: float = Field(..., examples=[3.06])
    nonflavanoid_phenols: float = Field(..., examples=[0.28])
    proanthocyanins: float = Field(..., examples=[2.29])
    color_intensity: float = Field(..., examples=[5.64])
    hue: float = Field(..., examples=[1.04])
    od280_od315_of_diluted_wines: float = Field(..., examples=[3.92])
    proline: float = Field(..., examples=[1065.0])

    def to_row(self) -> dict:
        """Map safe field names to the model's canonical column names."""
        values = [
            self.alcohol, self.malic_acid, self.ash, self.alcalinity_of_ash,
            self.magnesium, self.total_phenols, self.flavanoids,
            self.nonflavanoid_phenols, self.proanthocyanins, self.color_intensity,
            self.hue, self.od280_od315_of_diluted_wines, self.proline,
        ]
        return dict(zip(CANONICAL_COLUMNS, values))


# Python-safe field names, in the same order as CANONICAL_COLUMNS.
SAFE_FIELDS: List[str] = list(WineSample.model_fields.keys())


class PredictRequest(BaseModel):
    samples: List[WineSample]


class PredictResponse(BaseModel):
    predictions: List[int]
    model_alias: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap: float


class ExplainResponse(BaseModel):
    prediction: int
    probabilities: List[float]
    contributions: List[FeatureContribution]
    model_alias: str
