from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import joblib

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline


class Model(ABC):
    """Base class for all models"""

    @property
    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Pipeline used for the model"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the architecture"""
        ...

    @abstractmethod
    def _predict(self, text: str) -> int:
        """Predict the sentiment of the given text"""
        ...

    @staticmethod
    def from_file(path: Path) -> Model:
        """Load the model from the given file"""
        return joblib.load(path)

    def to_file(self, path: Path) -> None:
        """Save the model to the given file"""
        joblib.dump(self, path)

    def predict(self, text: str) -> int:
        """Perform sentiment analysis on the given text"""
        return self._predict(text)

    def train(self, x: list[str], y: list[int]) -> None:
        """Train the model on the given data"""
        self.pipeline.fit(x, y)
