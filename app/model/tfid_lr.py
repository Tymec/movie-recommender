from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base import Model


class TfidfLR(Model):
    """Sentiment analysis model using TF-IDF and Logistic Regression"""

    def __init__(self):
        self._pipeline = Pipeline(
            [
                (
                    "vectorize",
                    CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000),
                ),
                ("tfidf", TfidfTransformer()),
                ("clf", LogisticRegression(max_iter=1000, random_state=self.rng)),
            ],
            memory=self.cache,
        )

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def description(self) -> str:
        return "TF-IDF with Logistic Regression"

    def _predict(self, text: str) -> int:
        return self.pipeline.predict([text])[0]
