from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.constants import CACHE_DIR
from app.data import tokenize

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

__all__ = ["train_model", "evaluate_model", "infer_model"]


def _identity(x: list[str]) -> list[str]:
    """Identity function for use in TfidfVectorizer.

    Args:
        x: Input data

    Returns:
        Unchanged input data
    """
    return x


def train_model(
    token_data: list[str],
    label_data: list[int],
    max_features: int,
    folds: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[BaseEstimator, float]:
    """Train the sentiment analysis model.

    Args:
        model: Untrained model
        token_data: Tokenized text data
        label_data: Label data
        max_features: Maximum number of features
        folds: Number of cross-validation folds
        seed: Random seed (None for random seed)
        verbose: Whether to output additional information

    Returns:
        Trained model and accuracy
    """
    text_train, text_test, label_train, label_test = train_test_split(
        token_data,
        label_data,
        test_size=0.2,
        random_state=seed,
    )

    param_distributions = {
        "classifier__C": np.logspace(-4, 4, 20),
        "classifier__solver": ["liblinear", "saga"],
    }

    model = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    # disable text processing
                    tokenizer=_identity,
                    preprocessor=_identity,
                    lowercase=False,
                    token_pattern=None,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=None if seed == -1 else seed,
                ),
            ),
        ],
        memory=Memory(CACHE_DIR, verbose=0),
        verbose=verbose,
    )

    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=10,
        cv=folds,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1,
        verbose=verbose,
    )

    # os.environ["PYTHONWARNINGS"] = "ignore"
    search.fit(text_train, label_train)
    # del os.environ["PYTHONWARNINGS"]

    best_model = search.best_estimator_
    return best_model, best_model.score(text_test, label_test)


def evaluate_model(
    model: BaseEstimator,
    token_data: list[str],
    label_data: list[int],
    folds: int = 5,
    verbose: bool = False,
) -> tuple[float, float]:
    """Evaluate the model using cross-validation.

    Args:
        model: Trained model
        token_data: Tokenized text data
        label_data: Label data
        folds: Number of cross-validation folds
        verbose: Whether to output additional information

    Returns:
        Mean accuracy and standard deviation
    """
    os.environ["PYTHONWARNINGS"] = "ignore"
    scores = cross_val_score(
        model,
        token_data,
        label_data,
        cv=folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=verbose,
    )
    del os.environ["PYTHONWARNINGS"]
    return scores.mean(), scores.std()


def infer_model(
    model: BaseEstimator,
    text_data: list[str],
    batch_size: int = 32,
    n_jobs: int = 4,
) -> list[int]:
    """Predict the sentiment of the provided text documents.

    Args:
        model: Trained model
        text_data: Text data
        batch_size: Batch size for tokenization
        n_jobs: Number of parallel jobs

    Returns:
        Predicted sentiments
    """
    tokens = tokenize(
        text_data,
        batch_size=batch_size,
        n_jobs=n_jobs,
        show_progress=False,
    )
    return model.predict(tokens)
