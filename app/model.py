from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from joblib import Memory
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from app.constants import CACHE_DIR
from app.data import tokenize

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["train_model", "evaluate_model", "infer_model"]


def _identity(x: list[str]) -> list[str]:
    """Identity function for use in TfidfVectorizer.

    Args:
        x: Input data

    Returns:
        Unchanged input data
    """
    return x


def _get_vectorizer(
    name: Literal["tfidf", "count", "hashing"],
    n_features: int,
    df: tuple[float, float] = (0.1, 0.9),
    ngram: tuple[int, int] = (1, 2),
) -> TransformerMixin:
    """Get the appropriate vectorizer.

    Args:
        name: Type of vectorizer
        n_features: Maximum number of features
        df: Document frequency range [min_df, max_df] (ignored for HashingVectorizer)
        ngram: N-gram range [min_n, max_n]

    Returns:
        Vectorizer instance

    Raises:
        ValueError: If the vectorizer is not recognized
    """
    shared_params = {
        "ngram_range": ngram,
        # disable text processing
        "tokenizer": _identity,
        "preprocessor": _identity,
        "lowercase": False,
        "token_pattern": None,
    }

    match name:
        case "tfidf":
            return TfidfVectorizer(
                max_features=n_features,
                min_df=df[0],
                max_df=df[1],
                **shared_params,
            )
        case "count":
            return CountVectorizer(
                max_features=n_features,
                min_df=df[0],
                max_df=df[1],
                **shared_params,
            )
        case "hashing":
            if n_features < 2**15:
                warnings.warn(
                    "HashingVectorizer may perform poorly with small n_features, default is 2^20.",
                    stacklevel=2,
                )

            return HashingVectorizer(
                n_features=n_features,
                **shared_params,
            )
        case _:
            msg = f"Unknown vectorizer: {name}"
            raise ValueError(msg)


def train_model(
    token_data: Sequence[Sequence[str]],
    label_data: list[int],
    vectorizer: Literal["tfidf", "count", "hashing"],
    max_features: int,
    folds: int = 5,
    n_jobs: int = 4,
    seed: int = 42,
) -> tuple[BaseEstimator, float]:
    """Train the sentiment analysis model.

    Args:
        token_data: Tokenized text data
        label_data: Label data
        vectorizer: Which vectorizer to use
        max_features: Maximum number of features
        folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        seed: Random seed (None for random seed)

    Returns:
        Trained model and accuracy

    Raises:
        ValueError: If the vectorizer is not recognized
    """
    rs = None if seed == -1 else seed

    text_train, text_test, label_train, label_test = train_test_split(
        token_data,
        label_data,
        test_size=0.2,
        random_state=rs,
    )

    vectorizer = _get_vectorizer(vectorizer, max_features)
    classifiers = [
        (LogisticRegression(max_iter=1000, random_state=rs), {"C": np.logspace(-4, 4, 20)}),
        # (LinearSVC(max_iter=10000, random_state=rs), {"C": np.logspace(-4, 4, 20)}),
        # (KNeighborsClassifier(), {"n_neighbors": np.arange(1, 10)}),
        # (RandomForestClassifier(random_state=rs), {"n_estimators": np.arange(50, 500, 50)}),
        # (
        #     VotingClassifier(
        #         estimators=[
        #             ("lr", LogisticRegression(max_iter=1000, random_state=rs)),
        #             ("knn", KNeighborsClassifier()),
        #             ("rf", RandomForestClassifier(random_state=rs)),
        #         ],
        #     ),
        #     {
        #         "lr__C": np.logspace(-4, 4, 20),
        #         "knn__n_neighbors": np.arange(1, 10),
        #         "rf__n_estimators": np.arange(50, 500, 50),
        #     },
        # ),
    ]

    models = []
    for clf, param_dist in (pbar := tqdm(classifiers, unit="clf")):
        param_dist = {f"classifier__{k}": v for k, v in param_dist.items()}

        model = Pipeline(
            [("vectorizer", vectorizer), ("classifier", clf)],
            memory=Memory(CACHE_DIR, verbose=0),
        )

        search = RandomizedSearchCV(
            model,
            param_dist,
            cv=folds,
            random_state=rs,
            n_jobs=n_jobs,
            # verbose=2,
            scoring="accuracy",
            n_iter=7,
        )

        pbar.set_description(f"Searching for {clf.__class__.__name__}")

        with warnings.catch_warnings():
            warnings.filterwarnings("once", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning, message="Persisting input arguments took")

            search.fit(text_train, label_train)

        best_model = search.best_estimator_
        acc = best_model.score(text_test, label_test)
        models.append((best_model, acc))

    print("Final results:")
    print("--------------")
    print("\n".join(f"{model.named_steps['classifier'].__class__.__name__}: {acc:.2%}" for model, acc in models))

    best_model, best_acc = max(models, key=lambda x: x[1])
    print(f"Settled on {best_model.named_steps['classifier'].__class__.__name__}")
    return best_model, best_acc


def evaluate_model(
    model: BaseEstimator,
    token_data: Sequence[Sequence[str]],
    label_data: list[int],
    folds: int = 5,
    n_jobs: int = 4,
) -> tuple[float, float]:
    """Evaluate the model using cross-validation.

    Args:
        model: Trained model
        token_data: Tokenized text data
        label_data: Label data
        folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Mean accuracy and standard deviation
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scores = cross_val_score(
            model,
            token_data,
            label_data,
            cv=folds,
            scoring="accuracy",
            n_jobs=n_jobs,
            verbose=2,
        )
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
