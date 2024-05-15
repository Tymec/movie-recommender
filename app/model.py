from __future__ import annotations

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from constants import CLF_MAX_ITER, MAX_TOKENIZER_FEATURES
from utils import get_cache_memory, get_random_state

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import ndarray
    from numpy.random import RandomState


__all__ = ["predict", "tokenize"]


@lru_cache(maxsize=1)
def get_model(model_path: Path) -> Pipeline:
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_tokenizer(tokenizer_path: Path) -> Pipeline:
    return joblib.load(tokenizer_path)


def export_to_file(pipeline: Pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)


def tokenize(text: str, tokenizer_path: Path) -> ndarray:
    tokenizer = get_tokenizer(tokenizer_path)
    return tokenizer.transform([text])[0]


def predict(tokens: ndarray, model_path: Path) -> bool:
    model = get_model(model_path)
    prediction = model.predict([tokens])
    return prediction[0] == 1


def train_and_export(
    steps: Sequence[tuple],
    x: list[str],
    y: list[int],
    export_path: Path,
    cache: joblib.Memory,
) -> Pipeline:
    pipeline = Pipeline(steps, memory=cache)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(x, y)

    export_to_file(pipeline, export_path)
    return pipeline


def train_tokenizer_and_export(x: list[str], y: list[int], export_path: Path, cache: joblib.Memory) -> Pipeline:
    return train_and_export(
        [
            (
                "vectorize",
                CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=MAX_TOKENIZER_FEATURES),
            ),
            ("tfidf", TfidfTransformer()),
        ],
        x,
        y,
        export_path,
        cache,
    )


def train_model_and_export(
    x: ndarray,
    y: list[int],
    export_path: Path,
    cache: joblib.Memory,
    rs: RandomState,
) -> Pipeline:
    return train_and_export(
        [("clf", LogisticRegression(max_iter=CLF_MAX_ITER, random_state=rs))],
        x,
        y,
        export_path,
        cache,
    )


def train(x: list[str], y: list[int]) -> Pipeline:
    cache = get_cache_memory()
    rs = get_random_state()

    tokenizer = train_tokenizer(x, y, cache)
    x_tr = tokenizer.transform(x)

    model = train_model(x_tr, y, cache, rs)

    return Pipeline([("tokenizer", tokenizer), ("model", model)])


def train_tokenizer(x: list[str], y: list[int], cache: joblib.Memory) -> Pipeline:
    # TODO: In the future, allow for different tokenizers
    pipeline = Pipeline(
        [
            (
                "vectorize",
                CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=MAX_TOKENIZER_FEATURES),
            ),
            ("tfidf", TfidfTransformer()),
        ],
        memory=cache,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore joblib warnings
        pipeline.fit(x, y)

    return pipeline


def train_model(x: list[str], y: list[int], cache: joblib.Memory, rs: RandomState) -> Pipeline:
    # TODO: In the future, allow for different classifiers
    pipeline = Pipeline(
        [
            ("clf", LogisticRegression(max_iter=CLF_MAX_ITER, random_state=rs)),
        ],
        memory=cache,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore joblib warnings
        pipeline.fit(x, y)

    return pipeline
