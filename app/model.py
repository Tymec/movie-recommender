from __future__ import annotations

import warnings

import numpy as np
import spacy
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from app.constants import CACHE_DIR

__all__ = ["create_model", "train_model", "evaluate_model"]

try:
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "ner"])
except OSError:
    print("Downloading spaCy model...")

    from spacy.cli import download as spacy_download

    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "ner"])


class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        character_threshold: int = 2,
        batch_size: int = 1024,
        n_jobs: int = 8,
        progress: bool = True,
    ) -> None:
        self.character_threshold = character_threshold
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.progress = progress

    def fit(self, _data: list[str], _labels: list[int] | None = None) -> TextTokenizer:
        return self

    def transform(self, data: list[str]) -> list[list[str]]:
        tokenized = []
        for doc in tqdm(
            nlp.pipe(data, batch_size=self.batch_size, n_process=self.n_jobs),
            total=len(data),
            disable=not self.progress,
        ):
            tokens = []
            for token in doc:
                # Ignore stop words and punctuation
                if token.is_stop or token.is_punct:
                    continue
                # Ignore emails, URLs and numbers
                if token.like_email or token.like_email or token.like_num:
                    continue

                # Lemmatize and lowercase
                tok = token.lemma_.lower().strip()

                # Format hashtags
                if tok.startswith("#"):
                    tok = tok[1:]

                # Ignore short and non-alphanumeric tokens
                if len(tok) < self.character_threshold or not tok.isalnum():
                    continue

                # TODO: Emoticons and emojis
                # TODO: Spelling correction

                tokens.append(tok)
            tokenized.append(tokens)
        return tokenized


def identity(x: list[str]) -> list[str]:
    """Identity function for use in TfidfVectorizer.

    Args:
        x: Input data

    Returns:
        Unchanged input data
    """
    return x


def create_model(
    max_features: int,
    seed: int | None = None,
    verbose: bool = False,
) -> Pipeline:
    """Create a sentiment analysis model.

    Args:
        max_features: Maximum number of features
        seed: Random seed (None for random seed)
        verbose: Whether to log progress during training

    Returns:
        Untrained model
    """
    return Pipeline(
        [
            ("tokenizer", TextTokenizer(progress=True)),
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    # disable text processing
                    tokenizer=identity,
                    preprocessor=identity,
                    lowercase=False,
                    token_pattern=None,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=seed)),
        ],
        memory=Memory(CACHE_DIR, verbose=0),
        verbose=verbose,
    )


def train_model(
    model: BaseEstimator,
    text_data: list[str],
    label_data: list[int],
    seed: int = 42,
) -> tuple[BaseEstimator, float]:
    """Train the sentiment analysis model.

    Args:
        model: Untrained model
        text_data: Text data
        label_data: Label data
        seed: Random seed (None for random seed)

    Returns:
        Trained model and accuracy
    """
    text_train, text_test, label_train, label_test = train_test_split(
        text_data,
        label_data,
        test_size=0.2,
        random_state=seed,
    )

    param_distributions = {
        "classifier__C": np.logspace(-4, 4, 20),
        "classifier__penalty": ["l1", "l2"],
    }

    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # model.fit(text_train, label_train)
        search.fit(text_train, label_train)

    best_model = search.best_estimator_
    return best_model, best_model.score(text_test, label_test)


def evaluate_model(
    model: Pipeline,
    text_data: list[str],
    label_data: list[int],
    folds: int = 5,
) -> tuple[float, float]:
    """Evaluate the model using cross-validation.

    Args:
        model: Trained model
        text_data: Text data
        label_data: Label data
        folds: Number of cross-validation folds

    Returns:
        Mean accuracy and standard deviation
    """
    scores = cross_val_score(
        model,
        text_data,
        label_data,
        cv=folds,
        scoring="accuracy",
    )
    return scores.mean(), scores.std()
