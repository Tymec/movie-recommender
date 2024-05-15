from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import click
import joblib
import pandas as pd
from numpy.random import RandomState
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

SEED = 42
DATASET_PATH = Path("data/training.1600000.processed.noemoticon.csv")
STOPWORDS_PATH = Path("data/stopwords-en.txt")
CHECKPOINT_PATH = Path("cache/pipeline.pkl")
MODELS_DIR = Path("models")
CACHE_DIR = Path("cache")
MAX_FEATURES = 10000  # 500000

# Make sure paths exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Memory cache for sklearn pipelines
mem = joblib.Memory(CACHE_DIR, verbose=0)

# TODO: use xgboost


def get_random_state(seed: int = SEED) -> RandomState:
    return RandomState(seed)


def load_data() -> tuple[list[str], list[int]]:
    """The model takes in a list of strings and a list of integers where 1 is positive sentiment and 0 is negative sentiment."""
    data = pd.read_csv(
        DATASET_PATH,
        encoding="ISO-8859-1",
        names=[
            "target",  # 0 = negative, 2 = neutral, 4 = positive
            "id",  # The id of the tweet
            "date",  # The date of the tweet
            "flag",  # The query, NO_QUERY if not present
            "user",  # The user that tweeted
            "text",  # The text of the tweet
        ],
    )

    # Ignore rows with neutral sentiment
    data = data[data["target"] != 2]

    # Create new column called "sentiment" with 1 for positive and 0 for negative
    data["sentiment"] = data["target"] == 4

    # Drop the columns we don't need
    # data = data.drop(columns=["target", "id", "date", "flag", "user"]) # NOTE: No need, since we return the columns we need

    # Return as lists
    return list(data["text"]), list(data["sentiment"])


def create_pipeline(clf: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            # Preprocess
            # ("vectorize", CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=MAX_FEATURES)),
            # ("tfidf", TfidfTransformer()),
            ("vectorize", TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)),
            # Classifier
            ("clf", clf),
        ],
        memory=mem,
    )


def evaluate_pipeline(pipeline: Pipeline, x: list[str], y: list[int]) -> float:
    y_pred = pipeline.predict(x)
    report = classification_report(y, y_pred)
    click.echo(report)

    # TODO: Confusion matrix

    return accuracy_score(y, y_pred)


def export_pipeline(pipeline: Pipeline, name: str) -> None:
    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(pipeline, model_path)
    click.echo(f"Model exported to {model_path!r}")


@click.command()
@click.option("--retrain", is_flag=True, help="Train the model even if a checkpoint exists.")
@click.option("--evaluate", is_flag=True, help="Evaluate the model.")
@click.option("--flush-cache", is_flag=True, help="Clear sklearn cache.")
@click.option("--seed", type=int, default=SEED, help="Random seed.")
def train(retrain: bool, evaluate: bool, flush_cache: bool, seed: int) -> None:
    rng = get_random_state(seed)

    # Clear sklearn cache
    if flush_cache:
        click.echo("Clearing cache... ", nl=False)
        mem.clear(warn=False)
        click.echo("DONE")

    # Load and split data
    click.echo("Loading data... ", nl=False)
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rng)
    click.echo("DONE")

    # Train model
    if retrain or not CHECKPOINT_PATH.exists():
        click.echo("Training model... ", nl=False)
        clf = LogisticRegression(max_iter=1000, random_state=rng)
        model = create_pipeline(clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore joblib warnings
            model.fit(x_train, y_train)
        joblib.dump(model, CHECKPOINT_PATH)
        click.echo("DONE")
    else:
        click.echo("Loading model... ", nl=False)
        model = joblib.load(CHECKPOINT_PATH)
        click.echo("DONE")

    # Evaluate model
    if evaluate:
        evaluate_pipeline(model, x_test, y_test)

    # Quick test
    test_text = ["I love this movie", "I hate this movie"]
    click.echo("Quick test:")
    for text in test_text:
        click.echo(f"\t{'positive' if model.predict([text])[0] else 'negative'}: {text}")

    # Export model
    click.echo("Exporting model... ", nl=False)
    export_pipeline(model, "logistic_regression")
    click.echo("DONE")


if __name__ == "__main__":
    train()
