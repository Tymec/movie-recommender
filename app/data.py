from __future__ import annotations

import bz2
from typing import Literal

import pandas as pd

from app.constants import (
    AMAZONREVIEWS_PATH,
    AMAZONREVIEWS_URL,
    IMDB50K_PATH,
    IMDB50K_URL,
    SENTIMENT140_PATH,
    SENTIMENT140_URL,
)

__all__ = ["load_data"]


def load_sentiment140(include_neutral: bool = False) -> tuple[list[str], list[int]]:
    """Load the sentiment140 dataset and make it suitable for use.

    Args:
        include_neutral: Whether to include neutral sentiment

    Returns:
        Text and label data

    Raises:
        FileNotFoundError: If the dataset is not found
    """
    # Check if the dataset exists
    if not SENTIMENT140_PATH.exists():
        msg = (
            f"Sentiment140 dataset not found at: '{SENTIMENT140_PATH}'\n"
            "Please download the dataset from:\n"
            f"{SENTIMENT140_URL}"
        )
        raise FileNotFoundError(msg)

    # Load the dataset
    data = pd.read_csv(
        SENTIMENT140_PATH,
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
    if not include_neutral:
        data = data[data["target"] != 2]

    # Map sentiment values
    data["sentiment"] = data["target"].map(
        {
            0: 0,  # Negative
            4: 1,  # Positive
            2: 2,  # Neutral
        },
    )

    # Return as lists
    return data["text"].tolist(), data["sentiment"].tolist()


def load_amazonreviews(merge: bool = True) -> tuple[list[str], list[int]]:
    """Load the amazonreviews dataset and make it suitable for use.

    Args:
        merge: Whether to merge the test and train datasets (otherwise ignore test)

    Returns:
        Text and label data

    Raises:
        FileNotFoundError: If the dataset is not found
    """
    # Check if the dataset exists
    test_exists = AMAZONREVIEWS_PATH[0].exists() or not merge
    train_exists = AMAZONREVIEWS_PATH[1].exists()
    if not (test_exists and train_exists):
        msg = (
            f"Amazonreviews dataset not found at: '{AMAZONREVIEWS_PATH[0]}' and '{AMAZONREVIEWS_PATH[1]}'\n"
            "Please download the dataset from:\n"
            f"{AMAZONREVIEWS_URL}"
        )
        raise FileNotFoundError(msg)

    # Load the datasets
    dataset = []
    with bz2.BZ2File(AMAZONREVIEWS_PATH[1]) as train_file:
        dataset.extend([line.decode("utf-8") for line in train_file])

    if merge:
        with bz2.BZ2File(AMAZONREVIEWS_PATH[0]) as test_file:
            dataset.extend([line.decode("utf-8") for line in test_file])

    # Split the data into labels and text
    labels, texts = zip(*(line.split(" ", 1) for line in dataset))  # NOTE: Occasionally OOM

    # Free up memory
    del dataset

    # Map sentiment values
    sentiments = [int(label.split("__label__")[1]) - 1 for label in labels]

    # Return as lists
    return texts, sentiments


def load_imdb50k() -> tuple[list[str], list[int]]:
    """Load the imdb50k dataset and make it suitable for use.

    Returns:
        Text and label data

    Raises:
        FileNotFoundError: If the dataset is not found
    """
    # Check if the dataset exists
    if not IMDB50K_PATH.exists():
        msg = (
            f"IMDB50K dataset not found at: '{IMDB50K_PATH}'\n"
            "Please download the dataset from:\n"
            f"{IMDB50K_URL}"
        )  # fmt: off
        raise FileNotFoundError(msg)

    # Load the dataset
    data = pd.read_csv(IMDB50K_PATH)

    # Map sentiment values
    data["sentiment"] = data["sentiment"].map(
        {
            "positive": 1,
            "negative": 0,
        },
    )

    # Return as lists
    return data["review"].tolist(), data["sentiment"].tolist()


def load_data(dataset: Literal["sentiment140", "amazonreviews", "imdb50k"]) -> tuple[list[str], list[int]]:
    """Load and preprocess the specified dataset.

    Args:
        dataset: Dataset to load

    Returns:
        Text and label data

    Raises:
        ValueError: If the dataset is not recognized
    """
    match dataset:
        case "sentiment140":
            return load_sentiment140(include_neutral=False)
        case "amazonreviews":
            return load_amazonreviews(merge=True)
        case "imdb50k":
            return load_imdb50k()
        case _:
            msg = f"Unknown dataset: {dataset}"
            raise ValueError(msg)
