from __future__ import annotations

import bz2
import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Sequence

import emoji
import pandas as pd
import spacy
from tqdm import tqdm

from app.constants import (
    AMAZONREVIEWS_PATH,
    AMAZONREVIEWS_URL,
    IMDB50K_PATH,
    IMDB50K_URL,
    SENTIMENT140_PATH,
    SENTIMENT140_URL,
    SLANGMAP_PATH,
    SLANGMAP_URL,
    TEST_DATASET_PATH,
    TEST_DATASET_URL,
)

if TYPE_CHECKING:
    from re import Pattern

    from spacy.tokens import Doc

__all__ = ["load_data", "tokenize"]


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")

    from spacy.cli import download as spacy_download

    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def slang() -> tuple[Pattern, dict[str, str]]:
    """Compile a re pattern for slang terms.

    Returns:
        Slang pattern and mapping

    Raises:
        FileNotFoundError: If the file is not found
    """
    if not SLANGMAP_PATH.exists():
        # msg = f"Missing slang mapping file: {SLANG_PATH}"
        msg = (
            f"Slang mapping file not found at: '{SLANGMAP_PATH}'\n"
            "Please download the file from:\n"
            f"{SLANGMAP_URL}"
        )  # fmt: off
        raise FileNotFoundError(msg)

    with SLANGMAP_PATH.open() as f:
        mapping = json.load(f)

    return re.compile(r"\b(" + "|".join(map(re.escape, mapping.keys())) + r")\b"), mapping


def _clean(text: str) -> str:
    """Perform basic text cleaning.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Make text lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", "", text)

    # Map slang terms
    slang_pattern, slang_mapping = slang()
    text = slang_pattern.sub(lambda x: slang_mapping[x.group()], text)

    # Remove acronyms and abbreviations
    # text = re.sub(r"(?:[a-z]\.){2,}", "", text)
    text = re.sub(r"\b(?:[a-z]\.?)(?:[a-z]\.)\b", "", text)

    # Remove honorifics
    text = re.sub(r"\b(?:mr|mrs|ms|dr|prof|sr|jr)\.?\b", "", text)

    # Remove year abbreviations
    text = re.sub(r"\b(?:\d{3}0|\d0)s?\b", "", text)

    # Remove hashtags
    text = re.sub(r"#[^\s]+", "", text)

    # Replace mentions with a generic tag
    text = re.sub(r"@[^\s]+", "user", text)

    # Replace X/Y with X or Y
    text = re.sub(r"\b([a-z]+)[//]([a-z]+)\b", r"\1 or \2", text)

    # Convert emojis to text
    text = emoji.demojize(text, delimiters=("emoji_", ""))

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # EXTRA: imdb50k specific cleaning
    text = re.sub(r"mst3k", "", text)  # Very common acronym for Mystery Science Theater 3000

    return text.strip()


def _lemmatize(doc: Doc, threshold: int = 3) -> Sequence[str]:
    """Lemmatize the provided text using spaCy.

    Args:
        doc: spaCy document
        threshold: Minimum character length of tokens

    Returns:
        Sequence of lemmatized tokens
    """
    return [
        tok
        for token in doc
        if not token.is_stop  # Ignore stop words
        and not token.is_punct  # Ignore punctuation
        and not token.like_email  # Ignore email addresses
        and not token.like_url  # Ignore URLs
        and not token.like_num  # Ignore numbers
        and token.is_alpha  # Ignore non-alphabetic tokens
        and (len(tok := token.lemma_.lower().strip()) >= threshold)  # Ignore short tokens
    ]


def tokenize(
    text_data: Sequence[str],
    batch_size: int = 512,
    n_jobs: int = 4,
    character_threshold: int = 3,
    show_progress: bool = True,
) -> Sequence[Sequence[str]]:
    """Tokenize the provided text using spaCy.

    Args:
        text_data: Text data to tokenize
        batch_size: Batch size for tokenization
        n_jobs: Number of parallel jobs
        character_threshold: Minimum character length of tokens
        show_progress: Whether to show a progress bar

    Returns:
        Tokenized text data
    """
    text_data = [
        _clean(text)
        for text in tqdm(
            text_data,
            desc="Cleaning",
            unit="doc",
            disable=not show_progress,
        )
    ]

    return pd.Series(
        [
            _lemmatize(doc, character_threshold)
            for doc in tqdm(
                nlp.pipe(text_data, batch_size=batch_size, n_process=n_jobs, disable=["parser", "ner"]),
                total=len(text_data),
                desc="Lemmatization",
                unit="doc",
                disable=not show_progress,
            )
        ],
    )


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


def load_amazonreviews() -> tuple[list[str], list[int]]:
    """Load the amazonreviews dataset and make it suitable for use.

    Returns:
        Text and label data

    Raises:
        FileNotFoundError: If the dataset is not found
    """
    # Check if the dataset exists
    if not AMAZONREVIEWS_PATH.exists():
        msg = (
            f"Amazonreviews dataset not found at: '{AMAZONREVIEWS_PATH}'\n"
            "Please download the dataset from:\n"
            f"{AMAZONREVIEWS_URL}"
        )
        raise FileNotFoundError(msg)

    # Load the dataset
    with bz2.BZ2File(AMAZONREVIEWS_PATH) as f:
        dataset = [line.decode("utf-8") for line in f]

    # Split the data into labels and text
    labels, texts = zip(*(line.split(" ", 1) for line in dataset))

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


def load_test(include_neutral: bool = False) -> tuple[list[str], list[int]]:
    """Load the test dataset and make it suitable for use.

    Args:
        include_neutral: Whether to include neutral sentiment

    Returns:
        Text and label data

    Raises:
        FileNotFoundError: If the dataset is not found
    """
    # Check if the dataset exists
    if not TEST_DATASET_PATH.exists():
        msg = (
            f"Test dataset not found at: '{TEST_DATASET_PATH}'\n"
            "Please download the dataset from:\n"
            f"{TEST_DATASET_URL}"
        )
        raise FileNotFoundError(msg)

    # Load the dataset
    data = pd.read_csv(TEST_DATASET_PATH)

    # Ignore rows with neutral sentiment
    if not include_neutral:
        data = data[data["label"] != 1]

    # Map sentiment values
    data["label"] = data["label"].map(
        {
            0: 0,  # Negative
            1: 1,  # Neutral
            2: 2,  # Positive
        },
    )

    # Return as lists
    return data["text"].tolist(), data["label"].tolist()


def load_data(dataset: Literal["sentiment140", "amazonreviews", "imdb50k", "test"]) -> tuple[list[str], list[int]]:
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
            return load_amazonreviews()
        case "imdb50k":
            return load_imdb50k()
        case "test":
            return load_test(include_neutral=False)
        case _:
            msg = f"Unknown dataset: {dataset}"
            raise ValueError(msg)
