from __future__ import annotations

import bz2
import re
import warnings
from typing import Literal

import nltk
import pandas as pd
from joblib import Memory
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.constants import (
    AMAZONREVIEWS_PATH,
    AMAZONREVIEWS_URL,
    CACHE_DIR,
    EMOTICON_MAP,
    IMDB50K_PATH,
    IMDB50K_URL,
    SENTIMENT140_PATH,
    SENTIMENT140_URL,
    URL_REGEX,
)

__all__ = ["load_data", "create_model", "train_model", "evaluate_model"]


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        replace_url: bool = True,
        replace_hashtag: bool = True,
        replace_emoticon: bool = True,
        replace_emoji: bool = True,
        lowercase: bool = True,
        character_threshold: int = 2,
        remove_special_characters: bool = True,
        remove_extra_spaces: bool = True,
    ):
        self.replace_url = replace_url
        self.replace_hashtag = replace_hashtag
        self.replace_emoticon = replace_emoticon
        self.replace_emoji = replace_emoji
        self.lowercase = lowercase
        self.character_threshold = character_threshold
        self.remove_special_characters = remove_special_characters
        self.remove_extra_spaces = remove_extra_spaces

    def fit(self, _data: list[str], _labels: list[int] | None = None) -> TextCleaner:
        return self

    def transform(self, data: list[str], _labels: list[int] | None = None) -> list[str]:
        # Replace URLs, hashtags, emoticons, and emojis
        data = [re.sub(URL_REGEX, "URL", text) for text in data] if self.replace_url else data
        data = [re.sub(r"#\w+", "HASHTAG", text) for text in data] if self.replace_hashtag else data

        # Replace emoticons
        if self.replace_emoticon:
            for word, emoticons in EMOTICON_MAP.items():
                for emoticon in emoticons:
                    data = [text.replace(emoticon, f"EMOTE_{word}") for text in data]

        # Basic text cleaning
        data = [text.lower() for text in data] if self.lowercase else data  # Lowercase
        threshold_pattern = re.compile(rf"\b\w{{1,{self.character_threshold}}}\b")
        data = (
            [re.sub(threshold_pattern, "", text) for text in data] if self.character_threshold > 0 else data
        )  # Remove short words
        data = (
            [re.sub(r"[^a-zA-Z0-9\s]", "", text) for text in data] if self.remove_special_characters else data
        )  # Remove special characters
        data = [re.sub(r"\s+", " ", text) for text in data] if self.remove_extra_spaces else data  # Remove extra spaces

        # Remove leading and trailing whitespace
        return [text.strip() for text in data]


class TextLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, _data: list[str], _labels: list[int] | None = None) -> TextLemmatizer:
        return self

    def transform(self, data: list[str], _labels: list[int] | None = None) -> list[str]:
        return [self.lemmatizer.lemmatize(text) for text in data]


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
    with bz2.BZ2File(AMAZONREVIEWS_PATH[1]) as train_file:
        train_data = [line.decode("utf-8") for line in train_file]

    test_data = []
    if merge:
        with bz2.BZ2File(AMAZONREVIEWS_PATH[0]) as test_file:
            test_data = [line.decode("utf-8") for line in test_file]

    # Merge the datasets
    data = train_data + test_data

    # Split the data into labels and text
    labels, texts = zip(*(line.split(" ", 1) for line in data))

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
    # Download NLTK data if not already downloaded
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Load English stopwords
    stopwords_en = set(stopwords.words("english"))

    return Pipeline(
        [
            # Text preprocessing
            ("clean", TextCleaner()),
            ("lemma", TextLemmatizer()),
            # Preprocess (NOTE: Can be replaced with TfidfVectorizer, but left for clarity)
            (
                "vectorize",
                CountVectorizer(stop_words=stopwords_en, ngram_range=(1, 2), max_features=max_features),
            ),
            ("tfidf", TfidfTransformer()),
            # Classifier
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ],
        memory=Memory(CACHE_DIR, verbose=0),
        verbose=verbose,
    )


def train_model(
    model: Pipeline,
    text_data: list[str],
    label_data: list[int],
    seed: int = 42,
) -> tuple[float, list[str], list[int]]:
    """Train the sentiment analysis model.

    Args:
        model: Untrained model
        text_data: Text data
        label_data: Label data
        seed: Random seed (None for random seed)

    Returns:
        Model accuracy and test data
    """
    text_train, text_test, label_train, label_test = train_test_split(
        text_data,
        label_data,
        test_size=0.2,
        random_state=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(text_train, label_train)

    return model.score(text_test, label_test), text_test, label_test


def evaluate_model(
    model: Pipeline,
    text_test: list[str],
    label_test: list[int],
    cv: int = 5,
) -> tuple[float, float]:
    """Evaluate the model using cross-validation.

    Args:
        model: Trained model
        text_test: Text data
        label_test: Label data
        seed: Random seed (None for random seed)
        cv: Number of cross-validation folds

    Returns:
        Mean accuracy and standard deviation
    """
    scores = cross_val_score(
        model,
        text_test,
        label_test,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    return scores.mean(), scores.std()
