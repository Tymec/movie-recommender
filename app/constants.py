from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

TOKENIZER_CACHE_PATH = CACHE_DIR / "tokenizer"

SENTIMENT140_PATH = DATA_DIR / "sentiment140.csv"
SENTIMENT140_URL = "https://www.kaggle.com/datasets/kazanova/sentiment140"

AMAZONREVIEWS_PATH = DATA_DIR / "amazonreviews.txt.bz2"
AMAZONREVIEWS_URL = "https://www.kaggle.com/datasets/bittlingmayer/amazonreviews"

IMDB50K_PATH = DATA_DIR / "imdb50k.csv"
IMDB50K_URL = "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"

TEST_DATASET_PATH = DATA_DIR / "test.csv"
TEST_DATASET_URL = "https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset"

SLANGMAP_PATH = DATA_DIR / "slang.json"
SLANGMAP_URL = "Https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing"

CACHE_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TOKENIZER_CACHE_PATH.mkdir(exist_ok=True, parents=True)
