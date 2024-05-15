from pathlib import Path

DEFAULT_SEED: int = 42
MAX_TOKENIZER_FEATURES: int = 500000
CLF_MAX_ITER: int = 1000

DATASET_PATH: Path = Path("data/training.1600000.processed.noemoticon.csv")
STOPWORDS_PATH: Path = Path("data/stopwords-en.txt")
MODELS_DIR: Path = Path("models")
CACHE_DIR: Path = Path("cache")
CHECKPOINT_PATH: Path = CACHE_DIR / "pipeline.pkl"


# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
