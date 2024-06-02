from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["serialize", "deserialize"]


def serialize(data: list[list[str]], path: Path, max_size: int = 400) -> None:
    """Serialize data to a file

    Args:
        data: The data to serialize
        path: The path to save the serialized data
        max_size: The maximum size a chunk can be (in elements)
    """
    # first file is path, next chunks have ".1", ".2", etc. appended
    for i, chunk in enumerate(tqdm([data[i : i + max_size] for i in range(0, len(data), max_size)])):
        fd = path.with_suffix(f".{i}.pkl" if i else ".pkl")
        with fd.open("wb") as f:
            joblib.dump(chunk, f, compress=3)


def deserialize(path: Path) -> list[list[str]]:
    """Deserialize data from a file

    Args:
        path: The path to the serialized data

    Returns:
        The deserialized data
    """
    data = []
    i = 0
    while (fd := path.with_suffix(f".{i}.pkl" if i else ".pkl")).exists():
        with fd.open("rb") as f:
            data.extend(joblib.load(f))
        i += 1
    return data
