from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import joblib
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["serialize", "deserialize"]


def serialize(data: Sequence[str | int], path: Path, max_size: int = 100_000, show_progress: bool = False) -> None:
    """Serialize data to a file

    Args:
        data: The data to serialize
        path: The path to save the serialized data
        max_size: The maximum size a chunk can be (in elements)
        show_progress: Whether to show a progress bar
    """
    for i, chunk in enumerate(
        tqdm(
            [data[i : i + max_size] for i in range(0, len(data), max_size)],
            desc="Serializing",
            unit="chunk",
            disable=not show_progress,
        ),
    ):
        fd = path.with_suffix(f".{i}.pkl" if i else ".pkl")
        with fd.open("wb") as f:
            joblib.dump(chunk, f, compress=3)


def deserialize(path: Path) -> Sequence[str | int]:
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
