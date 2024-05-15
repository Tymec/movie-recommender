"""Utility functions"""

from __future__ import annotations

import itertools
import re
import warnings
from collections import deque
from enum import Enum
from functools import lru_cache
from threading import Event, Lock
from typing import Any

from joblib import Memory
from numpy.random import RandomState

from constants import CACHE_DIR, DEFAULT_SEED

__all__ = ["colorize", "wrap_queued_call", "get_random_state", "get_cache_memory"]


ANSI_RESET = 0


class Color(Enum):
    """ANSI color codes."""

    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37


class Style(Enum):
    """ANSI style codes."""

    BOLD = 1
    DIM = 2
    ITALIC = 3
    UNDERLINE = 4
    BLINK = 5
    INVERTED = 7
    HIDDEN = 8


# https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock:
    def __init__(self):
        self._lock = Lock()
        self._inner_lock = Lock()
        self._pending_threads = deque()

    def acquire(self, blocking: bool = True) -> bool:
        with self._inner_lock:
            lock_acquired = self._lock.acquire(False)
            if lock_acquired:
                return True
            if not blocking:
                return False

            release_event = Event()
            self._pending_threads.append(release_event)

        release_event.wait()
        return self._lock.acquire()

    def release(self) -> None:
        with self._inner_lock:
            if self._pending_threads:
                release_event = self._pending_threads.popleft()
                release_event.set()

            self._lock.release()

    __enter__ = acquire

    def __exit__(self, _t, _v, _tb):  # noqa: ANN001
        self.release()


@lru_cache(maxsize=1)
def get_queue_lock() -> FIFOLock:
    return FIFOLock()


@lru_cache(maxsize=1)
def get_random_state(seed: int = DEFAULT_SEED) -> RandomState:
    return RandomState(seed)


@lru_cache(maxsize=1)
def get_cache_memory() -> Memory:
    return Memory(CACHE_DIR, verbose=0)


def to_ansi(code: int) -> str:
    """Convert an integer to an ANSI escape code."""
    return f"\033[{code}m"


@lru_cache(maxsize=None)
def get_ansi_color(color: Color, bright: bool = False, background: bool = False) -> str:
    """Get ANSI color code for the specified color, brightness and background."""
    code = color.value
    if bright:
        code += 60
    if background:
        code += 10
    return to_ansi(code)


def replace_color_tag(color: Color, text: str) -> None:
    """Replace both dark and light color tags for background and foreground."""
    for bright, bg in itertools.product([False, True], repeat=2):
        tag = f"{'BG_' if bg else ''}{'BRIGHT_' if bright else ''}{color.name}"
        text = text.replace(f"[{tag}]", get_ansi_color(color, bright=bright, background=bg))
        text = text.replace(f"[/{tag}]", to_ansi(ANSI_RESET))

    return text


@lru_cache(maxsize=256)
def colorize(text: str, strip: bool = True) -> str:
    """Format text with ANSI color codes using tags [COLOR], [BG_COLOR] and [STYLE].
    Reset color/style with [/TAG].
    Escape with double brackets [[]]. Strip leading and trailing whitespace if strip=True.
    """

    # replace foreground and background color tags
    for color in Color:
        text = replace_color_tag(color, text)

    # replace style tags
    for style in Style:
        text = text.replace(f"[{style.name}]", to_ansi(style.value)).replace(f"[/{style.name}]", to_ansi(ANSI_RESET))

    # if there are any tags left, remove them and throw a warning
    pat1 = re.compile(r"((?<!\[)\[)([^\[\]]*)(\](?!\]))")
    for match in pat1.finditer(text):
        color = match.group(1)
        text = text.replace(match.group(0), "")
        warnings.warn(f"Invalid color tag: {color!r}", UserWarning, stacklevel=2)

    # escape double brackets
    pat2 = re.compile(r"\[\[[^\[\]\v]+\]\]")
    text = pat2.sub("", text)

    # reset color/style at the end
    text += to_ansi(ANSI_RESET)

    return text.strip() if strip else text


# https://github.com/AUTOMATIC1111/stable-diffusion-webui/modules/call_queue.py
def wrap_queued_call(func: callable) -> callable:
    def f(*args, **kwargs) -> Any:  # noqa: ANN003, ANN002
        with get_queue_lock():
            return func(*args, **kwargs)

    return f
