from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import gradio as gr
import joblib

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

__all__ = ["launch_gui"]


POSITIVE_LABEL = "Positive 😊"
NEUTRAL_LABEL = "Neutral 😐"
NEGATIVE_LABEL = "Negative 😤"


@lru_cache(maxsize=1)
def load_model() -> BaseEstimator:
    """Load the trained model and cache it."""
    model_path = os.environ.get("MODEL_PATH", None)
    if model_path is None:
        msg = "MODEL_PATH environment variable not set"
        raise ValueError(msg)
    return joblib.load(model_path)


def sentiment_analysis(text: str) -> str:
    """Perform sentiment analysis on the provided text."""
    model = load_model()
    prediction = model.predict([text])[0]

    if prediction == 0:
        return NEGATIVE_LABEL
    if prediction == 1:
        return POSITIVE_LABEL
    return NEUTRAL_LABEL


demo = gr.Interface(
    fn=sentiment_analysis,
    inputs="text",
    outputs="label",
    title="Sentiment Analysis",
)


def launch_gui(model_path: str, share: bool) -> None:
    """Launch the Gradio GUI."""
    os.environ["MODEL_PATH"] = model_path
    demo.launch(share=share)


if __name__ == "__main__":
    demo.launch()
