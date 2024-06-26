"""GUI using Gradio."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import gradio as gr
import joblib

from app.model import infer_model

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

__all__ = ["launch_gui"]


POSITIVE_LABEL = "Positive 😊"
NEUTRAL_LABEL = "Neutral 😐"
NEGATIVE_LABEL = "Negative 😤"


@lru_cache(maxsize=1)
def load_model() -> BaseEstimator:
    """Load the trained model and cache it.

    Returns:
        Loaded model
    """
    model_path = os.environ.get("MODEL_PATH", None)
    if model_path is None:
        msg = "MODEL_PATH environment variable not set"
        raise ValueError(msg)
    return joblib.load(model_path)


def sentiment_analysis(text: str) -> str:
    """Perform sentiment analysis on the provided text.

    Args:
        text: Input text

    Returns:
        Predicted sentiment label
    """
    prediction = infer_model(load_model(), [text])[0]

    if prediction == 0:
        return NEGATIVE_LABEL
    if prediction == 1:
        return POSITIVE_LABEL
    return NEUTRAL_LABEL


demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(lines=10, label="Enter text here"),
    outputs="label",
    title="Sentiment Analysis",
    description="Predict the sentiment of a given text.",
    examples=[
        ["I love the weather today!"],
        ["You are a terrible person."],
        ["The movie we watched was boring."],
        ["This website is amazing!"],
    ],
    allow_flagging=False,
)


def launch_gui(share: bool) -> None:
    """Launch the Gradio GUI.

    Args:
        share: Whether to create a public link
    """
    demo.launch(share=share)


if __name__ == "__main__":
    demo.launch()
