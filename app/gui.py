from __future__ import annotations

from pathlib import Path

import gradio as gr

from constants import MODELS_DIR
from model import predict, tokenize

CSS_PATH = Path("style.css")
TOKENIZER_EXT = ".tokenizer.pkl"
MODEL_EXT = ".model.pkl"
POSITIVE_LABEL = "Positive 😊"
NEGATIVE_LABEL = "Negative 😤"
REFRESH_SYMBOL = "🔄"


def load_style() -> str:
    if not CSS_PATH.is_file():
        return ""

    with Path.open(CSS_PATH) as f:
        return f.read()


def predict_wrapper(text: str, tokenizer: str, model: str) -> str:
    toks = tokenize(text, MODELS_DIR / f"{tokenizer}{TOKENIZER_EXT}")
    pred = predict(toks, MODELS_DIR / f"{model}{MODEL_EXT}")
    return POSITIVE_LABEL if pred else NEGATIVE_LABEL


def train_wrapper() -> None:
    msg = "Training is not supported in the GUI."
    raise NotImplementedError(msg)


def evaluate_wrapper() -> None:
    msg = "Evaluation is not supported in the GUI."
    raise NotImplementedError(msg)


with gr.Blocks(css=load_style()) as demo:
    gr.Markdown("## Sentiment Analysis")

    with gr.Row(equal_height=True):
        textbox = gr.Textbox(
            lines=10,
            label="Enter text to analyze",
            placeholder="Enter text here",
            key="input-textbox",
        )

        with gr.Column():
            output = gr.Label()

            with gr.Row(elem_classes="justify-between"):
                clear_btn = gr.ClearButton([textbox, output], value="Clear 🧹")
                analyze_btn = gr.Button(
                    "Analyze 🔍",
                    variant="primary",
                    interactive=False,
                )

            with gr.Row():
                tokenizer_selector = gr.Dropdown(
                    choices=[tkn.stem[: -len(".tokenizer")] for tkn in MODELS_DIR.glob(f"*{TOKENIZER_EXT}")],
                    label="Tokenizer",
                    key="tokenizer-selector",
                )

                model_selector = gr.Dropdown(
                    choices=[mdl.stem[: -len(".model")] for mdl in MODELS_DIR.glob(f"*{MODEL_EXT}")],
                    label="Model",
                    key="model-selector",
                )

                # TODO: Refresh button

    # Event handlers
    textbox.input(
        fn=lambda text: gr.update(interactive=bool(text.strip())),
        inputs=[textbox],
        outputs=[analyze_btn],
    )
    analyze_btn.click(
        fn=predict_wrapper,
        inputs=[textbox, tokenizer_selector, model_selector],
        outputs=[output],
    )

demo.queue()
demo.launch()
