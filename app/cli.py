from __future__ import annotations

from pathlib import Path
from typing import Literal

import click

__all__ = ["cli_wrapper"]

ERROR_STR = click.style("ERROR", fg="red")
DONE_STR = click.style("DONE", fg="green")
POSITIVE_STR = click.style("POSITIVE", fg="green")
NEUTRAL_STR = click.style("NEUTRAL", fg="yellow")
NEGATIVE_STR = click.style("NEGATIVE", fg="red")


@click.group()
def cli() -> None: ...


@cli.command()
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to the trained model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--share/--no-share",
    default=False,
    help="Whether to create a shareable link",
)
def gui(model_path: Path, share: bool) -> None:
    """Launch the Gradio GUI"""
    from app.gui import launch_gui

    launch_gui(model_path, share)


@cli.command()
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to the trained model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
)
@click.argument("text", nargs=-1)
def predict(model_path: Path, text: list[str]) -> None:
    """Perform sentiment analysis on the provided text.

    Note: Piped input takes precedence over the text argument
    """
    import sys

    import joblib

    text = " ".join(text).strip()
    if not sys.stdin.isatty():
        piped_text = sys.stdin.read().strip()
        text = piped_text or text

    if not text:
        click.echo(f"{ERROR_STR}: No text provided")
        return

    click.echo("Loading model... ", nl=False)
    model = joblib.load(model_path)
    click.echo(DONE_STR)

    click.echo("Performing sentiment analysis... ", nl=False)
    prediction = model.predict([text])[0]
    if prediction == 0:
        sentiment = NEGATIVE_STR
    elif prediction == 1:
        sentiment = POSITIVE_STR
    else:
        sentiment = NEUTRAL_STR
    click.echo(sentiment)


@cli.command()
@click.option(
    "--dataset",
    required=True,
    help="Dataset to train the model on",
    type=click.Choice(["sentiment140", "amazonreviews", "imdb50k"]),
)
@click.option(
    "--max-features",
    default=20000,
    help="Maximum number of features",
    show_default=True,
    type=click.IntRange(1, None),
)
@click.option(
    "--seed",
    default=42,
    help="Random seed (-1 for random seed)",
    show_default=True,
    type=click.IntRange(-1, None),
)
def train(
    dataset: Literal["sentiment140", "amazonreviews", "imdb50k"],
    max_features: int,
    seed: int,
) -> None:
    """Train the model on the provided dataset"""
    import joblib

    from app.constants import MODELS_DIR
    from app.model import create_model, load_data, train_model

    model_path = MODELS_DIR / f"{dataset}_tfidf_ft-{max_features}.pkl"
    if model_path.exists():
        click.confirm(f"Model file '{model_path}' already exists. Overwrite?", abort=True)

    click.echo("Preprocessing dataset... ", nl=False)
    text_data, label_data = load_data(dataset)
    click.echo(DONE_STR)

    click.echo("Creating model... ", nl=False)
    model = create_model(max_features, seed=None if seed == -1 else seed)
    click.echo(DONE_STR)

    click.echo("Training model... ", nl=False)
    accuracy = train_model(model, text_data, label_data)
    joblib.dump(model, model_path)
    click.echo(DONE_STR)

    click.echo("Model accuracy: ")
    click.secho(f"{accuracy:.2%}", fg="blue")

    # TODO: Add hyperparameter options
    # TODO: Random/grid search for finding best classifier and hyperparameters


def cli_wrapper() -> None:
    cli(max_content_width=120)


if __name__ == "__main__":
    cli_wrapper()
