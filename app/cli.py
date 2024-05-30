from __future__ import annotations

from pathlib import Path
from typing import Literal

import click

__all__ = ["cli_wrapper"]

DONE_STR = click.style("DONE", fg="green")


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
        msg = "No text provided"
        raise click.UsageError(msg)

    click.echo("Loading model... ", nl=False)
    model = joblib.load(model_path)
    click.echo(DONE_STR)

    click.echo("Performing sentiment analysis... ", nl=False)
    prediction = model.predict([text])[0]
    if prediction == 0:
        sentiment = click.style("NEGATIVE", fg="red")
    elif prediction == 1:
        sentiment = click.style("POSITIVE", fg="green")
    else:
        sentiment = click.style("NEUTRAL", fg="yellow")
    click.echo(sentiment)


@cli.command()
@click.option(
    "--dataset",
    required=True,
    help="Dataset to train the model on",
    type=click.Choice(["sentiment140", "amazonreviews", "imdb50k"]),
)
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to the trained model",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
)
@click.option(
    "--cv",
    default=5,
    help="Number of cross-validation folds",
    show_default=True,
    type=click.IntRange(1, 50),
)
def evaluate(
    dataset: Literal["sentiment140", "amazonreviews", "imdb50k"],
    model_path: Path,
    cv: int,
) -> None:
    """Evaluate the model on the test dataset"""
    import joblib

    from app.data import load_data
    from app.model import evaluate_model

    click.echo("Loading dataset... ", nl=False)
    text_data, label_data = load_data(dataset)
    click.echo(DONE_STR)

    click.echo("Loading model... ", nl=False)
    model = joblib.load(model_path)
    click.echo(DONE_STR)

    click.echo("Evaluating model... ", nl=False)
    acc_mean, acc_std = evaluate_model(model, text_data, label_data, folds=cv)
    click.secho(f"{acc_mean:.2%} ± {acc_std:.2%}", fg="blue")


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
    "--cv",
    default=5,
    help="Number of cross-validation folds",
    show_default=True,
    type=click.IntRange(1, 50),
)
@click.option(
    "--seed",
    default=42,
    help="Random seed (-1 for random seed)",
    show_default=True,
    type=click.IntRange(-1, None),
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the model file if it already exists",
)
def train(
    dataset: Literal["sentiment140", "amazonreviews", "imdb50k"],
    max_features: int,
    cv: int,
    seed: int,
    force: bool,
) -> None:
    """Train the model on the provided dataset"""
    import joblib

    from app.constants import MODELS_DIR
    from app.data import load_data
    from app.model import create_model, evaluate_model, train_model

    model_path = MODELS_DIR / f"{dataset}_tfidf_ft-{max_features}.pkl"
    if model_path.exists() and not force:
        click.confirm(f"Model file '{model_path}' already exists. Overwrite?", abort=True)

    click.echo("Loading dataset... ", nl=False)
    text_data, label_data = load_data(dataset)
    click.echo(DONE_STR)

    click.echo("Creating model... ", nl=False)
    model = create_model(max_features, seed=None if seed == -1 else seed, verbose=True)
    click.echo(DONE_STR)

    click.echo("Training model... ")
    accuracy = train_model(model, text_data, label_data)
    click.echo("Model accuracy: ", nl=False)
    click.secho(f"{accuracy:.2%}", fg="blue")

    click.echo("Model saved to: ", nl=False)
    joblib.dump(model, model_path)
    click.secho(str(model_path), fg="blue")

    click.echo("Evaluating model... ", nl=False)
    acc_mean, acc_std = evaluate_model(model, text_data, label_data, folds=cv)
    click.secho(f"{acc_mean:.2%} ± {acc_std:.2%}", fg="blue")


def cli_wrapper() -> None:
    cli(max_content_width=120)


if __name__ == "__main__":
    cli_wrapper()
