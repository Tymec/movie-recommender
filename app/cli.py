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
    import os

    from app.gui import launch_gui

    os.environ["MODEL_PATH"] = model_path.as_posix()
    launch_gui(share)


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

    from app.model import infer_model

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
    prediction = infer_model(model, [text])[0]
    # prediction = model.predict([text])[0]
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
    default="test",
    help="Dataset to evaluate the model on",
    type=click.Choice(["test", "sentiment140", "amazonreviews", "imdb50k"]),
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
@click.option(
    "--batch-size",
    default=512,
    help="Size of the batches used in tokenization",
    show_default=True,
)
@click.option(
    "--processes",
    default=8,
    help="Number of parallel jobs during tokenization",
    show_default=True,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show verbose output",
)
def evaluate(
    dataset: Literal["test", "sentiment140", "amazonreviews", "imdb50k"],
    model_path: Path,
    cv: int,
    batch_size: int,
    processes: int,
    verbose: bool,
) -> None:
    """Evaluate the model on the the specified dataset"""
    import joblib

    from app.constants import CACHE_DIR
    from app.data import load_data, tokenize
    from app.model import evaluate_model

    cached_data_path = CACHE_DIR / f"{dataset}_tokenized.pkl"
    use_cached_data = False
    if cached_data_path.exists():
        use_cached_data = click.confirm(f"Found existing tokenized data for '{dataset}'. Use it?", default=True)

    if use_cached_data:
        click.echo("Loading cached data... ", nl=False)
        token_data, label_data = joblib.load(cached_data_path)
        click.echo(DONE_STR)
    else:
        click.echo("Loading dataset... ", nl=False)
        text_data, label_data = load_data(dataset)
        click.echo(DONE_STR)

        click.echo("Tokenizing data... ", nl=False)
        token_data = tokenize(text_data, batch_size=batch_size, n_jobs=processes, show_progress=True)
        joblib.dump((token_data, label_data), cached_data_path, compress=3)
        click.echo(DONE_STR)

        del text_data

    click.echo("Loading model... ", nl=False)
    model = joblib.load(model_path)
    click.echo(DONE_STR)

    click.echo("Evaluating model... ", nl=False)
    acc_mean, acc_std = evaluate_model(model, token_data, label_data, folds=cv, verbose=verbose)
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
    "--batch-size",
    default=512,
    help="Size of the batches used in tokenization",
    show_default=True,
)
@click.option(
    "--processes",
    default=4,
    help="Number of parallel jobs to run",
    show_default=True,
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
@click.option(
    "--verbose",
    is_flag=True,
    help="Show verbose output",
)
def train(
    dataset: Literal["sentiment140", "amazonreviews", "imdb50k"],
    max_features: int,
    cv: int,
    batch_size: int,
    processes: int,
    seed: int,
    force: bool,
    verbose: bool,
) -> None:
    """Train the model on the provided dataset"""
    import joblib

    from app.constants import CACHE_DIR, MODELS_DIR
    from app.data import load_data, tokenize
    from app.model import train_model

    model_path = MODELS_DIR / f"{dataset}_tfidf_ft-{max_features}.pkl"
    if model_path.exists() and not force:
        click.confirm(f"Model file '{model_path}' already exists. Overwrite?", abort=True)

    cached_data_path = CACHE_DIR / f"{dataset}_tokenized.pkl"
    use_cached_data = False
    if cached_data_path.exists():
        use_cached_data = click.confirm(f"Found existing tokenized data for '{dataset}'. Use it?", default=True)

    if use_cached_data:
        click.echo("Loading cached data... ", nl=False)
        token_data, label_data = joblib.load(cached_data_path)
        click.echo(DONE_STR)
    else:
        click.echo("Loading dataset... ", nl=False)
        text_data, label_data = load_data(dataset)
        click.echo(DONE_STR)

        click.echo("Tokenizing data... ", nl=False)
        token_data = tokenize(text_data, batch_size=batch_size, n_jobs=processes, show_progress=True)
        joblib.dump((token_data, label_data), cached_data_path, compress=3)
        click.echo(DONE_STR)

        del text_data

    click.echo("Training model... ")
    model, accuracy = train_model(
        token_data,
        label_data,
        max_features=max_features,
        folds=cv,
        n_jobs=processes,
        seed=seed,
        verbose=verbose,
    )
    click.echo("Model accuracy: ", nl=False)
    click.secho(f"{accuracy:.2%}", fg="blue")

    click.echo("Model saved to: ", nl=False)
    joblib.dump(model, model_path, compress=3)
    click.secho(str(model_path), fg="blue")


def cli_wrapper() -> None:
    cli(max_content_width=120)


if __name__ == "__main__":
    cli_wrapper()
