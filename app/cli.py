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
    "--token-batch-size",
    default=512,
    help="Size of the batches used in tokenization",
    show_default=True,
)
@click.option(
    "--token-jobs",
    default=4,
    help="Number of parallel jobs to run for tokenization",
    show_default=True,
)
@click.option(
    "--eval-jobs",
    default=1,
    help="Number of parallel jobs to run for evaluation",
    show_default=True,
)
@click.option(
    "--force-cache",
    is_flag=True,
    help="Always use the cached tokenized data (if available)",
)
def evaluate(
    dataset: Literal["test", "sentiment140", "amazonreviews", "imdb50k"],
    model_path: Path,
    cv: int,
    token_batch_size: int,
    token_jobs: int,
    eval_jobs: int,
    force_cache: bool,
) -> None:
    """Evaluate the model on the the specified dataset"""
    import gc

    import joblib

    from app.constants import CACHE_DIR
    from app.data import load_data, tokenize
    from app.model import evaluate_model
    from app.utils import deserialize, serialize

    cached_data_path = CACHE_DIR / f"{dataset}_tokenized.pkl"
    use_cached_data = False
    if cached_data_path.exists():
        use_cached_data = force_cache or click.confirm(
            f"Found existing tokenized data for '{dataset}'. Use it?",
            default=True,
        )

    click.echo("Loading dataset... ", nl=False)
    text_data, label_data = load_data(dataset)
    click.echo(DONE_STR)

    if use_cached_data:
        click.echo("Loading cached data... ", nl=False)
        token_data = deserialize(cached_data_path)
        click.echo(DONE_STR)
    else:
        click.echo("Tokenizing data... ", nl=False)
        token_data = tokenize(text_data, batch_size=token_batch_size, n_jobs=token_jobs, show_progress=True)
        click.echo(DONE_STR)

        click.echo("Caching tokenized data... ", nl=False)
        serialize(token_data, cached_data_path)
        click.echo(DONE_STR)

    del text_data
    gc.collect()

    click.echo("Loading model... ", nl=False)
    model = joblib.load(model_path)
    click.echo(DONE_STR)

    click.echo("Evaluating model... ", nl=False)
    acc_mean, acc_std = evaluate_model(
        model,
        token_data,
        label_data,
        folds=cv,
        n_jobs=eval_jobs,
    )
    click.secho(f"{acc_mean:.2%} ± {acc_std:.2%}", fg="blue")


@cli.command()
@click.option(
    "--dataset",
    required=True,
    help="Dataset to train the model on",
    type=click.Choice(["sentiment140", "amazonreviews", "imdb50k"]),
)
@click.option(
    "--vectorizer",
    default="tfidf",
    help="Vectorizer to use",
    type=click.Choice(["tfidf", "count", "hashing"]),
)
@click.option(
    "--max-features",
    default=20000,
    help="Maximum number of features (should be greater than 2^15 when using hashing vectorizer)",
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
    "--token-batch-size",
    default=512,
    help="Size of the batches used in tokenization",
    show_default=True,
)
@click.option(
    "--token-jobs",
    default=4,
    help="Number of parallel jobs to run for tokenization",
    show_default=True,
)
@click.option(
    "--train-jobs",
    default=1,
    help="Number of parallel jobs to run for training",
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
    "--overwrite",
    is_flag=True,
    help="Overwrite the model file if it already exists",
)
@click.option(
    "--force-cache",
    is_flag=True,
    help="Always use the cached tokenized data (if available)",
)
def train(
    dataset: Literal["sentiment140", "amazonreviews", "imdb50k"],
    vectorizer: Literal["tfidf", "count", "hashing"],
    max_features: int,
    cv: int,
    token_batch_size: int,
    token_jobs: int,
    train_jobs: int,
    seed: int,
    overwrite: bool,
    force_cache: bool,
) -> None:
    """Train the model on the provided dataset"""
    import gc

    import joblib

    from app.constants import CACHE_DIR, MODEL_DIR
    from app.data import load_data, tokenize
    from app.model import train_model
    from app.utils import deserialize, serialize

    model_path = MODEL_DIR / f"{dataset}_{vectorizer}_ft{max_features}.pkl"
    if model_path.exists() and not overwrite:
        click.confirm(f"Model file '{model_path}' already exists. Overwrite?", abort=True)

    cached_data_path = CACHE_DIR / f"{dataset}_tokenized.pkl"
    use_cached_data = False

    if cached_data_path.exists():
        use_cached_data = force_cache or click.confirm(
            f"Found existing tokenized data for '{dataset}'. Use it?",
            default=True,
        )

    click.echo("Loading dataset... ", nl=False)
    text_data, label_data = load_data(dataset)
    click.echo(DONE_STR)

    if use_cached_data:
        click.echo("Loading cached data... ", nl=False)
        token_data = deserialize(cached_data_path)
        click.echo(DONE_STR)
    else:
        click.echo("Tokenizing data... ", nl=False)
        token_data = tokenize(text_data, batch_size=token_batch_size, n_jobs=token_jobs, show_progress=True)
        click.echo(DONE_STR)

        click.echo("Caching tokenized data... ", nl=False)
        serialize(token_data, cached_data_path)
        click.echo(DONE_STR)

    del text_data
    gc.collect()

    click.echo("Training model... ")
    model, accuracy = train_model(
        token_data,
        label_data,
        vectorizer=vectorizer,
        max_features=max_features,
        folds=cv,
        n_jobs=train_jobs,
        seed=seed,
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
