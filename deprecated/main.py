from __future__ import annotations

from pathlib import Path

import click
import joblib

from app.utils import colorize


@click.group()
def cli() -> None: ...


@cli.command("predict")
@click.option(
    "-m",
    "--model",
    "model_path",
    default="models/model.pkl",
    help="Path to the model file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path),
)
@click.argument("text", nargs=-1)
def predict(model_path: Path, text: list[str]) -> None:
    input_text = " ".join(text).strip()
    if not input_text:
        click.echo("[RED]Error[/RED]: Input text is empty.")
        return

    # Load the model
    click.echo("Loading model... ", nl=False)
    model = joblib.load(model_path)
    click.echo(colorize("[GREEN]DONE"))

    # Run the model
    click.echo("Performing sentiment analysis... ", nl=False)
    prediction = model.predict([input_text])
    sentiment = "[GREEN]POSITIVE" if prediction[0] == 1 else "[RED]NEGATIVE"
    click.echo(colorize(sentiment))


if __name__ == "__main__":
    cli()
