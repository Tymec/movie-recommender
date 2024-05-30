#!/usr/bin/env just --justfile

@default:
  just --list

@lint:
  poetry run pre-commit run --all-files

@install:
  poetry install --only main
  poetry run spacy download en_core_web_sm

@install-dev:
  poetry self add poetry-plugin-export
  poetry install

@requirements:
  poetry export -f requirements.txt --output requirements.txt --without dev

[no-exit-message]
@app *ARGS:
  poetry run python -m app {{ARGS}}
