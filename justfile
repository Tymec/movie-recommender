#!/usr/bin/env just --justfile

@default:
  just --list

@lint:
  poetry run pre-commit run --all-files

@install:
  poetry install --only main

@install-dev:
  pipx inject poetry poetry-plugin-export
  poetry install
  poetry run spacy download en_core_web_sm
  poetry run pre-commit install

@requirements:
  poetry export -f requirements.txt --output requirements.txt --without dev
  poetry export -f requirements.txt --output requirements-dev.txt

[no-exit-message]
@run *ARGS:
  poetry run python -m app {{ARGS}}
