#!/usr/bin/env just --justfile

@default:
  echo "No target specified."

@lint:
  poetry run pre-commit run --all-files

@install:
  poetry install --without dev

@install-dev:
  poetry install

@requirements:
  poetry export -f requirements.txt --output requirements.txt --without dev

@run TEXT:
  poetry run python main.py {{TEXT}}
