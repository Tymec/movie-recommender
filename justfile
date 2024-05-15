#!/usr/bin/env just --justfile

@default:
  echo "No target specified."

@lint:
  poetry run pre-commit run --all-files

@install:
  poetry install --only main

@install-dev:
  poetry self add poetry-plugin-export
  poetry install

@requirements:
  poetry export -f requirements.txt --output requirements.txt --without dev

@run +TEXT:
  poetry run python main.py predict --model models/logistic_regression.pkl "{{TEXT}}"

@gui:
  poetry run gradio app/gui.py
