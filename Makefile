#!/usr/bin/make -f

default: install

install:
  @poetry install --only main
  @poetry run spacy download en_core_web_sm

install-dev:
  @poetry self add poetry-plugin-export
  @poetry install

requirements:
  @poetry export -f requirements.txt --output requirements.txt --without dev
  @poetry export -f requirements.txt --output requirements-dev.txt

lint:
  @poetry run pre-commit run --all-files

.PHONY: install install-dev requirements gradio lint run
