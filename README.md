---
title: Sentiment Analysis
emoji: 🤗
colorFrom: blue
colorTo: green
pinned: false
sdk: gradio
python_version: 3.11
app_file: app/gui.py
datasets:
  - mrshu/amazonreviews
  - stanfordnlp/sentiment140
  - stanfordnlp/imdb
  - Sp1786/multiclass-sentiment-analysis-dataset
models:
  - spacy/en_core_web_sm
---

# Sentiment Analysis

### Usage
1. Clone the repository
2. `cd` into the repository
3. Run `just install` to install the dependencies
4. Run `just run --help` to see the available commands

### Datasets
- [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Multiclass Sentiment Analysis](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) (Used only testing)

### Required tools
- `just`
- `poetry`
