---
title: Sentiment Analysis
emoji: 🤗
colorFrom: yellow
colorTo: orange
pinned: false
sdk: gradio
python_version: 3.11
app_file: app.py
datasets:
  - mrshu/amazonreviews
  - stanfordnlp/sentiment140
  - stanfordnlp/imdb
---

# Sentiment Analysis

### Usage
1. Clone the repository
2. `cd` into the repository
3. Run `just install` to install the dependencies
4. Run `just app --help` to see the available commands

### Datasets
- [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Required tools
- `just`
- `poetry`
