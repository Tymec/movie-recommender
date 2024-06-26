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


# Sentiment Analysis [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://tymec-sentiment-analysis.hf.space)


### Table of Contents
- [Description](#description)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Predict](#predict)
  - [GUI](#gui)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Options](#options)
  - [Datasets](#datasets)
  - [Vectorizers](#vectorizers)
  - [Environment Variables](#environment-variables)
- [Implementation](#implementation)
  - [Architecture](#architecture)
  - [Pre-trained Models](#pre-trained-models)
- [License](#license)


## Description
This is a simple sentiment analysis model written in Python, designed to predict whether the provided text has a positive or negative sentiment. The project comes with both a graphical user interface and a command-line interface. While training the model, the user can choose from a couple of datasets to train the model on and then evaluate the trained model on another dataset. Once the model is trained, it can be used to predict the sentiment of any text with the help of the GUI or CLI.


## Installation
Clone the repository and once inside the directory, run the following command to install the dependencies:
```bash
python -m pip install -r requirements.txt
```

Ensure that you have **at least** one dataset downloaded and placed in the data directory before running `train`.
For `evaluate`, you will need the `test` dataset. See [Datasets](#datasets) for more information.

The project comes with pre-trained models that can be used for prediction. See [Pre-trained Models](#pre-trained-models) for more information.


### Prerequisites
- Python 3.11+


## Usage
To see the available commands and options, run:
```bash
python -m app --help
```
![help](assets/help.png)


### Predict
To perform sentiment analysis on a given text, run the following command:
```bash
python -m app predict --model <model> I love this movie
```
where `<model>` is the path to the trained model.

Alternatively, you can pipe the text into the command:
```bash
echo "I love this movie" | python -m app predict --model <model>
```
![predict-help](assets/predict.png)


### GUI
To launch the GUI, run the following command:
```bash
python -m app gui --model <model>
```
where `<model>` is the path to the trained model. Add the `--share` flag to create a publicly accessible link.
![gui-help](assets/gui.png)

After running the command, open the link from the terminal in your browser to access the GUI.
![gui](assets/space.png)


### Training
Before training the model, ensure that the specified dataset is downloaded and can be accessed at its respective path. To train the model, run the following command:
```bash
python -m app train --dataset <dataset> {options}
```
where `<dataset>` is the name of the dataset to train the model on. For available datasets, see [Datasets](#datasets).

The trained model will be exported to the models directory.

To see all available options, run:
```bash
python -m app train --help
```
![train-help](assets/train.png)


### Evaluation
Once the model is trained, you can evaluate it on a different dataset by running the following command:
```bash
python -m app evaluate --model <model>
```
where `<model>` is the path to the trained model. For available datasets, see [Datasets](#datasets).

To see all available options, run:
```bash
python -m app evaluate --help
```
![evaluate-help](assets/evaluate.png)


## Options

### Datasets
| Option | Path | Notes | Dataset |
| --- | --- | --- | --- |
| sentiment140 | `data/sentiment140.csv` | | [Twitter Sentiment Analysis](https://www.kaggle.com/kazanova/sentiment140) |
| amazonreviews | `data/amazonreviews.bz2` | only train is used | [Amazon Product Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews) |
| imdb50k | `data/imdb50k.csv` | | [IMDB Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| test | `data/test.csv` | only used in `evaluate` | [Sentiment Analysis Evaluation Dataset](https://www.kaggle.com/datasets/prishasawhney/sentiment-analysis-evaluation-dataset) |

#### Other
During text preprocessing, this [slang map](Https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing) is used to convert slang words to their formal form.


### Vectorizers
| Option | Description | When to Use |
| --- | --- | --- |
| `count` | Count Vectorizer | When the frequency of words is important |
| `tfidf` | TF-IDF Vectorizer | When the importance of words is important |
| `hashing` | Hashing Vectorizer | When memory is a concern |


### Environment Variables
The following environment variables can be set to customize the behavior of the application:
| Name | Description | Default |
| --- | --- | --- |
| `MODEL_DIR` | the directory where the trained models are stored | `models` |
| `DATA_DIR` | the directory where the datasets are stored | `data` |
| `CACHE_DIR` | the directory where cached files are stored | `.cache` |


## Implementation


### Architecture
The input text is first preprocessed and tokenized using `re` and `spaCy` where:
- Any HTML tags are removed
- Emojis and slang words are converted to their text form
- Stop words, punctuation and special characters are removed
- URLs, email addresses and numbers are removed
- Words are converted to lowercase
- Lemmatization is performed (words are converted to their base form based on the surrounding context)

After tokenization, feature extraction is performed on the tokens using the chosen vectorizer. Each vectorizer has its own advantages and disadvantages, and the choice of vectorizer can affect the speed and accuracy of the model (see [Vectorizers](#vectorizers)). The extracted features are then passed to the classifier which predicts the class which in this case is the sentiment of the text. Both the vectorizer and classifier are trained on the specified dataset.

```mermaid
%%{ init : { "flowchart" : { "curve" : "monotoneX" }}}%%
graph LR
  START:::hidden --> |text|Preprocessing

  subgraph Preprocessing
    direction TB
    A[Tokenizer]
    B1[HashingVectorizer]
    B2[CountVectorizer]
    B3[TfidfVectorizer]

    A --> B1
    A --> |tokens|B2
    A --> B3

    B1 --> C1:::hidden
    B2 --> C2:::hidden
    B3 --> C3:::hidden
  end

  Preprocessing --> |features|Classification

  subgraph Classification
    direction LR
    D1[LogisticRegression]
  end

  Classification --> |sentiment|END:::hidden

  classDef hidden display: none;
```


### Pre-trained Models
The following pre-trained models are available for use:
| Dataset | Vectorizer | Classifier | Features | Accuracy on test | Accuracy on self | Model |
| --- | --- | --- | --- | --- | --- | --- |
| `imdb50k` | `tfidf` | `LinearRegression` | 20 000 | 75.63% ± 4.73% | 89.24% ± 0.13% (5-fold) | [Here](models/imdb50k_tfidf_ft20000.pkl) |
| `sentiment140` | `tfidf` | `LinearRegression` | 20 000 | 75.63% ± 4.73% | 77.32% ± 0.28% (5-fold) | [Here](models/sentiment140_tfidf_ft20000.pkl) |
| `amazonreviews` | `tfidf` | `LinearRegression` | 20 000 | 65.49% ± 7.03% | 90.08% ± 0.00% (train-test) | [Here](models/amazonreviews_tfidf_ft20000.pkl) |


## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
