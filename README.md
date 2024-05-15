Sentiment Analysis
---

### Usage
1. Clone the repository
2. `cd` into the repository
3. Run `just install` to install the dependencies
4. Run `just run --help` to see the available commands


### Required tools
- `just`
- `poetry`

### TODO
- [ ] CLI using `click` (commands: predict, train, evaluate) with settings set via flags or environment variables
- [ ] GUI using `gradio` (tabs: predict, train, evaluate, compare, settings)
- [ ] For the sklearn model, add more classifiers
- [ ] Use random search for hyperparameter tuning and grid search for fine-tuning
- [ ] Finish the text pre-processing transformer
- [ ] For vectorization, use custom stopwords
- [ ] Write own tokenizer/vectorizer
- [ ] Add more datasets
- [ ] Add more models (e.g. BERT)
- [ ] Write tests
- [ ] Use xgboost?
- [ ] Deploy to huggingface?
