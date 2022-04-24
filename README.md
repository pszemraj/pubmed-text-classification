# pubmed_textclassification

==============================

ETHZ Machine Learning for Healthcare Problem 2: classification of pubmed paper sentences or text into document sections.

by: Peter Szemraj & Lou Ancillon

## Approach

The pubmed_textclassification problem is a classification problem. The goal is to classify input text taken from a document into one of five classes:  background, objective; method, result; conclusion.

This is done through three different approaches:

1. TF-IDF (Term Frequency - Inverse Document Frequency) to vectorize the input text.
2. word2vec to vectorize the input text.
3. transformer model training to implement a neural network, implicitly learning the word embeddings and classifying the input text.

## Installing

Installion is completed through installing the packages listed in the `requirements.txt` file.

1. clone the repository
2. cd into the directory that you cloned
3. run `pip install -r requirements.txt`

## Project Organization

------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

--------

_Note:_ This project is hosted on the [cookiecutter data science project template](ttps://drivendata.github.io/cookiecutter-data-science/)
