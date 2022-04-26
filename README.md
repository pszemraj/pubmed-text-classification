# PubMed Text Classification

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

### conda/mamba install

Installation is also possible through conda/mamba. To do this, use the provided `environment.yml` file. The file contains the following commands:

```
conda env create -f environment.yml
```

The environment.yml will create an environment called `ml4hc_p2`, to activate the environment, use the following command:

```
conda activate ml4hc_p2
```

## Dataset

Re-create the dataset by running `src/data/make_dataset.py`. Additional details are provided in the README.md file in the data folder.

## Results

### Transformer Model

Our findings after training a variety of transformer models on the dataset are shown below. Overall, the transformer model performs the best on the dataset. With respect to "hyperparameters", we found that the appropriate choice of a pretrained model was more important than any other, including unfreezing layers other than the classification head.

A high-level interactive EDA of our results on the test set for various models can be found at [this link](<https://pubmed-ml4hc-transformers.netlify.app/)

```
## Model files

Due to file size constraints some fine-tuned models will be available at the following huggingface page: [ml4pubmed](https://huggingface.co/ml4pubmed)

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

--------
