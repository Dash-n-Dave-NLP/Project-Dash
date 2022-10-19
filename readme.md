# NLP Team Project: Predicting a Github Repository's Programming Language Using NLP

## Goals

The purpose of this project is to analyze the readme files of github repositories and develop a machine learning model to predict the programming language used in the repository using natural language processing. We obtained the dataset for this project from https://www.github.com. The code for acquiring this dataset is in our acquire.py file.

We are using this dataset for academic purposes only.

Initial Questions:

- What keywords readily identify a particular programming language?
- Can certain word combinations / n-grams in a readme improve identification of programming languages? 

## Executive Summary

- The dataset was split into train, validate, and test using a 60/20/20 split stratefied on language. The total number of observations after removing nulls was 601 readme files.
- We trained and evaluated three models: Decision Tree Classifier, Logistic Regression, and Naive-Bayes Multinomial Classifier. For each model we used a Count Vectorizer, Count Vectorizer with bigrams, and a TF-IDF Vectorizer. The TF-IDF Vectorizer produced the best overall accuracy in each model.
- The selected model is a Logistic Regression model using a TF-IDF Vectorizer. The model performed at 97 percent accuracy on train, but accuracy dipped to 65 percent on the validate set. When model performed at 58 percent accuracy on the test set. This is 37 percent above the baseline accuracy, which is 21 percent.

## Data Dictionary

1. repo : the namepath of the repository (string)
2. language : programming language (string)
3. original : readme file (string)
4. clean : cleaned version of original (string)
5. stemmed : stemmed version of original (string)
6. lemmatized: lemmatized version of original (string)
7. original_length: number of words in each original observation (int64)
8. true_clean : cleaned and lemmatized version of original with stopwords removed (string)


## Project Planning

- Acquire the data from Github and save to a local pickle file
- Prepare the data with the intent to discover the main predictors of programming language; clean the data and engineer features if necessary; ensure that the data is tidy
- Split the data into train, validate, and test datasets using a 60/20/20 split and a random seed of 217
- Explore the data:
    - Find top 20 words for each programming language
    - Find top 20 bigrams for each programming language
- Create graphical representations of the analyses
- Ask more questions about the data
- Document findings
- Train and test models:
    - Baseline accuracy with "other" language category is 16.7 percent; with "other" removed, baseline accuracy is 21 percent
    - Select vectorizer and train multiple classification models
    - Test the model on the validate set, adjust model parameters if necessary
- Select the best model for the project goals:
    - Determine which model performs best on the validate set
- Test and evaluate the model:
    - Use the model on the test set and evaluate its performance (accuracy, precision, recall, f1, etc.)
- Visualize the model's performance on the test set
- Document key findings and takeaways, answer the questions
- Create a final report

## How to Reproduce this Project

- In order to reproduce this project, you will need access to Github or the data file. Acquire the database from https://www.github.com using the function in our acquire.py file. The prepare.py file has the necessary functions to prepare and split the dataset.

- You will need to import the following python libraries into a python file or jupyter notebook:

    - import pandas as pd
    - import numpy as np
    - import matplotlib.pyplot as plt
    - import seaborn as sns
    - from scipy import stats
    - from sklearn.model_selection import train_test_split
    - from sklearn.tree import DecisionTreeClassifier, plot_tree
    - from sklearn.metrics import classification_report, accuracy_score
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    - from sklearn.naive_bayes import MultinomialNB, CategoricalNB
    - import re
    - import unicodedata
    - import nltk
    - from wordcloud import WordCloud
    - from requests import get
    - from bs4 import BeautifulSoup
    - from nltk.corpus import stopwords
    - import acquire
    - import prepare
    - import model
    - import warnings
    - warnings.filterwarnings("ignore")

- Prepare and split the dataset. The code for these steps can be found in the acquire.py file and prepare.py file within this repository.

- Use pandas to explore the dataframe and matplotlib to visualize word counts and n-grams.

- Analyze words in each programming language to find the most used words.

- Create models (decision tree, Naive-Bayes classifier, and logistic regression) using sklearn.

- Train each model and evaluate its accuracy on both the train and validate sets.

- Select the best performing model and use it on the test set.

- Graph the results of the test using probabilities.

- Document each step of the process and your findings.


## Key Findings and Takeaways

After training and evaluating three models using both a single-word count vectorizer, bigram count vectorizer, and TF-IDF vectorizer, the logistic regression model provdided the best overall performance on the validate set. Fitting of the models resulted in over 90 percent accuracy on train; however, the accuracy of all models fell considerably on the validate set. 