# NLP Team Project: Predicting a Github Repository's Programming Language Using NLP

## Goals

The purpose of this project is to analyze the readme files of github repositories and develop a machine learning model to predict the programming language used in the repository using natural language processing. We obtained the dataset for this project from https://www.github.com. The code for acquiring this dataset is in our acquire.py file.

We are using this dataset for academic purposes only.

Initial Questions:

- What keywords readily identify a particular programming language?
- Can certain word combinations / n-grams in a readme improve identification of programming languages? 

## Executive Summary

- The dataset was split into train, validate, and test using a 60/20/20 split stratefied on language. The total number of observations after removing nulls was 679 readme files.
- The selected model is a decision tree classifier with a depth of 3. 

## Data Dictionary

1. repo : the namepath of the repository (string)
2. language : programming language (string)
3. readme_contents : readme file (string)
4. clean : cleaned version of readme_contents (string)
5. lemmatized: lemmatized version of readme_contents (string)
6. stopped: lemmatized version of readme_contents with stopwords removed (string)


## Project Planning

- Acquire the data from Github and save to a local pickle file
- Prepare the data with the intent to discover the main predictors of programming language; clean the data and engineer features if necessary; ensure that the data is tidy
- Split the data into train, validate, and test datasets using a 60/20/20 split and a random seed of 217
- Explore the data:
    - 
- Create graphical representations of the analyses
- Ask more questions about the data
- Document findings
- Train and test models:
    - Baseline?
    - Select key features and train multiple classification models
    - Test the model on the validate set, adjust for overfitting if necessary
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
    - from sklearn.metrics import classification_report
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

- Create models (decision tree, random forest, and logistical regression) with the most important selected features using sklearn.

- Train each model and evaluate its accuracy on both the train and validate sets.

- Select the best performing model and use it on the test set.

- Graph the results of the test using probabilities.

- Document each step of the process and your findings.


## Key Findings and Takeaways

