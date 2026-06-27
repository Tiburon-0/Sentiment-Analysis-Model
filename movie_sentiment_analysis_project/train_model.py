import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer # transforms data and feeds data into MultinomialDB
from sklearn.naive_bayes import MultinomialNB

# persistence
import joblib as jl
from pathlib import Path

# save dataset

dataset = 'IMDB Dataset.csv'

# -----------[HELPER FUNCTIONS]-----------

# load and preprocess data

def load_and_preprocess(file):
    '''Loads and preprocesses dataset, returning inputs and outputs'''
    
    print(f'Loading dataset...')
    dataframe = pd.read_csv(Path(__file__).parent / file)
    print(f'Dataset loaded.')
    
    # assess dataset
    print(f'Analyzing dataset dimensions...')
    print(dataframe.head())
    print(f'This dataset has {dataframe.shape[0]} rows with {dataframe.shape[1]} columns.')

    # inputs and outputs

    print(f'Storing reviews as inputs...')
    inputs = dataframe['review'] 
    
    print(f'Storing sentiments as outputs...')
    outputs = dataframe['sentiment']

    print(f'Inputs and outputs stored successfully.')

    return inputs, outputs

def train_model(file):
    '''Vectorizes the preprocessed text data and feeds the vectorized data to the MultinomialNB classifier, 
    then trains the pipeline on entire dataset and returns the trained model'''

    X, y = load_and_preprocess(file)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()), # transformer
        ('clf', MultinomialNB()) # classifier
    ])

    print(f'Training the model...')

    fitted_model = pipeline.fit(X, y)

    print(f'Model trained.')

    return fitted_model

def save_model(trained_model, file_name):
    '''Saves trained_model for later access'''

    print(f'Saving the model...')
    jl.dump(trained_model, Path(__file__).parent / file_name)
    print(f'Model saved to {file_name}.')

def run_pipeline(data, **kwargs):
    '''Loads and preprocesses the data, trains the model, and saves the trained model'''

    saved_file_name = kwargs.get('saved_file_name', 'sentiment_model.pkl')

    trained_model = train_model(data)

    save_model(trained_model, saved_file_name)

    return f'Pipeline complete.'


run_pipeline(dataset, saved_file_name='sentiment_model.pkl')
