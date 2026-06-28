# Movie Review Sentiment Classifier

A machine learning web application that analyzes and predicts the sentiment of user text inputs. Built on the Multinomial Naive Bayes classification model. Data sourced from IMDB Dataset of 50k Movie Reviews.

## Requirements

* Python 3.8+
* Dependencies listed in `requirements.txt`

## Setup and Installation

* Clone the repository: `git clone https://github.com/Tiburon-0/Sentiment-Analysis-Model`
* Navigate into the project directory: `cd Sentiment-Analysis-Model`
* Install dependencies: `pip install -r requirements.txt`
* Download the IMDB dataset from [kaggle.com](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place the .csv file in the project root (required only if retraining)

## Train the Model

* Given the model is provided (saved in `sentiment_model.pkl`), there is no need to run `train_model.py` unless you aim to assess its functionality.

* If you aim to assess the functionality of `train_model.py`, execute the command `python train_model.py` to run the training script and save the model.

* Note: This command only needs to be run once, given that the model is saved to `sentiment_model.pkl` via `joblib.dump()` and loaded on startup via `joblib.load()`

## Run the App

* Launch the streamlit app via `streamlit run app.py`
* Open your browser to `http://localhost:8501/`
* Enter any movie review in the text box, type the command `Ctrl+Enter`, click `Analyze`
