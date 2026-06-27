import streamlit as st
import pandas as pd
import joblib as jl

# ----[Caching the model]----
@st.cache_data
def load_model(**kwargs):
    '''Loads the pre-trained model and target variables'''

    # uses kwargs for flexibility, defaulting to the saved model
    saved_model = kwargs.get('saved_model', 'sentiment_model.pkl')

    # loads model
    model = jl.load(saved_model)
    return model

# ----[Load the model]----
classifier = load_model()

# ----[App layout]----
st.title('Movie Review Sentiment Classifier')
st.header('A Machine-Learning Approach to Sentiment Analysis')
st.subheader('Overview')
st.markdown('''This app uses a Multinomial Naive Bayes (MNB) classification model
             to analyze and predict the sentiment of user text.📊 For more info 
            on MNB, visit\n:
             [scikit-learn.org](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)''')

# ----[User Input Interface]----
user_text = [st.text_area('Enter a movie review to analyze:')]
x = st.button('Analyze')

# ----[Make Predictions and Display Results]----
if x:
    if user_text:
        st.write(f'Analyzing review: {user_text[0]}...')
        st.subheader('Prediction:')
        prediction = classifier.predict(user_text)
        prediction_proba = classifier.predict_proba(user_text)
        if prediction == 'positive':
            st.write(f'The review is: {prediction[0]}🥂')
        else:
            st.write(f'The review is: {prediction[0]}😒')
    else:
        st.write('Please enter a movie review for analysis.')