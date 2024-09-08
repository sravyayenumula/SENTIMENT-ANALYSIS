import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def app():
    st.write("# Real Time Sentiment Analysis")

    user_input = st.text_input("Please enter your review >>: ")
    nltk.download("vader_lexicon")
    s = SentimentIntensityAnalyzer()
    score = s.polarity_scores(user_input)
    st.write('Positive Score',score["pos"])
    st.write('Neutral Score',score["neu"])
    st.write('Negative Score',score["neg"])
    st.write('Compound Score',score["compound"])
    if score["compound"] >=0.05:
        st.markdown("<h1 style='text-align: center; color: #f6c453;font-size: 100px;';>:-)</h1>", unsafe_allow_html=True)
    elif score["compound"] > -0.05 and score["compound"]< 0.05:
        st.markdown("<h1 style='text-align: center; color: #000000;font-size: 100px;';>:-|</h1>", unsafe_allow_html=True)
    elif score["compound"] <= -0.05:
        st.markdown("<h1 style='text-align: center; color: #FF0000;font-size: 100px;';> >:-(</h1>", unsafe_allow_html=True)

