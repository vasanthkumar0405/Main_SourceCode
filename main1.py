import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize emotion pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to scrape tweets
def scrape_tweets(query, max_tweets):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets.append([tweet.date, tweet.content])
    return pd.DataFrame(tweets, columns=["Date", "Tweet"])

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text

# Detect emotion
def detect_emotion(text):
    try:
        result = emotion_classifier(text)[0]
        return result['label']
    except:
        return 'Unknown'

# Streamlit UI
st.title("Decoding Emotions Through Sentiment Analysis")

query = st.text_input("Enter a topic/keyword:", "mental health")
max_tweets = st.slider("Number of tweets to analyze", 50, 500, 100)

if st.button("Analyze Emotions"):
    with st.spinner("Scraping tweets and analyzing emotions..."):
        df = scrape_tweets(query, max_tweets)
        df['Cleaned_Tweet'] = df['Tweet'].apply(clean_text)
        df['Emotion'] = df['Cleaned_Tweet'].apply(detect_emotion)

        st.subheader("Sample Tweets with Emotions")
        st.dataframe(df[['Tweet', 'Emotion']].head(10))

        st.subheader("Emotion Distribution")
        plt.figure(figsize=(10,5))
        sns.countplot(data=df, x='Emotion', order=df['Emotion'].value_counts().index)
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Option to download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "emotion_analysis.csv", "text/csv")