# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

@st.cache_data
def scrape_wikipedia_text(url):
    """Scrapes text content from a Wikipedia page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = []
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        for p in content_div.find_all("p"):
            if p.text.strip():
                paragraphs.append(p.text)
    raw_text = ' '.join(paragraphs)
    return raw_text

def clean_text(text):
    """Cleans the input text by removing citations, special characters, and extra spaces."""
    text = re.sub(r'\[\d+\]', '', text)  # Remove citations like [1], [2], etc.
    text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters and digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def get_sentiment(sentence):
    """Analyzes the sentiment of a sentence using TextBlob."""
    analysis = TextBlob(sentence)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

@st.cache_data
def analyze_sentiment(text):
    """Tokenizes text into sentences, calculates sentiment for each, and returns a DataFrame."""
    sentences = sent_tokenize(text)
    sentiment_results = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        sentiment = get_sentiment(sentence)
        sentiment_results.append({
            'sentence': sentence,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        })
    sentiment_df = pd.DataFrame(sentiment_results)
    return sentiment_df

@st.cache_data
def remove_stopwords(text):
    """Removes stop words from the given text."""
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words

def generate_wordcloud(words):
    """Generates and displays a word cloud from a list of words."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def plot_common_words(words, num_words=20):
    """Plots the most common words."""
    word_counts = Counter(words)
    common_words = word_counts.most_common(num_words)
    words_df = pd.DataFrame(common_words, columns=['word', 'count'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words_df['word'], words_df['count'])
    ax.set_title(f'Top {num_words} Most Common Words')
    ax.set_xlabel('Count')
    ax.set_ylabel('Word')
    ax.invert_yaxis()  # Invert y-axis for better readability
    st.pyplot(fig)

def train_and_evaluate_model(df, model_type='logistic_regression'):
    """
    Trains and evaluates a sentiment analysis model.

    Args:
        df (pd.DataFrame): DataFrame containing 'sentence' and 'sentiment' columns.
        model_type (str): 'logistic_regression' or 'naive_bayes'.

    Returns:
        tuple: Model, TF-IDF vectorizer, and classification report.
    """
    # Filter out neutral sentiments
    filtered_df = df[df['sentiment'] != 'neutral'].copy()

    # Prepare data
    X = filtered_df['sentence']
    y = filtered_df['sentiment']

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

    # Model Training and Evaluation
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
        model.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'naive_bayes'.")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    return model, tfidf_vectorizer, report, confusion, X_test, y_test

def display_confusion_matrix(confusion, model):
    """Displays the confusion matrix."""
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

def analyze_user_text(text):
    """Analyzes the sentiment of user-provided text."""
    if text.strip():  # Check if the input is not empty
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Polarity score (-1 to 1)
        subjectivity = blob.sentiment.subjectivity  # Subjectivity score (0 to 1)

        # Determine sentiment based on polarity
        if polarity > 0.1:
            sentiment = "Positive 😊"
        elif polarity < -0.1:
            sentiment = "Negative 😢"
        else:
            sentiment = "Neutral 😐"

        # Display results
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity:** {polarity:.2f}")
        st.write(f"**Subjectivity:** {subjectivity:.2f}")
    else:
        st.warning("Please enter some text to analyze.")

def main():
    # Create tabs
    tab1, tab2 = st.tabs(["Sentiment Analysis of South Korea", "Interactive Sentiment Analysis"])

    # Tab 1: Sentiment Analysis of South Korea
    with tab1:
        st.title("Sentiment Analysis of South Korea")

        # Step 1: Web scrape Wikipedia page for South Korea
        url = "https://en.wikipedia.org/wiki/South_Korea"
        raw_text = scrape_wikipedia_text(url)

        # Step 2: Clean the scraped text
        cleaned_text = clean_text(raw_text)

        # Step 3: Analyze sentiment
        sentiment_df = analyze_sentiment(cleaned_text)

        # Display Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color=['gray', 'green', 'red'], ax=ax)
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)

        # Step 4: Remove Stopwords and generate word cloud
        st.subheader("Word Cloud")
        filtered_words = remove_stopwords(cleaned_text)
        generate_wordcloud(filtered_words)

        # Step 5: Display most common words
        st.subheader("Most Common Words")
        plot_common_words(filtered_words)

        # Step 6: Train and Evaluate Model
        st.subheader("Model Training and Evaluation")
        model_type = st.selectbox("Choose Model Type", ['logistic_regression', 'naive_bayes'])
        model, tfidf_vectorizer, report, confusion, X_test, y_test = train_and_evaluate_model(sentiment_df, model_type)

        # Display Classification Report
        st.write("Classification Report:")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Display Confusion Matrix
        st.write("Confusion Matrix:")
        display_confusion_matrix(confusion, model)

        # Display Accuracy Score
        st.write("Accuracy Score:", accuracy_score(y_test, model.predict(X_test)))

    # Tab 2: Interactive Sentiment Analysis
    with tab2:
        st.title("Interactive Sentiment Analysis")
        st.subheader("Enter text to analyze its sentiment!")

        # Text input from the user
        user_input = st.text_area("Enter your text here:")

        # Analyze button
        if st.button("Analyze Sentiment"):
            analyze_user_text(user_input)

if __name__ == "__main__":
    main()
