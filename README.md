# 🌍 Sentiment Analysis on South Korea (Wikipedia) & Interactive Text Analyzer

This project uses **Natural Language Processing (NLP)** techniques to perform sentiment analysis on the Wikipedia page for **South Korea**, and also allows users to interactively analyze the sentiment of any custom text. The project is built with **Streamlit**, making it accessible via a user-friendly web interface.

---

## 🚀 Features

### 📄 Tab 1: Sentiment Analysis of South Korea
- Scrapes the latest text content from [South Korea Wikipedia page](https://en.wikipedia.org/wiki/South_Korea)
- Cleans and processes text using `nltk` and `TextBlob`
- Performs sentence-level sentiment classification (positive, negative, neutral)
- Visualizes:
  - Sentiment distribution (bar chart)
  - Word cloud of non-stopwords
  - Top 20 most common words
- Trains and evaluates sentiment classification models:
  - **Logistic Regression**
  - **Naive Bayes**
- Displays classification report, confusion matrix, and model accuracy

### ✍️ Tab 2: Interactive Sentiment Analysis
- Users can input any text
- Analyzes sentiment with polarity and subjectivity using `TextBlob`
- Displays a friendly emoji-based sentiment result

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** - Interactive web app
- **BeautifulSoup** - Web scraping
- **NLTK** - Tokenization, stopword removal
- **TextBlob** - Sentiment analysis
- **Scikit-learn** - Machine learning models (Logistic Regression, Naive Bayes)
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Word cloud generation

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-south-korea.git
   cd sentiment-analysis-south-korea
