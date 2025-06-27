import streamlit as st
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import random

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

import re
from bs4 import BeautifulSoup

# --- Clean Summary ---
def clean_summary(raw_html):
    # Use BeautifulSoup to remove HTML tags
    soup = BeautifulSoup(raw_html, 'html.parser')
    clean_text = soup.get_text()
    return clean_text
# --- TF-IDF-based Summarization ---
def tfidf_summarize(text, num_sentences=3):
    sentences = text.split('. ')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).flatten().tolist()[0]
    ranked_sentences = sorted(
        ((score, sentence) for sentence, score in zip(sentences, sentence_scores)),
        reverse=True,
    )
    summary = ". ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary

# --- Softmax Function ---
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

# Example Usage in Relevance Score Normalization
def normalize_relevance_scores(relevance_scores):
    """Normalize relevance scores using softmax."""
    return softmax(np.array(relevance_scores))

# # Example: Replace lambda with softmax
# relevance_scores = [2, 1, 0.5, 3]  # Example relevance scores
# normalized_scores = normalize_relevance_scores(relevance_scores)
# print("Normalized Scores:", normalized_scores)
# --- Fetch RSS News ---
import random

# --- Fetch RSS News ---
def fetch_rss_news(query):
    query = query.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)

    # Check if the feed is valid and contains entries
    if not feed.entries:
        return []  # Return an empty list if no articles are found

    articles = []
    relevance_scores = []  # Collect relevance scores for normalization
    random_summaries = [
        f"This is a random summary about {query}.",
        f"Learn more about {query} in this article.",
        f"Find out the latest news on {query}.",
        f"Explore insights related to {query}.",
        f"Breaking news and updates on {query}."
    ]

    for entry in feed.entries[:10]:  # Fetch up to 10 articles
        relevance_score = len(set(query.lower().split()) & set(entry.title.lower().split()))
        relevance_scores.append(relevance_score)
        summary = random.choice(random_summaries)  # Generate a random summary
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published if 'published' in entry else 'N/A',
            "source": entry.source.title if 'source' in entry else 'Google News',
            "relevance_score": relevance_score,
            "summary": summary
        })

    # Normalize relevance scores using softmax
    normalized_scores = softmax(np.array(relevance_scores))
    for i, article in enumerate(articles):
        article['relevance_score'] = normalized_scores[i]

    return sorted(articles, key=lambda x: x['relevance_score'], reverse=True)
# --- Streamlit Setup ---
st.set_page_config(page_title="RSS Content Aggregator", page_icon="🔗", layout="wide")

# --- Login Page ---
def login():
    st.title("🔐 CONTENT AGGREGATION (RSS ONLY)")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
            else:
                st.error("❌ Invalid credentials.")

# --- Main App ---
def main_app():
    st.title("🔗 RSS Content Aggregator")
    st.write("Enter a topic to fetch the latest news articles using RSS feeds.")

    query = st.text_input("🔍 Enter your topic:")
    if query:
        with st.spinner("Fetching articles from RSS feeds..."):
            articles = fetch_rss_news(query)

        if articles:
            # Display each article with its details
            for idx, article in enumerate(articles):  # Use `idx` to generate unique keys
                st.markdown(f"### 🔗 [{article['title']}]({article['link']})")
                st.caption(f"Source: {article['source']} | Published: {article['published']}")
                st.write(f"**Relevance Score:** {article['relevance_score']}")
                st.text_area(
                    f"🔹 **Summary:**",
                    article['summary'],
                    height=150,
                    disabled=True,
                    key=f"summary_{idx}"  # Unique key for each text_area
                )
                st.markdown("---")

            # Add a relevance score graph at the end
            st.write("### Relevance Score Graph")
            df = pd.DataFrame(articles)
            st.bar_chart(df[["title", "relevance_score"]].set_index("title"))
        else:
            st.warning("⚠️ No articles found for the given topic. Please try a different query.")
# --- Auth Routing ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    main_app()