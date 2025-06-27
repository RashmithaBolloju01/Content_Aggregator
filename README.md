# Smart Content Aggregator

Smart Content Aggregator is an AI-driven application that extracts, summarizes, and ranks relevant content from scanned documents and real-time web sources. It combines OCR, NLP, and web scraping to assist users in accessing useful, contextual information efficiently through a clean and interactive interface.

## Project Overview

This project is designed to automate content extraction and relevance-based summarization using image-based documents, live news feeds, and web content. The system leverages pre-trained models and statistical methods to perform topic-based filtering, summarization, and visualization of information.

## Features

- OCR-powered image and PDF text extraction using Tesseract
- Real-time web scraping from news articles using Google News RSS and Newspaper3k
- Keyword extraction, topic-based summarization using NLP (NLTK, TF-IDF)
- Relevance scoring using softmax-based normalization
- Data visualization of summary relevance using Altair charts
- Interactive front-end interface using Streamlit
- Basic user login and session management (for multi-user workflows)

## Technologies Used

| Category        | Tools and Libraries                          |
|----------------|-----------------------------------------------|
| Programming     | Python                                        |
| OCR             | Tesseract, Pillow (PIL)                       |
| Web Scraping    | BeautifulSoup, Newspaper3k, Requests          |
| NLP             | NLTK, Scikit-learn (TF-IDF), Softmax Scoring |
| Visualization   | Altair, Pandas                                |
| Front-End       | Streamlit                                     |
| APIs            | Google News RSS, Wikipedia API                |

## Project Structure

Content_Aggregator/
├── main.py # Streamlit-based interface
├── ocr_module.py # Handles OCR and image preprocessing
├── news_scraper.py # Fetches and parses live news articles
├── summarizer.py # Text preprocessing and summarization
├── visualizer.py # Data relevance scoring and visualization
├── utils/ # Helper functions and configurations
└── README.md


## How to Run

1. Clone the repository:

git clone https://github.com/RashmithaBolloju01/Content_Aggregator.git
cd Content_Aggregator

(Optional) Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run the application:
streamlit run main.py
Sample Use Case

A user uploads a scanned academic paper or image-based document and selects a keyword topic such as "Artificial Intelligence." The application extracts the text, pulls related live news articles and Wikipedia summaries, filters them using keyword-based similarity, and ranks them by contextual relevance. The output is presented with brief summaries and visual relevance scores.

License

This project is licensed under the MIT License.
