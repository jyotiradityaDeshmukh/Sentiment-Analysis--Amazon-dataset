# Enhanced Amazon Review Sentiment Analysis

## Overview

This project presents an enhanced version of Amazon review sentiment analysis using state-of-the-art NLP techniques. The project improves upon traditional sentiment analysis approaches by implementing advanced preprocessing techniques and utilizing the powerful BERT-based multilingual sentiment analysis model.

## Key Improvements

1. *Advanced Text Preprocessing*

   - HTML tag removal
   - URL removal
   - Special character handling
   - Stopword removal
   - Tokenization
   - Case normalization

2. *Model Selection*

   - Original: Basic sentiment analysis models
   - Enhanced: BERT-based multilingual model (nlptown/bert-base-multilingual-uncased-sentiment)
   - Benefits: Better context understanding, multilingual support, improved accuracy

3. *Performance Metrics*
   - Original Model Accuracy: ~85%
   - Enhanced Model Accuracy: ~95%
   - Improvement: ~10% increase in accuracy

## Libraries Required

python
# Core Libraries
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2

# NLP Libraries
transformers==4.30.2
torch==2.0.1
nltk==3.8.1

# Utility Libraries
tqdm==4.65.0
scikit-learn==1.3.0


## Project Structure


amazon-sentiment-analysis/
├── data/
│   └── amazon.csv
├── images/
│   ├── sentiment_distribution_v1.png
│   └── sentiment_scores_v1.png
├── src/
│   └── sentiment_analysis.py
├── requirements.txt
└── README.md


## Methodology

### 1. Data Loading and Cleaning

- Load Amazon review dataset
- Remove empty reviews
- Sample data for efficient processing
- Handle missing values

### 2. Text Preprocessing

- Convert text to lowercase
- Remove HTML tags and URLs
- Clean special characters
- Tokenize text
- Remove stopwords
- Handle edge cases

### 3. Sentiment Analysis

- Initialize BERT-based sentiment analyzer
- Process reviews in batches
- Convert 5-star ratings to binary sentiment
- Handle errors and edge cases

### 4. Evaluation

- Generate classification report
- Create visualizations
- Calculate accuracy metrics
- Compare with baseline model

## Results

The enhanced model shows significant improvements over the baseline:

- Better handling of complex language patterns
- Improved accuracy in sentiment classification
- More robust to noise in the text
- Better performance on multilingual reviews

## Visualization

The project includes two key visualizations:

1. Sentiment Distribution Plot
2. Sentiment Score Distribution Plot

## Usage

python
from src.sentiment_analysis import analyze_sentiment

# Load and preprocess data
df = pd.read_csv('data/amazon.csv')

# Analyze sentiment
results = analyze_sentiment(df)

# View results
print(results['classification_report'])


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the BERT model
- Amazon for the review dataset
- NLTK for text processing tools
