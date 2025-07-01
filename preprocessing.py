import pandas as pd
import numpy as np
import re
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Define a fallback tokenizer in case NLTK's tokenizer has issues
def simple_tokenize(text):
    """A simple tokenizer that splits text on whitespace and punctuation"""
    # Remove punctuation and replace with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace
    return text.split()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize using our simple tokenizer to avoid NLTK issues
    tokens = simple_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(cleaned_tokens)

def load_and_preprocess_data(file_path, sample_size=None):
    """
    Load IMDB dataset, clean text, and encode labels
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Map sentiment to binary values
    df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Clean the text
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], 
        df['sentiment_binary'], 
        test_size=0.2, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test, df

# We don't need the tokenize functions anymore since we're using TF-IDF vectorization in the model
