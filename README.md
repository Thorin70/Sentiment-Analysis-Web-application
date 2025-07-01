
# IMDB Sentiment Analysis Web Application 🎬

A streamlined web application for analyzing sentiment in IMDB movie reviews using machine learning techniques.

## Features 🌟

- **Data Exploration**: Visualize and understand the IMDB dataset through interactive charts and graphs
- **Model Training**: Train a sentiment analysis model using either Logistic Regression or Random Forest
- **Model Evaluation**: Assess model performance with various metrics (accuracy, precision, recall, F1-score)
- **Interactive Testing**: Test the model with your own movie reviews
- **Educational Resources**: Learn about neural networks and sentiment analysis concepts

## Technical Implementation 🛠️

- **Text Preprocessing**: Clean text, remove HTML tags, tokenization
- **Feature Extraction**: TF-IDF vectorization
- **Machine Learning Models**: 
  - Logistic Regression
  - Random Forest Classifier
- **Visualization**: Interactive plots using Plotly and Matplotlib
- **Web Interface**: Streamlit for the user interface

## Project Structure 📁

```
├── app.py              # Main Streamlit application
├── model.py            # Model implementation and training
├── preprocessing.py    # Text preprocessing functions
├── utils.py           # Utility functions
├── visualization.py   # Data visualization functions
└── attached_assets    # Dataset and other assets
```

## Getting Started 🚀

1. The application will automatically install required packages
2. Click the "Run" button to start the Streamlit server
3. Navigate through different sections using the sidebar
4. Upload your IMDB dataset or use the provided sample

## Pages 📑

1. **Introduction**: Project overview and features
2. **Data Exploration**: Dataset analysis and visualizations
3. **Model Training**: Train and configure the sentiment analysis model
4. **Model Evaluation**: View model performance metrics
5. **Interactive Testing**: Test the model with custom reviews
6. **Educational Resources**: Learn about the underlying concepts

## Requirements 📋

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Plotly

## Performance 📊

- Handles large datasets efficiently
- Fast text preprocessing
- Interactive visualizations
- Real-time sentiment predic
