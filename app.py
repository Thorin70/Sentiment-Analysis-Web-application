import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import re

# Import custom modules
from preprocessing import (
    load_and_preprocess_data,
    clean_text
)
from model import SentimentModel, explain_activation_functions, explain_cross_entropy

# Simple text cleaning function for interactive testing
def clean_text_simple(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
from visualization import (
    plot_training_history, 
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve,
    plot_word_distribution,
    plot_sentiment_distribution,
    plot_review_length_distribution,
    plot_activation_functions,
    plot_binary_cross_entropy
)
from utils import (
    get_example_reviews, 
    display_cross_entropy_calculation, 
    explain_neural_network_components
)

# Set page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
    
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
    
if 'trained' not in st.session_state:
    st.session_state.trained = False
    
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
    
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
    
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
    
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
    
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "Data Exploration", "Model Training", "Model Evaluation", "Interactive Testing", "Educational Resources"]
)

# Load data function
@st.cache_data
def load_data(file_path, sample_size=None):
    X_train, X_test, y_train, y_test, df = load_and_preprocess_data(file_path, sample_size)
    return X_train, X_test, y_train, y_test, df

# Introduction Page
if page == "Introduction":
    st.title("📊 IMDB Sentiment Analysis using Deep Learning")
    
    st.markdown("""
    ## Project Overview
    
    This application demonstrates sentiment analysis on IMDB movie reviews using deep learning techniques. 
    The goal is to classify reviews as positive or negative based on their textual content.
    
    ### Key Features:
    
    - **Data Exploration**: Visualize and understand the IMDB dataset
    - **Model Training**: Build and train a Deep Neural Network for sentiment classification
    - **Model Evaluation**: Assess model performance with various metrics
    - **Interactive Testing**: Test the model with your own reviews
    - **Educational Resources**: Learn about the key concepts of neural networks
    
    ### Dataset:
    
    The IMDB dataset contains 50,000 movie reviews labeled as positive or negative. We'll use this data to train our deep learning model.
    
    ### Technical Implementation:
    
    - **Text Preprocessing**: Cleaning, tokenization, and vectorization
    - **Neural Network Architecture**: Feedforward DNN with multiple layers
    - **Activation Functions**: ReLU, Sigmoid, and Tanh
    - **Loss Function**: Binary Cross-Entropy
    - **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score
    
    ### Getting Started:
    
    Use the navigation panel on the left to explore the different sections of the application.
    """)
    
    st.info("This project demonstrates how deep learning can be applied to natural language processing tasks like sentiment analysis.")

# Data Exploration Page
elif page == "Data Exploration":
    st.title("🔍 Data Exploration")
    
    # Load the dataset
    file_path = "attached_assets/IMDB Dataset.csv"
    
    if os.path.exists(file_path):
        # Sample size slider
        sample_size = st.slider(
            "Select sample size for exploration", 
            min_value=100, 
            max_value=10000, 
            value=5000, 
            step=100
        )
        
        with st.spinner("Loading and preprocessing data..."):
            X_train, X_test, y_train, y_test, df = load_data(file_path, sample_size)
        
        # Display the dataframe
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Positive Reviews", len(df[df['sentiment'] == 'positive']))
        with col3:
            st.metric("Negative Reviews", len(df[df['sentiment'] == 'negative']))
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        fig_sentiment = plot_sentiment_distribution(df)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Review length distribution
        st.subheader("Review Length Distribution")
        fig_length = plot_review_length_distribution(df)
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Word clouds
        st.subheader("Word Clouds by Sentiment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Positive Reviews")
            fig_pos = plot_word_distribution(df, 'cleaned_review', 'positive')
            st.pyplot(fig_pos)
        
        with col2:
            st.markdown("### Negative Reviews")
            fig_neg = plot_word_distribution(df, 'cleaned_review', 'negative')
            st.pyplot(fig_neg)
        
        # Display sample reviews
        st.subheader("Sample Reviews")
        
        tab1, tab2 = st.tabs(["Positive Reviews", "Negative Reviews"])
        
        with tab1:
            positive_samples = df[df['sentiment'] == 'positive'].sample(5)
            for i, row in enumerate(positive_samples.itertuples(), 1):
                st.markdown(f"**Review {i}**: {row.review[:300]}...")
        
        with tab2:
            negative_samples = df[df['sentiment'] == 'negative'].sample(5)
            for i, row in enumerate(negative_samples.itertuples(), 1):
                st.markdown(f"**Review {i}**: {row.review[:300]}...")
    
    else:
        st.error(f"Dataset not found at {file_path}. Please make sure the file exists.")

# Model Training Page
elif page == "Model Training":
    st.title("🧠 Model Training")
    
    # Load the dataset
    file_path = "attached_assets/IMDB Dataset.csv"
    
    if os.path.exists(file_path):
        # Training parameters
        st.subheader("Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size = st.slider(
                "Training sample size", 
                min_value=1000, 
                max_value=50000, 
                value=10000, 
                step=1000,
                help="Number of reviews to use for training and testing"
            )
            
            epochs = st.slider(
                "Number of epochs", 
                min_value=1, 
                max_value=10, 
                value=3
            )
            
            batch_size = st.select_slider(
                "Batch size", 
                options=[32, 64, 128, 256], 
                value=64
            )
        
        with col2:
            model_type = st.selectbox(
                "Model architecture",
                ["feedforward", "cnn"],
                help="Feedforward is a fully connected neural network, CNN uses convolutional layers"
            )
            
            max_words = st.slider(
                "Maximum vocabulary size", 
                min_value=1000, 
                max_value=20000, 
                value=10000
            )
            
            max_len = st.slider(
                "Maximum sequence length", 
                min_value=50, 
                max_value=500, 
                value=200
            )
        
        # Load and preprocess data when requested
        if st.button("Load and Preprocess Data"):
            with st.spinner("Loading and preprocessing data..."):
                X_train, X_test, y_train, y_test, df = load_data(file_path, sample_size)
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                st.success(f"Data loaded and preprocessed! Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Display model architecture
        if 'X_train' in st.session_state and st.session_state.X_train is not None:
            st.subheader("Model Architecture")
            
            # Build the model
            sentiment_model = SentimentModel(max_features=max_words)
            model = sentiment_model.build_model(model_type="logistic" if model_type == "feedforward" else "random_forest")
            
            # Display model summary for scikit-learn models
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                model_summary = [f"Model type: {type(model).__name__}", "Parameters:"]
                for param, value in model_params.items():
                    model_summary.append(f"  {param}: {value}")
                st.code("\n".join(model_summary))
            else:
                st.code("Model information not available")
            
            # Store the model in session state
            st.session_state.model = sentiment_model
            
            # Train the model
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a few minutes."):
                    start_time = time.time()
                    
                    # Split training data for validation
                    val_split = int(0.1 * len(st.session_state.X_train))
                    X_val = st.session_state.X_train[:val_split]
                    y_val = st.session_state.y_train[:val_split]
                    X_train_final = st.session_state.X_train[val_split:]
                    y_train_final = st.session_state.y_train[val_split:]
                    
                    # Train the model
                    history = sentiment_model.train(
                        X_train_final, 
                        y_train_final, 
                        X_val, 
                        y_val, 
                        batch_size=batch_size, 
                        epochs=epochs
                    )
                    
                    end_time = time.time()
                    training_time = end_time - start_time
                    
                    # Store training results
                    st.session_state.training_history = history
                    st.session_state.trained = True
                    
                    st.success(f"Model trained successfully in {training_time:.2f} seconds!")
                    
                    # Plot training history
                    st.subheader("Training History")
                    fig = plot_training_history(history)
                    st.pyplot(fig)
                    
                    # Display metrics for scikit-learn models
                    st.subheader("Final Training Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Training Accuracy", f"{history['accuracy'][0]:.4f}")
                    with col2:
                        st.metric("Validation Accuracy", f"{history['val_accuracy'][0]:.4f}")
                    with col3:
                        st.metric("Training Loss", f"{history['loss'][0]:.4f}")
                    with col4:
                        st.metric("Validation Loss", f"{history['val_loss'][0]:.4f}")
        
        else:
            st.info("Please load and preprocess the data first.")
    
    else:
        st.error(f"Dataset not found at {file_path}. Please make sure the file exists.")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("📏 Model Evaluation")
    
    if st.session_state.trained and st.session_state.model is not None:
        # Evaluate the model
        if st.button("Evaluate Model on Test Data"):
            with st.spinner("Evaluating model..."):
                # Get predictions
                y_pred = st.session_state.model.predict(st.session_state.X_test)
                st.session_state.y_pred = y_pred
                
                # Evaluate model
                evaluation_results = st.session_state.model.evaluate(st.session_state.X_test, st.session_state.y_test)
                st.session_state.evaluation_results = evaluation_results
                
                st.success("Model evaluated successfully!")
        
        if st.session_state.evaluation_results is not None:
            # Display metrics
            st.subheader("Evaluation Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{st.session_state.evaluation_results[1]:.4f}")
            with col2:
                st.metric("Loss", f"{st.session_state.evaluation_results[0]:.4f}")
            with col3:
                st.metric("Precision", f"{st.session_state.evaluation_results[2]:.4f}")
            with col4:
                st.metric("Recall", f"{st.session_state.evaluation_results[3]:.4f}")
            
            # Display visualizations
            st.subheader("Evaluation Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])
            
            with tab1:
                fig_cm = plot_confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with tab2:
                fig_roc = plot_roc_curve(st.session_state.y_test, st.session_state.y_pred)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with tab3:
                fig_pr = plot_precision_recall_curve(st.session_state.y_test, st.session_state.y_pred)
                st.plotly_chart(fig_pr, use_container_width=True)
            
            # Display misclassified examples
            st.subheader("Misclassified Examples")
            
            # Convert predictions to binary
            y_pred_binary = (st.session_state.y_pred > 0.5).flatten().astype(int)
            
            # Find misclassified examples
            misclassified_indices = np.where(y_pred_binary != st.session_state.y_test)[0]
            
            if len(misclassified_indices) > 0:
                # Load the dataset
                file_path = "attached_assets/IMDB Dataset.csv"
                _, _, _, _, df = load_data(file_path, 50000)  # Load all data to get original reviews
                
                # Get a subset of misclassified examples
                sample_indices = np.random.choice(misclassified_indices, min(5, len(misclassified_indices)), replace=False)
                
                for i, idx in enumerate(sample_indices, 1):
                    actual = "Positive" if st.session_state.y_test.iloc[idx] == 1 else "Negative"
                    predicted = "Positive" if y_pred_binary[idx] == 1 else "Negative"
                    confidence = st.session_state.y_pred[idx][0] if y_pred_binary[idx] == 1 else 1 - st.session_state.y_pred[idx][0]
                    
                    st.markdown(f"**Example {i}**")
                    st.markdown(f"**Actual Sentiment:** {actual}")
                    st.markdown(f"**Predicted Sentiment:** {predicted} (Confidence: {confidence:.2f})")
                    
                    # Get the original review text
                    review_text = df.iloc[idx + 40000]['review'] if idx >= 10000 else df.iloc[idx]['review']
                    st.markdown(f"**Review:** {review_text[:300]}...")
                    st.markdown("---")
            else:
                st.success("No misclassified examples found in the sample!")
    
    else:
        st.warning("Please train the model first on the 'Model Training' page.")

# Interactive Testing Page
elif page == "Interactive Testing":
    st.title("🧪 Interactive Testing")
    
    if st.session_state.trained and st.session_state.model is not None:
        st.subheader("Test with Your Own Review")
        
        # Text input for user review
        user_review = st.text_area(
            "Enter a movie review:",
            height=150,
            placeholder="Write your movie review here..."
        )
        
        # Test with user input
        if st.button("Analyze Sentiment") and user_review:
            with st.spinner("Analyzing sentiment..."):
                # Use simple text cleaning to avoid NLTK issues
                cleaned_review = clean_text_simple(user_review)
                
                # Make prediction
                prediction = st.session_state.model.predict([cleaned_review])[0][0]
                
                # Display result
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### Predicted Sentiment: {sentiment}")
                    st.markdown(f"### Confidence: {confidence:.2f}")
                
                with col2:
                    # Emoji based on sentiment
                    emoji = "😊" if sentiment == "Positive" else "😔"
                    st.markdown(f"# {emoji}")
                
                # Display processing steps
                st.subheader("Text Processing Steps")
                
                st.markdown("**Original Review:**")
                st.markdown(f"> {user_review}")
                
                st.markdown("**Cleaned Review:**")
                cleaned = clean_text_simple(user_review)
                st.markdown(f"> {cleaned}")
        
        # Example reviews
        st.subheader("Or Try with Example Reviews")
        
        positive_examples, negative_examples = get_example_reviews()
        
        tab1, tab2 = st.tabs(["Positive Examples", "Negative Examples"])
        
        with tab1:
            for i, example in enumerate(positive_examples, 1):
                if st.button(f"Test Positive Example {i}", key=f"pos_{i}"):
                    with st.spinner("Analyzing sentiment..."):
                        # Use simple text cleaning to avoid NLTK issues
                        cleaned_review = clean_text_simple(example)
                        
                        # Make prediction
                        prediction = st.session_state.model.predict([cleaned_review])[0][0]
                        
                        # Display result
                        sentiment = "Positive" if prediction > 0.5 else "Negative"
                        confidence = prediction if prediction > 0.5 else 1 - prediction
                        
                        st.markdown(f"**Example:** {example}")
                        st.markdown(f"**Predicted Sentiment:** {sentiment}")
                        st.markdown(f"**Confidence:** {confidence:.2f}")
                        st.markdown(f"**Correct?** {'✓' if sentiment == 'Positive' else '✗'}")
        
        with tab2:
            for i, example in enumerate(negative_examples, 1):
                if st.button(f"Test Negative Example {i}", key=f"neg_{i}"):
                    with st.spinner("Analyzing sentiment..."):
                        # Use simple text cleaning to avoid NLTK issues
                        cleaned_review = clean_text_simple(example)
                        
                        # Make prediction
                        prediction = st.session_state.model.predict([cleaned_review])[0][0]
                        
                        # Display result
                        sentiment = "Positive" if prediction > 0.5 else "Negative"
                        confidence = prediction if prediction > 0.5 else 1 - prediction
                        
                        st.markdown(f"**Example:** {example}")
                        st.markdown(f"**Predicted Sentiment:** {sentiment}")
                        st.markdown(f"**Confidence:** {confidence:.2f}")
                        st.markdown(f"**Correct?** {'✓' if sentiment == 'Negative' else '✗'}")
    
    else:
        st.warning("Please train the model first on the 'Model Training' page.")

# Educational Resources Page
elif page == "Educational Resources":
    st.title("📚 Educational Resources")
    
    st.markdown("""
    ## Learning About Neural Networks and Sentiment Analysis
    
    This page provides educational resources to help you understand the concepts behind the sentiment analysis model.
    """)
    
    # Tabs for different topics
    tab1, tab2, tab3, tab4 = st.tabs([
        "Neural Network Basics", 
        "Activation Functions", 
        "Loss Functions", 
        "Text Preprocessing"
    ])
    
    with tab1:
        st.subheader("Neural Network Components")
        
        # Get explanations
        nn_explanations = explain_neural_network_components()
        
        # Display explanations
        for component, explanation in nn_explanations.items():
            st.markdown(f"**{component}:**")
            st.markdown(f"> {explanation}")
            st.markdown("---")
        
        st.subheader("Feedforward Neural Network")
        
        st.markdown("""
        A feedforward neural network is the simplest type of artificial neural network. Information flows in one direction, from input to output.
        
        **Key characteristics:**
        
        1. **Unidirectional flow**: Data moves forward only, no loops/cycles
        2. **Hidden layers**: One or more layers between input and output
        3. **Fully connected**: Typically, each neuron connects to every neuron in the next layer
        4. **Non-linear activation**: Enables learning complex patterns
        
        **How it works:**
        
        1. Input features are fed to the network
        2. Each neuron calculates weighted sum of inputs + bias
        3. Activation function is applied to introduce non-linearity
        4. Output is passed to the next layer
        5. Final layer produces classification/regression output
        """)
        
        # Add a simple diagram
        st.image("https://www.researchgate.net/publication/344602060/figure/fig1/AS:945454449254400@1602387510803/The-architecture-of-feedforward-neural-network.png", 
                caption="Feedforward Neural Network Architecture", 
                use_column_width=True)
    
    with tab2:
        st.subheader("Activation Functions")
        
        # Plot activation functions
        st.markdown("### Visual Comparison of Common Activation Functions")
        fig_activations = plot_activation_functions()
        st.plotly_chart(fig_activations, use_container_width=True)
        
        # Display explanations
        st.markdown("### Activation Functions Explained")
        
        activations = explain_activation_functions()
        
        for activation, details in activations.items():
            st.markdown(f"**{activation}**")
            st.markdown(f"**Formula:** {details['formula']}")
            st.markdown(f"**Description:** {details['description']}")
            
            st.markdown("**Advantages:**")
            for adv in details['advantages']:
                st.markdown(f"- {adv}")
            
            st.markdown("**Disadvantages:**")
            for dis in details['disadvantages']:
                st.markdown(f"- {dis}")
            
            st.markdown("---")
    
    with tab3:
        st.subheader("Cross-Entropy Loss Function")
        
        # Display explanation
        cross_entropy = explain_cross_entropy()
        
        st.markdown(f"**Formula:** {cross_entropy['formula']}")
        st.markdown(f"**Description:** {cross_entropy['description']}")
        st.markdown(f"**Binary Classification Case:** {cross_entropy['binary_case']}")
        st.markdown(f"**Why It's Used:** {cross_entropy['why_used']}")
        
        # Plot binary cross-entropy
        st.markdown("### Binary Cross-Entropy Loss Visualization")
        fig_ce = plot_binary_cross_entropy()
        st.plotly_chart(fig_ce, use_container_width=True)
        
        # Example calculation
        st.markdown("### Example Cross-Entropy Calculation")
        df_ce, mean_loss = display_cross_entropy_calculation()
        
        st.dataframe(df_ce)
        st.markdown(f"**Mean Binary Cross-Entropy Loss:** {mean_loss:.4f}")
        
        st.markdown("""
        #### Interpretation:
        
        - When the true label is 1, the loss is -log(p)
        - When the true label is 0, the loss is -log(1-p)
        - Loss increases as predictions diverge from true labels
        - Perfect predictions would have zero loss
        - Incorrect predictions with high confidence are penalized heavily
        """)
    
    with tab4:
        st.subheader("Text Preprocessing for Sentiment Analysis")
        
        st.markdown("""
        Text preprocessing is crucial for effective sentiment analysis. It transforms raw text into a format suitable for machine learning algorithms.
        
        ### Preprocessing Steps
        
        1. **Text Cleaning**
           - Remove HTML tags and special characters
           - Convert to lowercase
           - Remove numbers and punctuation
        
        2. **Tokenization**
           - Split text into individual words (tokens)
           - Example: "I loved this movie" → ["I", "loved", "this", "movie"]
        
        3. **Stop Word Removal**
           - Remove common words that add little meaning (e.g., "the", "is", "and")
           - Reduces dimensionality and focuses on important words
        
        4. **Lemmatization/Stemming**
           - Reduce words to their base or root form
           - Example: "loving", "loved", "loves" → "love"
        
        5. **Vectorization**
           - Convert text to numerical features
           - Methods: Bag-of-Words, TF-IDF, Word Embeddings
        
        6. **Sequence Creation**
           - Create fixed-length sequences for neural networks
           - Pad shorter sequences, truncate longer ones
        """)
        
        # Show example
        st.markdown("### Preprocessing Example")
        
        example_text = "I really loved this movie! The acting was amazing and the plot kept me engaged throughout. Definitely a 10/10 experience. <br><br> Would watch again!"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Text:**")
            st.markdown(f"> {example_text}")
        
        with col2:
            st.markdown("**Processed Text:**")
            st.markdown(f"> {clean_text_simple(example_text)}")

# Main app execution
if __name__ == "__main__":
    # Check if model file exists
    if os.path.exists("sentiment_model.pkl") and not st.session_state.trained:
        st.sidebar.info("Pre-trained model found. You can load it on the Model Training page.")

# Add a footer
st.markdown("""
---
### 📝 IMDB Sentiment Analysis using Deep Learning

Developed as a demonstration of deep learning techniques for NLP tasks.
""")
